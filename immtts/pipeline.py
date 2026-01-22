from einops import rearrange
import random
import torch
import torch.nn.functional as F
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
from transformers import ClapModel, ClapProcessor, SpeechT5HifiGan
from diffusers import AutoencoderKL
from torchinfo import summary

from immtts.models.model import ImmersiveTTS
from immtts.text_utils.en import TxtProcessor
from immtts.text_utils.text_encoder import build_token_encoder
from immtts.data.data_utils import intersperse


def load_ckpt(model, ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"]
    msg = model.load_state_dict(state_dict, strict=True)
    print(msg)
    return model


def raw_text_to_phoneme_tokens(raw_text, token_list_file):
    processed_text = TxtProcessor.preprocess_text(raw_text)
    phoneme_list = TxtProcessor.g2p(processed_text)
    if phoneme_list and phoneme_list[-1] == " ":
        phoneme_list = phoneme_list[:-1]

    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>" 
    phoneme_str = " ".join([BOS_TOKEN] + phoneme_list + [EOS_TOKEN])

    token_encoder = build_token_encoder(token_list_file)
    token_ids = token_encoder.encode(phoneme_str)
    token_ids = intersperse(token_ids, 80)
    token_ids = torch.IntTensor(token_ids)

    return token_ids



class ImmersivettsPipeline():
    def __init__(self, model_config = None, ckpt_path = None, device = None, spk_embedding_prepared = False):
        
        self.device = device
        self.spk_embedding_prepared = spk_embedding_prepared
        if not self.spk_embedding_prepared:
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-base-sv')
            self.wavlm_encoder = WavLMForXVector.from_pretrained('microsoft/wavlm-base-sv').eval()
        else:
            self.feature_extractor = None
            self.wavlm_encoder = None

        self.vae = AutoencoderKL.from_pretrained("cvssp/audioldm2", subfolder="vae")
        self.vocoder = SpeechT5HifiGan.from_pretrained("cvssp/audioldm2", subfolder="vocoder").eval()
        self.clap_model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
        self.clap_processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused") 
        self.vae.to(self.device)
        self.vocoder.to(self.device)
        self.clap_model.to(self.device)
        self.vae.eval()
        self.vocoder.eval()    
        self.clap_model.eval()

        self.model = load_ckpt(ImmersiveTTS(config = model_config), ckpt_path)
        self.model.to(self.device) 
        self.model.eval()
        summary(self.model)


    def generate(self, 
                 cont_prompt: str,
                 speaker_prompt = None,
                 env_prompt: str = None,
                 num_samples_per_prompt: int = 1,
                 num_inference_steps: int = 50,
                 duration: int = 10,
                 cont_guidance_scale: float = None,
                 env_guidance_scale: float = None,
                 save_dir: str = None,
                 file_id: str = None,
                 **kwargs,
        ):

        phone_set_path = '/your/path/to/phone_set.json'
        cont_tokens = raw_text_to_phoneme_tokens(cont_prompt, phone_set_path).to(self.device)
        if cont_tokens.dim() == 1:
            cont_tokens = cont_tokens.unsqueeze(0)
        cont_lengths = torch.LongTensor([cont_tokens.size(1)]).to(self.device)

        if self.spk_embedding_prepared:
            spk_embeddings = speaker_prompt.to(self.device)
        else:
            if speaker_prompt is not None:
                three_sec_samples = 3 * 16000  
                feature_waveform = speaker_prompt.clone()
                if feature_waveform.shape[1] > three_sec_samples:
                    start_idx = random.randint(0, feature_waveform.shape[1] - three_sec_samples)
                    feature_waveform = feature_waveform[:, start_idx:start_idx + three_sec_samples]
                else:
                    padding_length = three_sec_samples - feature_waveform.shape[1]
                    feature_waveform = F.pad(feature_waveform, (0, padding_length))

                with torch.no_grad():
                    wavlm_inputs = self.feature_extractor(
                        feature_waveform.squeeze().numpy(),
                        sampling_rate=16000,
                        return_tensors="pt"
                    )
                    spk_embeddings = self.wavlm_encoder(**wavlm_inputs).embeddings
                    spk_embeddings = spk_embeddings.to(self.device)
            else:
                spk_embeddings = torch.zeros(1, 512).to(self.device)
                        
        clap_inputs = self.clap_processor(text=env_prompt, padding=True, return_tensors="pt")    
        uc_clap_inputs = self.clap_processor(text=[""], padding=True, return_tensors="pt")
        with torch.no_grad():
            env_clap_features = self.clap_model.get_text_features(
                input_ids=clap_inputs["input_ids"].to(self.device),
                attention_mask=clap_inputs["attention_mask"].to(self.device)
            )
            uc_clap_features = self.clap_model.get_text_features(
                input_ids=uc_clap_inputs["input_ids"].to(self.device),
                attention_mask=uc_clap_inputs["attention_mask"].to(self.device)
            )

        with torch.no_grad():
            latents = self.model.inference_flow(
                cont_prompt=cont_tokens,
                cont_length=cont_lengths,
                spk_embed=spk_embeddings,
                env_prompt=env_prompt,
                env_clap_features=env_clap_features,
                num_inference_steps=num_inference_steps,
                cont_guidance_scale=cont_guidance_scale,
                env_guidance_scale=env_guidance_scale,
                duration=duration,
                num_samples_per_prompt=num_samples_per_prompt,
                save_dir=save_dir,
                file_id=file_id,  
                uncond_embed=uc_clap_features,
            )
            
        latents = rearrange(latents,
                    "b (h w) (c ph pw) -> b c (h ph) (w pw)",
                    w=8, ph=2, pw=2,
                )

        mel_spectrogram = self.decode_latents(latents)
        audio = self.mel_spectrogram_to_waveform(mel_spectrogram)
        audio = self.normalize_wav(audio)

        return audio

    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        mel_spectrogram = self.vae.decode(latents).sample
        return mel_spectrogram

    def mel_spectrogram_to_waveform(self, mel_spectrogram):
        if mel_spectrogram.dim() == 4:
            mel_spectrogram = mel_spectrogram.squeeze(1)

        waveform = self.vocoder(mel_spectrogram)
        waveform = waveform.cpu().float()
        return waveform

    def normalize_wav(self, waveform):
        waveform = waveform - torch.mean(waveform)
        waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)
        return waveform
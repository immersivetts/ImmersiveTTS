import random
import torch
import torch.nn.functional as F
import torchaudio
import numpy as np

from torch.utils.data import Dataset
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
from immtts.audio import raw_waveform_to_fbank, TacotronSTFT
from immtts.data.data_utils import intersperse, collate_1d_or_2d, position_based_augmentation

class AudioDataset(Dataset):
    def __init__(self, args, df, df_noise, clap_processor):
        self.args = args
        self.df = df
        self.df_noise = df_noise
        self.clap_processor = clap_processor
        self.limit_dur = 15.1
        
        self.uncond_env_prob = args.uncond_env_prob
        self.uncond_cont_prob = args.uncond_cont_prob
        self.add_noise_prob = args.add_noise_prob

        self.max_text_seq_len = 64
        self.hidden_dim = 1024
        
        self.position_options = ['front', 'near front', 'middle', 'near back', 'back']
        self.clean_prompts = {
                1: "A person is speaking in a silent environment.",
                2: "A Human is talking without any background noise.",
                3: "A person is talking in a controlled, quiet recording space.",
                4: "A clean voice recording made in a silent environment.",
                5: "A human voice is recorded in a quiet room.",
                6: "A person is speaking in an acoustically treated space.",
                7: "A human is speaking in a perfectly quiet and stable environment.",
                8: "A person is talking in an environment with zero ambient noise.",
                9: "A human is speaking in the room.",
                10: "A speaker is talking in a room designed for clear audio capture."
            }

        self.stft = TacotronSTFT(
            filter_length=1024,
            hop_length=160,
            win_length=1024,
            n_mel_channels=64,
            sampling_rate=16000,
            mel_fmin=0,
            mel_fmax=8000,
        )
        
        if args.spk_embedding_prepared:
            self.feature_extractor = None
            self.wavlm_encoder = None
        else:
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-base-sv')
            self.wavlm_encoder = WavLMForXVector.from_pretrained('microsoft/wavlm-base-sv').eval()
            for p in self.wavlm_encoder.parameters():
                p.requires_grad = False

    def get_mel(self, audio, _stft):
        audio = torch.clip(torch.FloatTensor(audio).unsqueeze(0), -1, 1)
        audio = torch.autograd.Variable(audio, requires_grad=False)
        melspec, _, _ = _stft.mel_spectrogram(audio)
        melspec = torch.squeeze(melspec, 0).float()
        return melspec.T 
                
    def __getitem__(self, index):
        row = self.df.iloc[index]
        item_name = row["item_name"]
        file_path = row["wav_fn"]
        speech_dur = row["duration"]

        if speech_dur > self.limit_dur :
            return None

        try:
            clean_wav, sr = torchaudio.load(file_path)
            
            if sr != 16000: 
                clean_wav = torchaudio.functional.resample(clean_wav, orig_freq=sr, new_freq=16000)
            if clean_wav.shape[0] > 1:
                clean_wav = torch.mean(clean_wav, dim=0, keepdim=True)
            
        except Exception as e:
            print(f"[ERROR] Failed to load clean audio file: {file_path}, idx={index}")
            raise e 

        clean_fbank = self.get_mel(clean_wav[0], self.stft)


        if self.args.spk_embedding_prepared:
            spk_emb = row["spk_embedding"]
            spk_embed  = torch.load(spk_emb, map_location="cpu", weights_only=True)
            if spk_embed.dim() == 1:
                spk_embed = spk_embed.unsqueeze(0)
        else:
            three_sec_samples = 3 * 16000  
            feature_wav = clean_wav.clone().detach()
            if feature_wav.shape[1] > three_sec_samples:
                start_idx = random.randint(0, feature_wav.shape[1] - three_sec_samples)
                feature_wav = feature_wav[:, start_idx:start_idx + three_sec_samples]
            else:
                padding_length = three_sec_samples - feature_wav.shape[1]
                feature_wav = F.pad(feature_wav, (0, padding_length))
                
            with torch.no_grad():
                wavlm_inputs = self.feature_extractor(
                    feature_wav.squeeze().numpy(),
                    sampling_rate=16000,
                    return_tensors="pt",
                )
                spk_embed = self.wavlm_encoder(**wavlm_inputs).embeddings.detach()

        aug_wav = clean_wav.clone().detach()
        add_duration = random.randint(1, 3) 
        target_dur = speech_dur + add_duration
        aug_prob = random.random()
        
        if aug_prob < self.add_noise_prob:  
            aug_type = 'noise'
            noise_row = self.df_noise.iloc[random.randint(0, len(self.df_noise)-1)]
            text_prompt = noise_row["label"]
            noise_file = noise_row["wav_fn"]
            
            try:
                noise, noise_sr = torchaudio.load(noise_file)
                
                if noise_sr != 16000: # resample to 16000 Hz
                    noise = torchaudio.functional.resample(noise, orig_freq=noise_sr, new_freq=16000)
                if noise.shape[0] > 1: # stereo to mono
                    noise = torch.mean(noise, dim=0, keepdim=True)
                    
            except Exception as e:
                print(f"[ERROR] Failed to load noise audio file: {noise_file}, idx={index}")
                raise e 
            
            position = random.choice(self.position_options)
            aug_success, aug_wav, front_samples, back_samples = position_based_augmentation(
                sr=16000,
                speech=aug_wav, 
                noise=noise, 
                position=position, 
                target_dur=target_dur, 
                speech_dur=speech_dur
            )
            
            if not aug_success:
                prompt_num= random.randint(1, len(self.clean_prompts))
                text_prompt = self.clean_prompts[prompt_num]
                front_samples = 0
                back_samples = 0
                 
        else: 
            prompt_num= random.randint(1, len(self.clean_prompts))
            text_prompt = self.clean_prompts[prompt_num]
            front_samples = 0
            back_samples = 0  
  
        fbank, _, waveform_after_stft = raw_waveform_to_fbank(
            aug_wav[0],  
            target_length=None,
            fn_STFT=self.stft
        )

        # for CFG dropout (cont prompt)
        if random.random() < self.uncond_cont_prob:
            ph_token = torch.LongTensor([8,1]) 
            content_prompt = ""
        else:
            ph_token = torch.LongTensor(row["ph_token"])
            ph_token = intersperse(ph_token, 80)
            ph_token = torch.IntTensor(ph_token)
            content_prompt = row["txt"]
        # for CFG dropout (env prompt)
        if random.random() < self.uncond_env_prob:
            text_prompt = ""        
            
        # resample to 48k for CLAP
        aug_wav_48k = torchaudio.functional.resample(aug_wav, orig_freq=16000, new_freq=48000)
        
        return (
            fbank, clean_fbank, ph_token, content_prompt, spk_embed, text_prompt, 
            front_samples, back_samples, aug_wav, clean_wav, aug_wav_48k
        )
        
    def __len__(self):
        return len(self.df)


class CollateFn:
    def __init__(self, text_processor = None, clap_processor = None,  clap_type = "audio", wavlm_ft_extractor = None, expected_batch = 8):
        self.text_processor = text_processor
        self.clap_processor = clap_processor
        self.clap_type = clap_type
        self.wavlm_ft_extractor = wavlm_ft_extractor
        self.expected_batch = expected_batch
        
    def __call__(self, examples):
        examples = [e for e in examples if e is not None]
        
        if len(examples) != self.expected_batch:
            print(f"[ERROR] got: {len(examples)} samples")
            return None
        
        batch = {
            "fbank": collate_1d_or_2d([ex[0] for ex in examples], pad_idx=0.0),
            "clean_fbank": collate_1d_or_2d([ex[1] for ex in examples], pad_idx=0.0),
            "content_tokens": collate_1d_or_2d([ex[2] for ex in examples], pad_idx=0),
        }
        batch["mel_lengths"] = torch.LongTensor([ex[0].shape[0] for ex in examples])
        batch["clean_mel_lengths"] = torch.LongTensor([ex[1].shape[0] for ex in examples])
        batch["cont_lengths"] = torch.LongTensor([ex[2].numel() for ex in examples])
        
        batch["spk_embeddings"] = torch.cat([ex[4] for ex in examples]).detach()
        batch["text_prompts"] = [ex[5] for ex in examples]
        batch["front_samples"] = [ex[6] for ex in examples]
        batch["back_samples"] = [ex[7] for ex in examples]
        
        # for REPA (USAD, ATST ver.)
        batch["aug_wav"] = collate_1d_or_2d([ex[8].squeeze(0) for ex in examples], pad_idx=0.0)
        
        # for REPA (WavLM ver.)
        if self.wavlm_ft_extractor is not None:
            wavs = [ex[9].squeeze(0).cpu().numpy().astype("float32") for ex in examples]
            inputs = self.wavlm_ft_extractor(
                wavs, sampling_rate=16000, padding=True, return_tensors="pt"
            )
            batch["wavlm_inputs"] = inputs

        # for CLAP
        if self.clap_type == "text":
            clap_inputs = self.clap_processor(text=[ex[5] for ex in examples], padding=True, return_tensors="pt")
            batch.update(clap_inputs)    
        elif self.clap_type == "audio":
            MAX = 48_000 * 10 

            def _prep(w):
                w = w.squeeze(0).cpu().float().numpy()
                return w[:MAX] if len(w) > MAX else np.pad(w, (0, MAX-len(w)))

            wavs = [_prep(ex[9]) for ex in examples]   # 48kHz for CLAP audio
            clap_inputs = self.clap_processor(
                audios=wavs, sampling_rate=48000, return_tensors="pt"
            )
            batch["input_features"] = clap_inputs.input_features   # (B,1,80,T)
            batch["clap_is_longer"] = clap_inputs.is_longer        # (B,1)
                        
        return batch
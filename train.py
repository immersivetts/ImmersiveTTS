import os
import sys
import math
import yaml
import argparse
import shutil
import pandas as pd
import torch
import torch.distributed as dist
import datetime

from tqdm.auto import tqdm
from diffusers import AutoencoderKL
from diffusers.optimization import get_scheduler
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, DistributedDataParallelKwargs
from torchinfo import summary

from transformers import ClapModel, ClapProcessor, Wav2Vec2FeatureExtractor, WavLMModel, AutoModel
from immtts.data.data import AudioDataset, CollateFn
from immtts.models.model import ImmersiveTTS

sys.path.append("/your/path/to/audiossl")
from audiossl.methods.atstframe.embedding import load_model,get_timestamp_embedding
import audiossl.methods.atstframe.embedding as atst_emb
atst_emb.N_BLOCKS = 1 

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", default=None, type=str, help="Path to configuration file (overrides all other arguments when provided)")
    parser.add_argument("--output_dir", default="outputs", help="Directory name to save output files")
    parser.add_argument("--load_from_ckpt_path", default=None, help="Path to load checkpoint from")
    parser.add_argument("--checkpoints_total_limit", type=int, default=2, help="Total limit of checkpoints")
    parser.add_argument("--mixed_precision", default=None, help="'fp16' to enable mixed precision training")
    parser.add_argument("--max_train_steps", type=int, default=1000000)
    parser.add_argument("--num_train_epochs", type=int, default=None)
    parser.add_argument("--checkpointing_steps", type=int, default=20000)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--dataloader_num_workers", type=int, default=8)
    parser.add_argument("--lr_scheduler", default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--uncond_env_prob", type=float, default=0.1, help="prob. to drop environment condition")
    parser.add_argument("--uncond_cont_prob", type=float, default=0.1, help="prob. to drop content condition")
    args = parser.parse_args()

    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            for key, value in config.items():
                setattr(args, key, value)
    return args     

def main():

    args = parse_args()
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        automatic_checkpoint_naming=True,
        total_limit=args.checkpoints_total_limit,
    )
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs],
    )

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "config.yaml"), "w") as f:
            yaml.dump(vars(args), f)

    if torch.distributed.is_available() and not torch.distributed.is_initialized():
        dist.init_process_group(backend='nccl', timeout=datetime.timedelta(seconds=6000))

    vae = AutoencoderKL.from_pretrained("cvssp/audioldm2", subfolder="vae")
    clap_model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
    clap_processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
    uncond_embed = torch.load('/your/path/to/uncond_embed.pt')
    
    for name, param in vae.named_parameters():
        param.requires_grad = False
    vae.eval()

    for name, param in clap_model.named_parameters():
        param.requires_grad = False
    clap_model.eval()
    
    if 'wavlm' in args.model['encoder_type']:
        wavlm_ft_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-large")
        wavlm_model = WavLMModel.from_pretrained("microsoft/wavlm-large").eval()
        for name, param in wavlm_model.named_parameters():
            param.requires_grad = False
        wavlm_model.eval()
    else:
        wavlm_ft_extractor = None
        wavlm_model = None
    
    if 'usad' in args.model['encoder_type']:
        usad_model = AutoModel.from_pretrained("MIT-SLS/USAD-Base", trust_remote_code=True).eval()
        for name, param in usad_model.named_parameters():
            param.requires_grad = False
        usad_model.eval()
    else:
        usad_model = None

    if 'atst-frame' in args.model['encoder_type']:
        atst_model = load_model('/your/path/to/audiossl/atstframe_base.ckpt').eval()
        for name, param in atst_model.named_parameters():
            param.requires_grad = False
    else:
        atst_model = None

    model = ImmersiveTTS(config=args.model) 
    if accelerator.is_main_process:
        summary(model)

    json_paths = ['libri_tts_path']
    noise_json_paths = ['as_nonspeech_path', 'bbc_sound_effects_path', 'sound_bible_path', 'free_sound_path']
    speech_jsons = [pd.read_json(getattr(args, path)) for path in json_paths if hasattr(args, path) and getattr(args, path)]    
    noise_jsons = [pd.DataFrame(df['data'].tolist()) for df in 
                  [pd.read_json(getattr(args, path)) for path in noise_json_paths if hasattr(args, path) and getattr(args, path)]]
    
    df = pd.concat(speech_jsons, ignore_index=True)
    df_noise = pd.concat(noise_jsons, ignore_index=True)
    
    train_dataset = AudioDataset(args, df, df_noise, clap_processor)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=CollateFn(text_processor = None, 
                             clap_processor = clap_processor, 
                             clap_type = args.model['clap_type'], 
                             wavlm_ft_extractor = wavlm_ft_extractor,
                             expected_batch = args.train_batch_size),
                             batch_size=args.train_batch_size,
                            num_workers=args.dataloader_num_workers,
    )

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    clap_model.to(accelerator.device, dtype=weight_dtype)

    if wavlm_model is not None:
        wavlm_model.to(accelerator.device, dtype=weight_dtype)
    if usad_model is not None:
        usad_model.to(accelerator.device, dtype=weight_dtype)
    if atst_model is not None:
        atst_model.to(accelerator.device, dtype=weight_dtype)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    global_step = 0
    first_epoch = 0
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            if batch is None:
                print('warning: batch is None')
                continue

            mel = batch['fbank']
            clean_mel = batch['clean_fbank']
            cont_embed = batch['content_tokens'] # [bsz, length]
            cont_length = batch['cont_lengths']
            mel_lengths = batch['mel_lengths']
            clean_mel_lengths = batch['clean_mel_lengths']
            spk_embed = batch['spk_embeddings']
            env_prompt = batch['text_prompts']
            front_len = batch['front_samples']
            back_len = batch['back_samples']

            if args.model['clap_type'] == "audio":
                with torch.no_grad():
                    audio_outputs = clap_model.audio_model(
                        input_features=batch['input_features'].to(weight_dtype),
                        is_longer=batch['clap_is_longer'],
                    )
                    audio_embeds = audio_outputs.pooler_output
                    env_clap_features = clap_model.audio_projection(audio_embeds)
                env_clap_features[torch.rand(env_clap_features.size(0)) < args.uncond_env_prob] = uncond_embed.to(env_clap_features.device, dtype=weight_dtype)
            
            elif args.model['clap_type'] == "text":
                with torch.no_grad():
                    env_clap_features = clap_model.get_text_features(
                        input_ids=batch["input_ids"].to(accelerator.device),
                        attention_mask=batch["attention_mask"].to(accelerator.device),
                    )
           
            # REPA
            if args.model['repa']:
                with torch.no_grad():
                    gt_zs = []
                    gt_zs_masks = []
                    
                    if 'wavlm' in args.model['encoder_type']:
                        layer_idx = 24
                        wavlm_inputs = batch['wavlm_inputs']
                        wavlm_outputs = wavlm_model(
                            **wavlm_inputs,
                            output_hidden_states=True,
                            return_dict=True
                        )
                        all_tokens = wavlm_outputs.hidden_states[layer_idx] 
                        wavlm_mask = wavlm_inputs.attention_mask
                        gt_zs.append(all_tokens)
                        gt_zs_masks.append(wavlm_mask)
                    else:
                        gt_zs.append(None)
                        gt_zs_masks.append(None)
                        
                    if 'usad' in args.model['encoder_type']:
                        layer_idx = 12
                        waveform = batch['aug_wav'].to(weight_dtype)
                        usad_outputs = usad_model(waveform)                 
                        gt_zs.append(usad_outputs['hidden_states'][layer_idx-1])
                    else:
                        gt_zs.append(None)
                        
                    if 'atst-frame' in args.model['encoder_type']:
                        waveform = batch['aug_wav'].to(weight_dtype)
                        emb_timestamp,t = get_timestamp_embedding(waveform,atst_model)
                        gt_zs.append(emb_timestamp)
                    else:
                        gt_zs.append(None)
                        
            else:
                gt_zs = None
                gt_zs_masks = None
                args.model['encoder_type'] = None
                
            with accelerator.accumulate(model):
                with torch.no_grad():
                    latents = vae.encode(mel.unsqueeze(1).to(weight_dtype)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor    
                loss, fm_loss, duration_loss, prior_loss, proj_loss = model(
                                                                        latents=latents, 
                                                                        cont_prompt=cont_embed, cont_length=cont_length, 
                                                                        mel=mel, mel_lengths=mel_lengths, 
                                                                        clean_mel=clean_mel, clean_mel_lengths=clean_mel_lengths,
                                                                        spk_embed=spk_embed, 
                                                                        env_prompt=env_prompt,
                                                                        env_clap_features=env_clap_features,
                                                                        front_len=front_len,
                                                                        back_len=back_len,
                                                                        gt_zs = gt_zs, 
                                                                        )               
            
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                
                # Backpropagate
                accelerator.backward(loss)                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                train_loss = 0.0
                model.global_step = global_step
            
                if global_step % args.checkpointing_steps == 0:
                    print('global_step: ', global_step)
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        if os.path.exists(save_path):
                            shutil.rmtree(save_path)
                        os.makedirs(save_path, exist_ok=True)                                          
                        torch.save({
                            'model_state_dict': unwrapped_model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler is not None else None,
                            'global_step': global_step,
                            'args': args
                        }, os.path.join(save_path, "model.pt"))    
                    accelerator.wait_for_everyone()                                
            logs = {"step_loss": loss.detach().item(), "fm": fm_loss.detach().item(), "dur": duration_loss.detach().item(), "prior": prior_loss.detach().item(), "proj": proj_loss.detach().item()}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

if __name__ == "__main__":
    main()

         
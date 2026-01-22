import torch
import torch.nn.functional as F
import copy
import numpy as np
import math
import inspect

from torch import nn
from typing import List
from tqdm import tqdm
from typing import Optional, Union, List
from math import pi
from einops import rearrange

from immtts.models.glow_text_encoder import GlowTextEncoder
from immtts.models.modules import LatentMapper
from immtts.data.data_utils import sequence_mask, duration_loss, fix_len_compatibility, generate_path
from immtts.models.fluxtransformer import FluxTransformer2DModelWithREPA
import monotonic_align

from diffusers import FlowMatchEulerDiscreteScheduler
from transformers import T5EncoderModel, T5TokenizerFast
from diffusers.training_utils import compute_density_for_timestep_sampling

class REPAManager:
    
    def __init__(self, encoder_types):
        self.encoder_types = encoder_types if encoder_types else []
        self.encoder_to_gt_idx = {'wavlm': 0, 'usad': 1, 'atst-frame': 2}
    
    def create_mappings(self, zs, gt_zs):
        encoder_to_zs = {}
        encoder_to_gt_zs = {}
        zs_idx = 0
        for encoder_name in self.encoder_types:
            if encoder_name in self.encoder_to_gt_idx:
                gt_idx = self.encoder_to_gt_idx[encoder_name]
                if gt_idx < len(gt_zs) and gt_zs[gt_idx] is not None:
                    encoder_to_gt_zs[encoder_name] = gt_zs[gt_idx]
                if zs_idx < len(zs):
                    encoder_to_zs[encoder_name] = zs[zs_idx]
                    zs_idx += 1
        return encoder_to_zs, encoder_to_gt_zs
    
    def is_encoder_active(self, encoder_name):
        return encoder_name in self.encoder_types

def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):

    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"
        )
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class ImmersiveTTS(nn.Module):

    def __init__(self, config,):
        super().__init__()
        self.num_layers = config.get("num_layers", 6)
        self.num_single_layers = config.get("num_single_layers", 18)
        self.in_channels = config.get("in_channels", 64)
        self.attention_head_dim = config.get("attention_head_dim", 128)
        self.joint_attention_dim = config.get("joint_attention_dim", 1024)
        self.num_attention_heads = config.get("num_attention_heads", 8)
        self.audio_seq_len = config.get("audio_seq_len", 645)
        self.max_duration = config.get("max_duration", 30)
        self.uncond_cont_prob = config.get("uncond_cont_prob", 0.1)
        self.uncond_env_prob = config.get("uncond_env_prob", 0.1)
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
        self.noise_scheduler_copy = copy.deepcopy(self.noise_scheduler)
        self.max_text_seq_len = 64
        self.text_encoder_name = config.get("text_encoder_name", "google/flan-t5-large")        
        self.text_encoder = T5EncoderModel.from_pretrained(self.text_encoder_name).eval() 
        self.tokenizer = T5TokenizerFast.from_pretrained(self.text_encoder_name)
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        self.text_embedding_dim = self.text_encoder.config.d_model  
        self.content_encoder = GlowTextEncoder(n_vocab = 81,
                                            n_feats = 64,
                                            n_channels = 192,
                                            filter_channels = 768,
                                            filter_channels_dp = 256,
                                            n_heads = 2,
                                            n_layers = 6,
                                            kernel_size = 3,
                                            p_dropout = 0.1,
                                            window_size = 4,
                                            pre_spk_emb_dim = 512, 
                                            spk_emb_dim = 192,
                                            n_spks = 904,
                                            )
        self.latent_proj = LatentMapper(in_features=self.content_encoder.n_feats, 
                                          out_channels=8, 
                                          kernel_size=3, hidden_channels=16)
        self.fc_clap = nn.Sequential(nn.Linear(512, self.joint_attention_dim), nn.ReLU())
        self.transformer = FluxTransformer2DModelWithREPA(
            in_channels=self.in_channels,
            num_layers=self.num_layers,
            num_single_layers=self.num_single_layers,
            attention_head_dim=self.attention_head_dim,
            num_attention_heads=self.num_attention_heads,
            joint_attention_dim=self.joint_attention_dim,
            pooled_projection_dim=self.text_embedding_dim,
            guidance_embeds=False,
            repa=config.get("repa", True),
            repa_depth=config.get("repa_encoder_depth", 4),
            repa_z_dims=config.get("repa_z_dims", [768]),
            repa_projector_dim=config.get("repa_projector_dim", 2048),
        )
        self.final_layer = nn.Linear(self.in_channels, self.in_channels//2)
        self.register_buffer("content_null_embedding", nn.Parameter(torch.randn(1, 32) * math.sqrt(1.0 / 32)))
        self.repa_manager = REPAManager(config.get("encoder_type", []))
        

    def get_sigmas(self, timesteps, n_dim=3, dtype=torch.float32, device=None):
        sigmas = self.noise_scheduler_copy.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = self.noise_scheduler_copy.timesteps.to(device)
        timesteps = timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
            
        return sigma

    def encode_text_classifier_free(self, prompt: List[str], num_samples_per_prompt=1, guidance="dual"):
        device = self.text_encoder.device
        batch = self.tokenizer(
            prompt,
            max_length=self.tokenizer.model_max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        input_ids, attention_mask = batch.input_ids.to(device), batch.attention_mask.to(device)
        with torch.no_grad():
            prompt_embeds = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
        prompt_embeds = prompt_embeds.repeat_interleave(num_samples_per_prompt, 0)
        attention_mask = attention_mask.repeat_interleave(num_samples_per_prompt, 0)

        uncond_tokens = [""]
        max_length = prompt_embeds.shape[1]
        uncond_batch = self.tokenizer(
            uncond_tokens,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        uncond_input_ids = uncond_batch.input_ids.to(device)
        uncond_attention_mask = uncond_batch.attention_mask.to(device)

        with torch.no_grad():
            negative_prompt_embeds = self.text_encoder(input_ids=uncond_input_ids, attention_mask=uncond_attention_mask)[0]
        negative_prompt_embeds = negative_prompt_embeds.repeat_interleave(num_samples_per_prompt, 0)
        uncond_attention_mask = uncond_attention_mask.repeat_interleave(num_samples_per_prompt, 0)

        if guidance == "dual":
            prompt_embeds = torch.cat([prompt_embeds, prompt_embeds, negative_prompt_embeds, negative_prompt_embeds])
            prompt_mask = torch.cat([attention_mask, attention_mask, uncond_attention_mask, uncond_attention_mask])
        elif guidance == "single":
            prompt_embeds = torch.cat([prompt_embeds, negative_prompt_embeds])
            prompt_mask = torch.cat([attention_mask, uncond_attention_mask])
        boolean_prompt_mask = (prompt_mask == 1).to(device)

        return prompt_embeds, boolean_prompt_mask
    
    def encode_content_glow(self, cont_embed, cont_length, mel, mel_lengths, spk_embed):
        mu_x, logw, x_mask = self.content_encoder(cont_embed, cont_length, spk_embed)
        mel_max_length = mel.shape[-1]
        mel_mask = sequence_mask(mel_lengths, mel_max_length).unsqueeze(1).to(x_mask)
        attn_mask = x_mask.unsqueeze(-1) * mel_mask.unsqueeze(2)

        with torch.no_grad():
            const = -0.5 * math.log(2 * math.pi) * self.content_encoder.n_feats
            factor = -0.5 * torch.ones(mu_x.shape, dtype=mu_x.dtype, device=mu_x.device)
            mel_square = torch.matmul(factor.transpose(1, 2), mel**2)
            mel_mu_double = torch.matmul(2.0 * (factor * mu_x).transpose(1, 2), mel)
            mu_square = torch.sum(factor * (mu_x**2), 1).unsqueeze(-1)
            log_prior = mel_square - mel_mu_double + mu_square + const
            
            attn = monotonic_align.maximum_path(log_prior, attn_mask.squeeze(1))
            attn = attn.detach()

        logw_ = torch.log(1e-8 + torch.sum(attn.unsqueeze(1), -1)) * x_mask
        dur_loss = duration_loss(logw, logw_, cont_length)
  
        mu_mel = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        mu_mel = mu_mel.transpose(1, 2) 

        prior_loss = torch.sum(0.5 * ((mel - mu_mel) ** 2 + math.log(2 * math.pi)) * mel_mask)
        prior_loss = prior_loss / ( torch.sum(mel_mask) * self.content_encoder.n_feats )

        # return mu_mel.transpose(1,2), dur_loss, prior_loss, attn, mel_mask, segment_info, wavlm_features
        return mu_mel.transpose(1,2), dur_loss, prior_loss, attn

    def insert_unvoiced_padding(self, hop_length, mu_mel_mapped, mel_lengths, clean_mel_lengths, front_len, back_len):
        B, L_max, D = mu_mel_mapped.shape
        new_lengths = []
        segmentation_info = []
        
        for i in range(B):
            L_real = min(clean_mel_lengths[i], mu_mel_mapped[i].shape[0])
            p_front = round(front_len[i] / hop_length)
            if mel_lengths is not None:
                p_back = max(0, mel_lengths[i] - p_front - L_real)
            else:
                p_back = round(back_len[i] / hop_length)
            new_len = p_front + L_real + p_back
            new_lengths.append(new_len)
            segmentation_info.append({
                "p_front":  (0, p_front),
                "speech":   (p_front, p_front + L_real),
                "p_back":   (p_front + L_real, p_front + L_real + p_back),
            })
            
        max_len = max(new_lengths)
        padded_mu_mel_mapped = mu_mel_mapped.new_zeros((B, max_len, D))
        
        for i in range(B):
            L_real = min(clean_mel_lengths[i], mu_mel_mapped[i].shape[0])
            p_front = round(front_len[i] / hop_length)
            if mel_lengths is not None:
                p_back = max(0, mel_lengths[i] - p_front - L_real)
            else:
                p_back = round(back_len[i] / hop_length)
            valid_region = mu_mel_mapped[i, :L_real]
            padded_mu_mel_mapped[i, p_front : p_front + L_real] = valid_region

        return padded_mu_mel_mapped, segmentation_info


    @torch.no_grad()
    def encode_text(self, prompt):
        device = self.text_encoder.device
        batch = self.tokenizer(
            prompt,
            max_length=self.max_text_seq_len,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        input_ids, attention_mask = batch.input_ids.to(device), batch.attention_mask.to(
            device
        )
        # print("input_ids:", input_ids)
        encoder_hidden_states = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )[0]
        # print("encoder_hidden_states.shape:", encoder_hidden_states.shape)
        boolean_encoder_mask = (attention_mask == 1).to(device)

        return encoder_hidden_states, boolean_encoder_mask

    @torch.no_grad()
    def inference_flow(self, cont_prompt=None, cont_length=None, spk_embed=None,
        env_prompt = None, env_clap_features = None,
        num_inference_steps=50, timesteps=None,
        cont_guidance_scale=1.0, env_guidance_scale=1.0,
        duration=10, disable_progress=False,
        num_samples_per_prompt=1,
        uncond_embed = None,
        sft = False,
    ):

        device = self.transformer.device
        scheduler = self.noise_scheduler
        target_len = 16000 * duration
        target_feat_len = target_len // 160
        
        if env_guidance_scale is None and cont_guidance_scale is None:
            do_classifier_free_guidance = False
            guidance = None
            bsz = num_samples_per_prompt
        else:
            do_classifier_free_guidance = True
            guidance = "dual"
            bsz = 4 * num_samples_per_prompt
            
        if not isinstance(env_prompt, list): 
            env_prompt = [env_prompt]
            
        length_scale = 1.1
        mu_x, logw, x_mask = self.content_encoder(cont_prompt, cont_length, spk_embed)
        w = torch.exp(logw) * x_mask
        w_ceil = torch.ceil(w) * length_scale
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_max_length = int(y_lengths.max())
        y_max_length_ = fix_len_compatibility(y_max_length)
        y_mask = sequence_mask(y_lengths, y_max_length_).unsqueeze(1).to(x_mask.dtype)
        
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)
        
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2).contiguous(), 
                            mu_x.transpose(1, 2).contiguous())
        mu_mel = mu_y
        mu_mel_mapped = self.latent_proj(mu_mel)

        h = mu_mel_mapped.shape[2]
        if h % 2 != 0:
            pad_h = 2 - (h % 2)
            mu_mel_mapped = F.pad(mu_mel_mapped, (0, 0, 0, pad_h), mode="constant", value=0)
            h = h + pad_h
        mu_mel_mapped = rearrange(mu_mel_mapped, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2).float()

        if do_classifier_free_guidance:
            null_expanded = self.content_null_embedding.expand(mu_mel_mapped.shape[1], -1).unsqueeze(0)
            mu_mel_mapped = torch.cat([mu_mel_mapped, null_expanded]*2, dim=0)
        
        speech_feature_len= mu_mel_mapped.shape[1]

        if speech_feature_len < target_feat_len: 
            if (speech_feature_len // 8) % 2 == 1:  
                padding_left = (target_feat_len - speech_feature_len) // 2 + 4
                padding_right = target_feat_len - speech_feature_len - padding_left 
            else:  
                padding_left = (target_feat_len - speech_feature_len) // 2
                padding_right = target_feat_len - speech_feature_len - padding_left
            mu_mel_mapped = F.pad(mu_mel_mapped, (0, 0,  padding_left, padding_right, 0, 0), mode="constant", value=0) 
        
        elif speech_feature_len >= target_feat_len:
            target_feat_len = speech_feature_len
            target_len = target_feat_len * 160   
                
        encoder_hidden_states, boolean_encoder_mask = self.encode_text_classifier_free(env_prompt, num_samples_per_prompt=num_samples_per_prompt, guidance=guidance)         
        if do_classifier_free_guidance:
            uncond_embed = uncond_embed.to(device)
            env_clap_features = torch.cat([env_clap_features, env_clap_features, uncond_embed, uncond_embed], dim=0)
        pooled_projection = self.fc_clap(env_clap_features)        
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        timesteps, num_inference_steps = retrieve_timesteps(scheduler, num_inference_steps, device, timesteps, sigmas)

        latents = torch.randn(1, mu_mel_mapped.shape[1], mu_mel_mapped.shape[2])      
        txt_ids = torch.zeros(bsz, encoder_hidden_states.shape[1], 3).to(device)
        audio_ids = (torch.arange(latents.shape[1]).unsqueeze(0).unsqueeze(-1).repeat(bsz, 1, 3).to(device))

        timesteps = timesteps.to(device)
        latents = latents.to(device)
        encoder_hidden_states = encoder_hidden_states.to(device)
        mu_mel_mapped = mu_mel_mapped.to(device)

        progress_bar = tqdm(range(num_inference_steps), disable=disable_progress)
        for step_idx, t in enumerate(timesteps):
         
            latents_input = torch.cat([latents] * 4) if do_classifier_free_guidance else latents
            latents_input = torch.cat([latents_input, mu_mel_mapped], dim=2) 

            model_pred = self.transformer(
                hidden_states=latents_input,
                timestep=torch.tensor([t / 1000], device=device),
                guidance=None,
                pooled_projections=pooled_projection,
                encoder_hidden_states=encoder_hidden_states,
                txt_ids=txt_ids,
                img_ids=audio_ids,
                output_hidden_states=self.transformer.repa,
                return_dict=True,
                sft = sft,
            )  
            
            pred_vf = model_pred.sample
            pred_vf = self.final_layer(pred_vf)
            
            if do_classifier_free_guidance:   
                pred_vf = pred_vf.chunk(4, dim=0)
                pred_vf = (
                    pred_vf[0] + 
                    env_guidance_scale * (pred_vf[1] - pred_vf[3]) + 
                    cont_guidance_scale * (pred_vf[2] - pred_vf[3])
                )
        
            latents = scheduler.step(pred_vf, t, latents).prev_sample
            progress_bar.update(1)

        return latents 


    def forward(self, 
                latents, 
                cont_prompt, cont_length, 
                mel, mel_lengths, clean_mel, clean_mel_lengths, 
                spk_embed, 
                env_prompt = None, 
                env_clap_features = None,
                front_len = None,
                back_len = None,
                sft=True,
                out_size = None,
                gt_zs = None,
                ):

        device = latents.device
        bsz, c, h, w = latents.shape
        mu_mel, duration_loss, prior_loss, attn = self.encode_content_glow(cont_prompt, cont_length, clean_mel.transpose(1,2), clean_mel_lengths, spk_embed)
        mu_mel_mapped = self.latent_proj(mu_mel)
        
        if latents.shape[2] % 2 != 0:
            latents_pad_h = 2 - (latents.shape[2] % 2)
            latents = F.pad(latents, (0, 0, 0, latents_pad_h), mode="constant", value=0)
        if mu_mel_mapped.shape[2] % 2 != 0:
            mu_mel_pad_h = 2 - (mu_mel_mapped.shape[2] % 2)
            mu_mel_mapped = F.pad(mu_mel_mapped, (0, 0, 0, mu_mel_pad_h), mode="constant", value=0)
        latents = rearrange(latents, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2).float()
        mu_mel_mapped = rearrange(mu_mel_mapped, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2).float()
        uncond_cont_mask = (torch.rand(mu_mel_mapped.shape[0], device=mu_mel_mapped.device) < self.uncond_cont_prob)
        
        if uncond_cont_mask.any():
            null_expanded = self.content_null_embedding.expand(mu_mel_mapped.shape[1], -1)
            mu_mel_mapped[uncond_cont_mask] = null_expanded.unsqueeze(0).expand(uncond_cont_mask.sum(), -1, -1)
        mu_mel_mapped, seg_info = self.insert_unvoiced_padding(160, mu_mel_mapped, mel_lengths, clean_mel_lengths, front_len, back_len)
        if mu_mel_mapped.size(1) > latents.size(1):
            mu_mel_mapped = mu_mel_mapped[:, : latents.size(1), :]
        elif mu_mel_mapped.size(1) < latents.size(1):
            latents = latents[:, : mu_mel_mapped.size(1), :]

        encoder_hidden_states, boolean_encoder_mask = self.encode_text(env_prompt)      
        pooled_projection = self.fc_clap(env_clap_features)
        txt_ids = torch.zeros(bsz, encoder_hidden_states.shape[1], 3).to(device)
        audio_ids = torch.arange(latents.shape[1]).unsqueeze(0).unsqueeze(-1).repeat(bsz, 1, 3).to(device)

        noise = torch.randn_like(latents)   
        u = compute_density_for_timestep_sampling(
                weighting_scheme="logit_normal",
                batch_size=bsz,
                logit_mean=0,
                logit_std=1,
                mode_scale=None,
            )
        indices = (u * self.noise_scheduler_copy.config.num_train_timesteps).long()
        timesteps = self.noise_scheduler_copy.timesteps[indices].to(device=latents.device)
        sigmas = self.get_sigmas(timesteps, n_dim=latents.ndim, dtype=latents.dtype, device=device)

        noisy_model_input = (1.0 - sigmas) * latents + sigmas * noise
        noisy_model_input = torch.cat([noisy_model_input, mu_mel_mapped], dim=2) 
        
        model_pred = self.transformer(
                hidden_states=noisy_model_input,
                encoder_hidden_states=encoder_hidden_states,
                pooled_projections=pooled_projection,
                img_ids=audio_ids,
                txt_ids=txt_ids,
                guidance=None,
                timestep=timesteps / 1000,
                output_hidden_states=self.transformer.repa,
                return_dict=True,
                sft = sft
            )

        pred_vf = model_pred.sample
        zs      = model_pred.repa_projected

        model_pred = self.final_layer(pred_vf)
            
        # Calculate FM loss
        B, T, C = model_pred.shape   
        target = noise - latents            
        mask_lengths = []
        for i in range(B):
            mask_len = seg_info[i]["p_back"][1]
            mask_lengths.append(mask_len)
        mask_lengths = torch.tensor(mask_lengths, device=device)
        loss_mask = sequence_mask(mask_lengths, T).to(device).unsqueeze(-1) 

        diff = (model_pred - target) ** 2 
        diff = diff * loss_mask
        valid_elements = loss_mask.sum() * C 
        total_mse = diff.sum()
        fm_loss = total_mse / valid_elements
        
        fm_weight = 1.0
        duration_weight = 1.0
        prior_weight = 1.0
        proj_weight = 1.0 
        wavlm_weight = 1.0
        usad_weight = 1.0
        atst_weight = 1.0
            
        if self.transformer.repa and gt_zs is not None:
            wavlm_loss = 0.0
            usad_loss = 0.0
            atst_loss = 0.0
            encoder_to_zs, encoder_to_gt_zs = self.repa_manager.create_mappings(zs, gt_zs)
            if self.repa_manager.is_encoder_active('wavlm') and 'wavlm' in encoder_to_zs and 'wavlm' in encoder_to_gt_zs:
                wavlm_loss += self.compute_repa_loss_wavlm([encoder_to_zs['wavlm']], [encoder_to_gt_zs['wavlm']], seg_info)
            if self.repa_manager.is_encoder_active('usad') and 'usad' in encoder_to_zs and 'usad' in encoder_to_gt_zs:
                usad_loss += self.compute_repa_loss_usad([encoder_to_zs['usad']], [encoder_to_gt_zs['usad']], loss_mask)
            if self.repa_manager.is_encoder_active('atst-frame') and 'atst-frame' in encoder_to_zs and 'atst-frame' in encoder_to_gt_zs:
                atst_loss += self.compute_repa_loss_atst([encoder_to_zs['atst-frame']], [encoder_to_gt_zs['atst-frame']], loss_mask)
            proj_loss = wavlm_weight * wavlm_loss + usad_weight * usad_loss + atst_weight * atst_loss
            
        else:
            proj_loss = torch.tensor(0.0, device=device)
                            
        loss = fm_weight * fm_loss + duration_weight * duration_loss + prior_weight * prior_loss + proj_weight * proj_loss
            
        return loss, fm_loss, duration_loss, prior_loss, proj_loss


    def mean_flat(self, x):
        return torch.mean(x, dim=list(range(1, len(x.size()))))


    def compute_repa_loss_wavlm(self, zs, gt_zs, seg_info=None):
        if not zs or not gt_zs or len(zs) == 0 or len(gt_zs) == 0:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        proj_loss = 0.0
        bsz = gt_zs[0].shape[0]
        
        for i, (z_tilde, z) in enumerate(zip(zs, gt_zs)):
            
            if seg_info is not None:
                z_tilde_speech_list = []
                for j in range(bsz):
                    speech_start, speech_end = seg_info[j]["speech"]
                    z_tilde_speech = z_tilde[j, speech_start:speech_end, :]  
                    z_tilde_speech_list.append(z_tilde_speech)
                max_speech_len = max(z_speech.shape[0] for z_speech in z_tilde_speech_list)
                z_tilde_padded = torch.zeros(bsz, max_speech_len, z_tilde.shape[-1], device=z_tilde.device, dtype=z_tilde.dtype)           
                for j, z_speech in enumerate(z_tilde_speech_list):
                    z_tilde_padded[j, :z_speech.shape[0], :] = z_speech
                z_tilde = z_tilde_padded

            if z_tilde.shape[1] != z.shape[1]:
                z = F.interpolate(z.transpose(1, 2), size=z_tilde.shape[1], mode="linear", align_corners=False).transpose(1, 2)         
                
            for j, (z_tilde_j, z_j) in enumerate(zip(z_tilde, z)):
                z_tilde_j = F.normalize(z_tilde_j, dim=-1) 
                z_j = F.normalize(z_j, dim=-1) 
                cosine_sim = (z_j * z_tilde_j).sum(dim=-1)
                proj_loss += self.mean_flat(-cosine_sim)
        
        proj_loss /= (len(zs) * bsz)

        return proj_loss


    def compute_repa_loss_usad(self, z_tilde, z, valid_mask=None):
        if not z_tilde or not z or len(z_tilde) == 0 or len(z) == 0:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        proj_loss = 0.0
        bsz = z[0].shape[0]
        if isinstance(z_tilde, (list, tuple)): z_tilde = z_tilde[0]
        if isinstance(z,  (list, tuple)): z  = z[0]

        if z_tilde.shape[1] != z.shape[1]:
            z = F.interpolate(z.transpose(1, 2), size=z_tilde.shape[1], mode="linear", align_corners=False).transpose(1, 2)
        z_tilde = F.normalize(z_tilde, dim=-1) 
        z = F.normalize(z, dim=-1) 

        cosine_sim = (z * z_tilde).sum(dim=-1)
        if valid_mask is not None:
            vm = valid_mask.to(cosine_sim.device)
            if vm.dim() == 3:
                vm = vm.squeeze(-1)
            if vm.shape[1] != cosine_sim.shape[1]:
                L = cosine_sim.shape[1]
                vm = vm[:, :L] if vm.shape[1] > L else F.pad(vm, (0, L - vm.shape[1]), value=0)
            vm = vm.bool()
            proj_loss = -(cosine_sim * vm).sum() / vm.sum().clamp_min(1)  
        else:
            proj_loss = -cosine_sim.mean()

        return proj_loss


    def compute_repa_loss_atst(self, z_tilde, z, valid_mask=None):
        if not z_tilde or not z or len(z_tilde) == 0 or len(z) == 0:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        proj_loss = 0.0
        bsz = z[0].shape[0]
        if isinstance(z_tilde, (list, tuple)): z_tilde = z_tilde[0]
        if isinstance(z,  (list, tuple)): z  = z[0]

        if z_tilde.shape[1] != z.shape[1]:
            z = F.interpolate(z.transpose(1, 2), size=z_tilde.shape[1], mode="linear", align_corners=False).transpose(1, 2)
        z_tilde = F.normalize(z_tilde, dim=-1) 
        z = F.normalize(z, dim=-1) 
            
        cosine_sim = (z * z_tilde).sum(dim=-1)
        if valid_mask is not None:
            vm = valid_mask.to(cosine_sim.device)
            if vm.dim() == 3:
                vm = vm.squeeze(-1)
            if vm.shape[1] != cosine_sim.shape[1]:
                L = cosine_sim.shape[1]
                vm = vm[:, :L] if vm.shape[1] > L else F.pad(vm, (0, L - vm.shape[1]), value=0)
            vm = vm.bool()
            proj_loss = -(cosine_sim * vm).sum() / vm.sum().clamp_min(1)  
        else:
            proj_loss = -cosine_sim.mean()

        return proj_loss

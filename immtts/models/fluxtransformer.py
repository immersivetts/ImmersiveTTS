from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import math
import copy
import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

from diffusers import FluxTransformer2DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput

def build_mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim), nn.SiLU(),
        nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        nn.Linear(hidden_dim, out_dim),
    )
    
@dataclass
class Transformer2DModelOutputWithREPA(Transformer2DModelOutput):
    hidden_states: Optional[torch.Tensor] = None                  # (B, L, D_int)
    repa_projected: Optional[Tuple[torch.Tensor, ...]] = None     # tuple of projector outputs
    
class FluxTransformer2DModelWithREPA(FluxTransformer2DModel):
    def __init__(
        self,
        *model_args,
        repa: bool = True,
        repa_depth: Iterable[int] = (4,),
        repa_z_dims: Iterable[int] = (768,),
        repa_projector_dim: int = 2048,
        **model_kwargs,
    ) -> None:
        super().__init__(*model_args, **model_kwargs)
        self.repa = repa
        self.repa_depth = repa_depth
        if repa:
            self.projectors = nn.ModuleList([
                build_mlp(self.inner_dim, repa_projector_dim, z) for z in repa_z_dims
            ])

    def forward(self, *, output_hidden_states: bool=False, detach_hidden: bool=False, sft: bool=True, **kwargs):
        hidden_states = kwargs.pop("hidden_states")
        encoder_hidden_states = kwargs.pop("encoder_hidden_states")
        pooled_projections = kwargs.pop("pooled_projections")
        timestep = kwargs.pop("timestep")
        img_ids = kwargs.pop("img_ids")
        txt_ids = kwargs.pop("txt_ids")
        guidance = kwargs.pop("guidance", None)
        joint_attention_kwargs = kwargs.pop("joint_attention_kwargs", None)
        return_dict = kwargs.pop("return_dict", True)
        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)
        
        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        
        ids = torch.cat((txt_ids, img_ids), dim=1)
        image_rotary_emb = self.pos_embed(ids)

        repa_hiddens = [] 

        # dual blocks
        for i, block in enumerate(self.transformer_blocks):
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                # joint_attention_kwargs=joint_attention_kwargs,
            )
            
            if output_hidden_states and self.repa and sft:
                for depth_idx, target_depth in enumerate(self.repa_depth):
                    if target_depth > 0 and (i+1) == target_depth:
                        current_hidden = hidden_states.detach() if detach_hidden else hidden_states
                        repa_hiddens.append(current_hidden)
                
        # single blocks
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        for j, block in enumerate(self.single_transformer_blocks):
            hidden_states = block(
                hidden_states=hidden_states, 
                temb=temb, 
                image_rotary_emb=image_rotary_emb,
                # joint_attention_kwargs=joint_attention_kwargs
            )
            
            if output_hidden_states and self.repa and sft:
                for depth_idx, target_depth in enumerate(self.repa_depth):
                    if target_depth < 0 and (j+1) == abs(target_depth):
                        audio_hidden = hidden_states[:, encoder_hidden_states.shape[1]:]
                        current_hidden = audio_hidden.detach() if detach_hidden else audio_hidden
                        repa_hiddens.append(current_hidden)
                        
        hidden_states = hidden_states[:, encoder_hidden_states.shape[1]:]
    
        hidden_states = self.norm_out(hidden_states, temb)
        sample = self.proj_out(hidden_states)

        repa_projected = None
        if output_hidden_states and repa_hiddens and sft:
            repa_projected = []
            for hidden, projector in zip(repa_hiddens, self.projectors):
                repa_projected.append(projector(hidden))
            repa_projected = tuple(repa_projected)

        if not return_dict:
            return sample, repa_hiddens, repa_projected
        return Transformer2DModelOutputWithREPA(
            sample=sample, 
            hidden_states=repa_hiddens, 
            repa_projected=repa_projected
        )
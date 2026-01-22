import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


class ConvBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, dropout, activation=nn.ReLU()
    ):
        super(ConvBlock, self).__init__()
        
        self.conv_layer = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=int((kernel_size - 1) / 2),
                dilation=1,
            ),
            nn.BatchNorm1d(out_channels),
            activation,
        )
        self.dropout = dropout
        self.layer_norm = nn.LayerNorm(out_channels)
        
    def forward(self, x, mask=None):
        x = x.contiguous().transpose(1, 2)
        x = F.dropout(self.conv_layer(x), self.dropout, self.training)
        x = self.layer_norm(x.contiguous().transpose(1, 2))
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0)
            
        return x
    
    
class SwishBlock(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels):
        super(SwishBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_channels, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_channels, bias=True),
        )
        
    def forward(self, S, E, V):
        out = torch.cat(
            [
                S.unsqueeze(-1),
                E.unsqueeze(-1),
                V.unsqueeze(1).expand(-1, E.size(1), -1, -1),
            ],
            dim=-1,
        )
        out = self.layer(out)
        
        return out
    
class LatentMapper(nn.Module):
    def __init__(self, in_features=64, out_channels=8, kernel_size=3, hidden_channels=16):
        super(LatentMapper, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, 
            out_channels=hidden_channels, 
            kernel_size=kernel_size, 
            stride=2, 
            padding=kernel_size // 2
        )
        self.conv2 = nn.Conv2d(
            in_channels=hidden_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=2, 
            padding=kernel_size // 2
        )

    def forward(self, x):
        batch, h, w = x.shape
        target_h = math.floor(h / 4)
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        if x.size(2) != target_h:
            x = x[:, :, :target_h, :]
            
        return x
    
    
class LatentMapper2(nn.Module):
    def __init__(self, d_model: int = 64, use_bias: bool = False):
        super().__init__()
        self.post = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=1, bias=use_bias),
            nn.GELU()
        )
        self.ln = nn.LayerNorm(d_model)

    @torch.no_grad()
    def _pool_one(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        return F.adaptive_avg_pool1d(x.T.unsqueeze(0), int(target_len.item())).squeeze(0).T

    def forward(
        self,
        mu_mel: torch.Tensor,    
        mel_lengths: torch.Tensor, 
        target_lens: torch.Tensor
    ) -> torch.Tensor:
        B, _, D = mu_mel.shape
        pooled = [
            self._pool_one(mu_mel[i, :mel_lengths[i]], target_lens[i])
            for i in range(B)
        ]
        mapped = pad_sequence(pooled, batch_first=True)
        mapped = self.post(mapped.transpose(1, 2))
        mapped = self.ln(mapped.transpose(1, 2))
        return mapped
""" from https://github.com/jaywalnut310/glow-tts """

import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
import pyroomacoustics as pra
import random

def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(int(max_length), dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def fix_len_compatibility(length, num_downsamplings_in_unet=2):
    while True:
        if length % (2**num_downsamplings_in_unet) == 0:
            return length
        length += 1


def convert_pad_shape(pad_shape):
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape


def generate_path(duration, mask):
    device = duration.device

    b, t_x, t_y = mask.shape
    cum_duration = torch.cumsum(duration, 1)
    path = torch.zeros(b, t_x, t_y, dtype=mask.dtype).to(device=device)

    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
    path = path.view(b, t_x, t_y)
    path = (
        path
        - torch.nn.functional.pad(path, convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[
            :, :-1
        ]
    )
    path = path * mask
    return path


def duration_loss(logw, logw_, lengths):
    loss = torch.sum((logw - logw_) ** 2) / torch.sum(lengths)
    return loss


def collate_1d_or_2d(values, pad_idx=0, left_pad=False, shift_right=False, max_len=None, shift_id=1):
    if len(values[0].shape) == 1:
        return collate_1d(values, pad_idx, left_pad, shift_right, max_len, shift_id)
    else:
        return collate_2d(values, pad_idx, left_pad, shift_right, max_len)


def collate_1d(values, pad_idx=0, left_pad=False, shift_right=False, max_len=None, shift_id=1):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values) if max_len is None else max_len
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if shift_right:
            dst[1:] = src[:-1]
            dst[0] = shift_id
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
    return res


def collate_2d(values, pad_idx=0, left_pad=False, shift_right=False, max_len=None):
    """Convert a list of 2d tensors into a padded 3d tensor."""
    size = max(v.size(0) for v in values) if max_len is None else max_len
    res = values[0].new(len(values), size, values[0].shape[1]).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if shift_right:
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
    return res

def intersperse(lst, item):
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def generate_rir(room_dim, source_position, mic_position, fs=16000, rt60=0.5):
    e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)
    room = pra.ShoeBox(room_dim, fs=fs, materials=pra.Material(e_absorption), max_order=max_order)
    room.add_source(source_position)
    room.add_microphone_array(np.array([mic_position]).T)
    room.compute_rir()
    rir = room.rir[0][0]
    return rir

def apply_rir_filter(waveform, rir):
    rir_tensor = torch.tensor(rir, dtype=waveform.dtype, device=waveform.device)
    waveform = waveform.unsqueeze(0)
    rir_tensor = rir_tensor.view(1, 1, -1)
    padding = rir_tensor.shape[-1] // 2
    filtered = F.conv1d(waveform, rir_tensor, padding=padding)
    return filtered.squeeze(0)

def match_noise_length(aug_waveform: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    L_speech = aug_waveform.shape[1]
    L_noise = noise.shape[1]

    if L_noise < L_speech:
        repeat_factor = (L_speech // L_noise) + 1  
        noise_tiled = noise.repeat(1, repeat_factor)  
        noise_matched = noise_tiled[:, :L_speech]
    else:
        start_idx = random.randint(0, L_noise - L_speech)
        noise_matched = noise[:, start_idx : start_idx + L_speech]

    return noise_matched



def position_based_augmentation(sr, speech, noise, position, target_dur, speech_dur):
    
    target_len = int(target_dur * sr)
    speech_len = speech.shape[1]
    remaining_len = target_len - speech_len
    
    if noise.shape[1] < target_len:
        repeat_factor = (target_len // noise.shape[1]) + 1
        noise = noise.repeat(1, repeat_factor)
    
    if noise.shape[1] > target_len:
        start_idx = random.randint(0, noise.shape[1] - target_len)
        noise = noise[:, start_idx:start_idx + target_len]
        
    if torch.linalg.vector_norm(noise, ord=2, dim=-1).item() == 0.0:        
        return False, speech, 0, 0

    augmented = torch.zeros((speech.shape[0], target_len), device=speech.device)
    
    if position == 'front': 
        front_len = 0
    elif position == 'near front': 
        front_len = int(remaining_len * 0.25)
    elif position == 'middle':
        front_len = remaining_len // 2
    elif position == 'near back': 
        front_len = int(remaining_len * 0.75)
    elif position == 'back': 
        front_len = remaining_len
    back_len = remaining_len - front_len

    augmented[:, front_len:front_len+speech_len] = speech
    snr = torch.Tensor(1).uniform_(2, 10)
    augmented_with_noise = torchaudio.functional.add_noise(augmented, noise, snr)

    return True, augmented_with_noise, front_len, back_len

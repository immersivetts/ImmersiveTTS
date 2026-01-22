<h2 align="center"> ImmersiveTTS: Environment-Aware Text-to-Speech with Multimodal Diffusion Transformer and Domain-Specific Representation Alignment </h2>

<h3 align="center"> Anonymous authors </h3>

<img src="figure/overview.jpg" align="center" width="1000" height="450">
Overview of ImmersiveTTS. A dual-stream MM-DiT backbone conditions the speech stream on content prompt-aligned linguistic features. At the same time, Flan-T5 token embeddings drive the environmental context stream, and CLAP embeddings modulate AdaLN for global conditioning. The model is trained with flow matching and domain-specific REPA objectives.

### Clone our repository
```
git clone https://github.com/immersivetts/ImmersiveTTS.git
cd ImmersiveTTS
```

### Install the requirements
```
pip install -r requirements.txt
```

### Data download 
* Download the speech ([LibriTTS](https://www.openslr.org/60/)) and background audio ([WavCaps](https://huggingface.co/datasets/cvssp/WavCaps)) datasets.

### Training

We train ImmersiveTTS with **Hugging Face Accelerate** for multi-GPU support. Training is configured via:
- `configs/model.yaml`: experiment/output settings, training schedule, optimizer, data manifest paths, and model hyperparameters.
- `configs/accelerator.yaml`: distributed setup for Accelerate (e.g., number of processes/GPUs, port, mixed precision).

#### 1) Set dataset paths
Update the dataset manifest paths in `configs/model.yaml` to point to your local files:
- `libri_tts_path`, `test_libri_tts_path`
- `as_nonspeech_path`, `bbc_sound_effects_path`, `sound_bible_path`, `free_sound_path`

(Each `*_path` expects a JSON manifest. See the data preparation scripts/docs in this repository for the expected format.)

#### 2) Configure distributed training (optional)
For multi-GPU training, adjust `configs/accelerator.yaml` (e.g., `gpu_ids`, `num_processes`, `main_process_port`).  
If you prefer bf16 training and your hardware supports it, enable bf16 in the Accelerate config.

#### 3) Run training
We provide a reference launcher script:

```bash
export TORCH_DISTRIBUTED_DEBUG=DETAIL
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
  --config_file configs/accelerator.yaml train.py \
  --config configs/model.yaml

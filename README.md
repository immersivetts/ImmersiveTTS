<h2 align="center"> ImmersiveTTS: Environment-Aware Text-to-Speech with Multimodal Diffusion Transformer and Domain-Specific Representation Alignment </h2>

<h3 align="center"> Anonymous authors </h3>

<img src="figure/overview.jpg" align="center" width="1000" height="450">
Recent advancements in text-guided audio generation have yielded promising results in diverse domains, including sound effects, environmental audio, speech, and music. However, jointly generating speech with environmental audio remains challenging due to the inherent disparities in their acoustic patterns and temporal dynamics. We propose ImmersiveTTS, an Environment-Aware text-to-speech (TTS) model that generates natural speech seamlessly integrated within environmental contexts by explicitly modeling cross-modal interactions. Our model builds on a multimodal diffusion transformer and fuses transcript-aligned speech latent with text-conditioned environmental context via joint attention. To enhance semantic consistency, we introduce a domain-specific representation alignment objective tailored to Environment-Aware TTS, leveraging complementary self-supervised representations from speech and audio encoders. Experimental results show that ImmersiveTTS achieves higher naturalness, intelligibility, and audio fidelity than existing approaches across objective metrics and human listening tests.

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

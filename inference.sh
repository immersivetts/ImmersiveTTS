export TORCH_DISTRIBUTED_DEBUG=DETAIL
CUDA_VISIBLE_DEVICES=0 python inference.py \
            --config configs/inference_repa.yaml \
            --output_dir /your/path/to/output_dir \
            --ckpt_path /your/path/to/checkpoint.pt \
            --env_prompt "your environment prompt text" \
            --cont_prompt "your content prompt text" \
            --spk_prompt_path /path/to/speaker_audio.wav \
            --num_inference_steps 25 \
            --cont_guidance_scale 3 \
            --env_guidance_scale 3 \

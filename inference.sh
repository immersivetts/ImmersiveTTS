export TORCH_DISTRIBUTED_DEBUG=DETAIL
CUDA_VISIBLE_DEVICES=3 python inference_tts.py --config configs/inference_repa.yaml \
            --ckpt_path /your/path/to/checkpoint.pt \
            --output_dir /your/path/to/output_dir \
            --num_inference_steps 25 \
            --cont_guidance_scale 3 \
            --env_guidance_scale 3 \
            --spk_embedding_prepared False

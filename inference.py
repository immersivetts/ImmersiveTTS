import argparse
import yaml
import torch
import os
import torchaudio

from immtts.pipeline import ImmersivettsPipeline

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", default=None, type=str, help="Path to configuration file")
    parser.add_argument("--output_dir", default="outputs", help="Directory name to save output files")
    parser.add_argument("--ckpt_path", default=None, help="Path to load checkpoint from")
    parser.add_argument("--env_prompt", type=str, required=True, help="Environment prompt text")
    parser.add_argument("--cont_prompt", type=str, required=True, help="Content prompt text")
    parser.add_argument("--spk_prompt_path", type=str, default=None, help="Path to prompt speech audio file")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--cont_guidance_scale", type=float, default=1.0, help="Content guidance scale for dual CFG")
    parser.add_argument("--env_guidance_scale", type=float, default=1.0, help="Environment guidance scale for dual CFG")
    args = parser.parse_args()

    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            for key, value in config.items():
                setattr(args, key, value)
    return args

    
def main():
    args = parse_args()

    if not args.ckpt_path:
        raise ValueError("--ckpt_path is required")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    pipeline = ImmersivettsPipeline(
        model_config=args.model, 
        ckpt_path=args.ckpt_path, 
        device=device,
        spk_embedding_prepared=False
    )

    if args.ckpt_path:
        ckpt_dir_name = os.path.basename(os.path.dirname(args.ckpt_path))
        output_dir = os.path.join(args.output_dir, ckpt_dir_name, 'immersive_tts')
    else:
        output_dir = os.path.join(args.output_dir, 'immersive_tts')
    os.makedirs(output_dir, exist_ok=True)

    try:
        speaker_speech, sr = torchaudio.load(args.spk_prompt_path)
        print(f"Speaker speech loaded: shape={speaker_speech.shape}, sample_rate={sr}")
    except Exception as e:
        raise RuntimeError(f"Speaker speech load error: {e}")

    output_path = os.path.join(output_dir, f"sample.wav")
    
    print(f"Environment prompt: {args.env_prompt}")
    print(f"Content prompt: {args.cont_prompt}")
    print(f"Speech prompt: {args.spk_prompt_path}")
    print(f"Number of inference steps: {args.num_inference_steps}")
    print(f"Content guidance scale: {args.cont_guidance_scale}")
    print(f"Environment guidance scale: {args.env_guidance_scale}")
    
    try:
        print("Generating audio...")
        audio = pipeline.generate(
            cont_prompt=args.cont_prompt, 
            speaker_prompt=speaker_speech, 
            env_prompt=args.env_prompt,
            num_samples_per_prompt=1,
            num_inference_steps=args.num_inference_steps, 
            duration=10,
            cont_guidance_scale=args.cont_guidance_scale,
            env_guidance_scale=args.env_guidance_scale,
            save_dir=output_dir,
        )  
        
        torchaudio.save(output_path, audio, sample_rate=16000)
    except Exception as e:
        print(f"Audio generation error: {e}")
    
if __name__ == "__main__":
    main() 
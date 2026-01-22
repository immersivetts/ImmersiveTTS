export TORCH_DISTRIBUTED_DEBUG=DETAIL
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
        --config_file='configs/accelerator.yaml' train.py \
        --config='configs/model.yaml' 
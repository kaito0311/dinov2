CUDA_VISIBLE_DEVICES=1 python train.py \
    --nodes 1 \
    --config-file dinov2/configs/train/vitl14.yaml \
    --output-dir ./output_dinov2 \
    train.dataset_path=data/root/test
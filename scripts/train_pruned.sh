CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 4885 \
    train_prune.py \
    --data_path /home/data/imagenet/train \
    --val_data_path /home/data/oxford102_flowers/ \
    --val_hint_dir ./data/oxford102_flowers/1234 \
    --output_dir ./saved_models/oxford_layer_real \
    --log_dir ./saved_models/oxford_layer_real \
    --exp_name exp \
    --save_args_txt \
    --epochs 25 \
    --batch_size 64 \
    --model block_icolorit_tiny_4ch_patch16_224 \
    --model_path ./pretrained/icolorit_tiny_4ch_patch16_224.pth \
    --flops 0.5 \
    --no_use_rpb \
    $opt

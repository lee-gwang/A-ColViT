# for latency
# python infer_pruned.py \
#     --model_path ./pretrained/pruning/icolorit_tiny_4ch_patch16_224-0.3.pth \
#     --model block_icolorit_tiny_4ch_patch16_224 \
#     --batch_size 1 \
#     --val_data_path  /home/data/imagenet/ctest10k \
#     --val_hint_dir  ./data/ctest10k/1234 \
#     --no_use_rpb \
#     --device cpu \
#     $opt


# python infer_pruned.py \
#     --model_path ./pretrained/pruning/icolorit_tiny_4ch_patch16_224-0.5-2.pth \
#     --model block_icolorit_tiny_4ch_patch16_224 \
#     --batch_size 1 \
#     --val_data_path  /home/data/imagenet/ctest10k \
#     --val_hint_dir  ./data/ctest10k/1234 \
#     --no_use_rpb \
#     --device cpu \
#     $opt


# python infer_pruned.py \
#     --model_path ./pretrained/pruning/icolorit_tiny_4ch_patch16_224-0.7.pth \
#     --model block_icolorit_tiny_4ch_patch16_224 \
#     --batch_size 1 \
#     --val_data_path  /home/data/imagenet/ctest10k \
#     --val_hint_dir  ./data/ctest10k/1234 \
#     --no_use_rpb \
#     --device cpu \
#     $opt

###########
# # imagenet
# 0.3
# python infer_pruned.py \
#     --model_path ./pretrained/pruning/icolorit_tiny_4ch_patch16_224-0.3.pth \
#     --model block_icolorit_tiny_4ch_patch16_224 \
#     --batch_size 32 \
#     --val_data_path  /home/data/imagenet/ctest10k \
#     --val_hint_dir  ./data/ctest10k/1234 \
#     --no_use_rpb \
#     $opt
# # 0.5
# python infer_pruned.py \
#     --model_path ./pretrained/pruning/icolorit_tiny_4ch_patch16_224-0.5-2.pth \
#     --model block_icolorit_tiny_4ch_patch16_224 \
#     --batch_size 32 \
#     --val_data_path  /home/data/imagenet/ctest10k \
#     --val_hint_dir  ./data/ctest10k/1234 \
#     --no_use_rpb \
#     $opt

# # 0.7
# python infer_pruned.py \
#     --model_path ./pretrained/pruning/icolorit_tiny_4ch_patch16_224-0.7.pth \
#     --model block_icolorit_tiny_4ch_patch16_224 \
#     --batch_size 32 \
#     --val_data_path  /home/data/imagenet/ctest10k \
#     --val_hint_dir  ./data/ctest10k/1234 \
#     --no_use_rpb \
#     $opt

# # 0.9
# python infer_pruned.py \
#     --model_path ./pretrained/pruning/icolorit_tiny_4ch_patch16_224-0.9.pth \
#     --model block_icolorit_tiny_4ch_patch16_224 \
#     --batch_size 32 \
#     --val_data_path  /home/data/imagenet/ctest10k \
#     --val_hint_dir  ./data/ctest10k/1234 \
#     --no_use_rpb \
#     $opt

#######################
# flower
#######################

# # 0.3
# python infer_pruned.py \
#     --model_path ./pretrained/pruning/icolorit_tiny_4ch_patch16_224-0.3.pth \
#     --model block_icolorit_tiny_4ch_patch16_224 \
#     --batch_size 32 \
#     --val_data_path  /home/data/oxford102_flowers \
#     --val_hint_dir  ./data/oxford102_flowers/1234 \
#     --no_use_rpb \
#     $opt

# # 0.5
# python infer_pruned.py \
#     --model_path ./pretrained/pruning/icolorit_tiny_4ch_patch16_224-0.5-2.pth \
#     --model block_icolorit_tiny_4ch_patch16_224 \
#     --batch_size 32 \
#     --val_data_path  /home/data/oxford102_flowers \
#     --val_hint_dir  ./data/oxford102_flowers/1234 \
#     --no_use_rpb \
#     $opt

# 0.7
# python infer_pruned.py \
#     --model_path ./pretrained/pruning/icolorit_tiny_4ch_patch16_224-0.7.pth \
#     --model block_icolorit_tiny_4ch_patch16_224 \
#     --batch_size 32 \
#     --val_data_path  /home/data/oxford102_flowers \
#     --val_hint_dir  ./data/oxford102_flowers/1234 \
#     --no_use_rpb \
#     $opt


# # 0.9
# python infer_pruned.py \
#     --model_path ./pretrained/pruning/icolorit_tiny_4ch_patch16_224-0.9.pth \
#     --model block_icolorit_tiny_4ch_patch16_224 \
#     --batch_size 32 \
#     --val_data_path  /home/data/oxford102_flowers \
#     --val_hint_dir  ./data/oxford102_flowers/1234 \
#     --no_use_rpb \
#     $opt

#######################
# cub
# #######################
# # 0.3
# python infer_pruned.py \
#     --model_path ./pretrained/pruning/icolorit_tiny_4ch_patch16_224-0.3.pth \
#     --model block_icolorit_tiny_4ch_patch16_224 \
#     --batch_size 32 \
#     --val_data_path  /home/data/cub_200 \
#     --val_hint_dir  ./data/cub_200/1234 \
#     --no_use_rpb \
#     $opt

# # 0.5
# python infer_pruned.py \
#     --model_path ./pretrained/pruning/icolorit_tiny_4ch_patch16_224-0.5-2.pth \
#     --model block_icolorit_tiny_4ch_patch16_224 \
#     --batch_size 32 \
#     --val_data_path  /home/data/cub_200 \
#     --val_hint_dir  ./data/cub_200/1234 \
#     --no_use_rpb \
#     $opt



#0.7
# python infer_pruned.py \
#     --model_path ./pretrained/pruning/icolorit_tiny_4ch_patch16_224-0.7.pth \
#     --model block_icolorit_tiny_4ch_patch16_224 \
#     --batch_size 32 \
#     --val_data_path  /home/data/cub_200 \
#     --val_hint_dir  ./data/cub_200/1234 \
#     --no_use_rpb \
#     $opt



#0.9
# python infer_pruned.py \
#     --model_path ./pretrained/pruning/icolorit_tiny_4ch_patch16_224-0.9.pth \
#     --model block_icolorit_tiny_4ch_patch16_224 \
#     --batch_size 32 \
#     --val_data_path  /home/data/cub_200 \
#     --val_hint_dir  ./data/cub_200/1234 \
#     --no_use_rpb \
#     $opt
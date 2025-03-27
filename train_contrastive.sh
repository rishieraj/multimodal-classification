#!/bin/bash


EXP_NAME="contrastive_exp"

DATA_ROOT="dataset/"
CONFIG_PATH="configs/config.yaml"
SPLIT_PATH="contrastive_split.json"

# python train.py \
#     --task contrastive \
#     --modality_list vision touch audio \
#     --batch_size 32 \
#     --lr 1e-2 \
#     --weight_decay 1e-4 \
#     --epochs 50 \
#     --temperature 0.07 \
#     --data_location ${DATA_ROOT} \
#     --config_location ${CONFIG_PATH} \
#     --split_location ${SPLIT_PATH} \
#     --exp ${EXP_NAME} \
#     --num_workers 4 \
#     --seed 42

# Define grid search ranges
# lr_values=(1e-4 3e-4 5e-4)
# wd_values=(1e-2 3e-2 5e-2)
batch_size_values=(32 64 128)

for bz in "${batch_size_values[@]}"; do
    EXP_NAME="contrastive_exp_bz_${bz}"
    echo "Running experiment: ${EXP_NAME}"
    python train.py \
        --task contrastive \
        --modality_list vision touch audio \
        --batch_size ${bz} \
        --lr 1e-2 \
        --weight_decay 0.0 \
        --epochs 20 \
        --temperature 0.07 \
        --data_location ${DATA_ROOT} \
        --config_location ${CONFIG_PATH} \
        --split_location ${SPLIT_PATH} \
        --exp ${EXP_NAME} \
        --num_workers 4 \
        --seed 42
done
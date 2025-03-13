#!/bin/bash

DATA_ROOT="dataset/"
CONFIG_PATH="configs/config.yaml"
SPLIT_PATH="split_0.json"

# Vision
python train.py \
    --finetune \
    --task unimodal \
    --modality vision \
    --batch_size 32 \
    --lr 5e-4 \
    --weight_decay 1e-2 \
    --epochs 100 \
    --data_location ${DATA_ROOT} \
    --config_location ${CONFIG_PATH} \
    --split_location ${SPLIT_PATH} \
    --backbone fenet \
    --exp visual_exp \
    --num_workers 2 \
    --seed 42

# Touch
python train.py \
    --finetune \
    --task unimodal \
    --modality touch \
    --batch_size 32 \
    --lr 5e-4 \
    --weight_decay 1e-2 \
    --epochs 100 \
    --data_location ${DATA_ROOT} \
    --config_location ${CONFIG_PATH} \
    --split_location ${SPLIT_PATH} \
    --backbone fenet \
    --exp touch_exp \
    --num_workers 2 \
    --seed 42

# Audio
python train.py \
    --finetune \
    --task unimodal \
    --modality audio \
    --batch_size 32 \
    --lr 3e-4 \
    --weight_decay 5e-2 \
    --epochs 100 \
    --data_location ${DATA_ROOT} \
    --config_location ${CONFIG_PATH} \
    --split_location ${SPLIT_PATH} \
    --backbone fenet \
    --exp audio_exp \
    --num_workers 2 \
    --seed 42

# # Define grid search ranges
# lr_values=(1e-4 3e-4 5e-4)
# wd_values=(1e-2 3e-2 5e-2)

# for lr in "${lr_values[@]}"; do
#     for wd in "${wd_values[@]}"; do
#         EXP_NAME="audio_exp_lr_${lr}_wd_${wd}"
#         echo "Running experiment: ${EXP_NAME}"
#         python train.py \
#             --finetune \
#             --task unimodal \
#             --modality audio \
#             --batch_size 32 \
#             --lr ${lr} \
#             --weight_decay ${wd} \
#             --epochs 100 \
#             --data_location ${DATA_ROOT} \
#             --config_location ${CONFIG_PATH} \
#             --split_location ${SPLIT_PATH} \
#             --backbone fenet \
#             --exp ${EXP_NAME} \
#             --num_workers 2 \
#             --seed 42
#     done
# done
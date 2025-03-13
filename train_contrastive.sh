#!/bin/bash


EXP_NAME="contrastive_exp"

DATA_ROOT="dataset/"
CONFIG_PATH="configs/config.yaml"
SPLIT_PATH="contrastive_split.json"

python train.py \
    --task contrastive \
    --modality_list vision touch audio \
    --batch_size 32 \
    --lr 1e-3 \
    --weight_decay 0.0 \
    --epochs 20 \
    --temperature 0.07 \
    --data_location ${DATA_ROOT} \
    --config_location ${CONFIG_PATH} \
    --split_location ${SPLIT_PATH} \
    --exp ${EXP_NAME} \
    --num_workers 4 \
    --seed 42

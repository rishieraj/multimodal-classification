#!/bin/bash


EXP_NAME="contrastive_exp"

DATA_ROOT="Your data root"
CONFIG_PATH="Your config path"
SPLIT_PATH="The path to contrastive_split.json"

python train.py \
    --task contrastive \
    --modality_list vision touch audio \
    --batch_size 32 \
    --lr 1e-4 \
    --weight_decay 1e-3 \
    --epochs 100 \
    --temperature 0.07 \
    --data_location ${DATA_ROOT} \
    --config_location ${CONFIG_PATH} \
    --split_location ${SPLIT_PATH} \
    --exp ${EXP_NAME} \
    --num_workers 4 \
    --seed 42

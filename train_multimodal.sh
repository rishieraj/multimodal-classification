#!/bin/bash

DATA_ROOT="dataset/"
CONFIG_PATH="configs/config.yaml"
SPLIT_PATH="split_0.json"

# Late Fusion
python train.py \
    --task multimodal \
    --fusion_type late \
    --modality_list vision touch audio \
    --batch_size 32 \
    --lr 1e-3 \
    --weight_decay 1e-2 \
    --epochs 50 \
    --data_location ${DATA_ROOT} \
    --config_location ${CONFIG_PATH} \
    --split_location ${SPLIT_PATH} \
    --exp late_fusion_exp \
    --num_workers 2 \
    --seed 42

# Attention Fusion
python train.py \
    --task multimodal \
    --fusion_type attention \
    --modality_list vision touch audio \
    --batch_size 32 \
    --lr 1e-3 \
    --weight_decay 1e-2 \
    --epochs 50 \
    --data_location ${DATA_ROOT} \
    --config_location ${CONFIG_PATH} \
    --split_location ${SPLIT_PATH} \
    --exp attention_fusion_exp \
    --num_workers 2 \
    --seed 42
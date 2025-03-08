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
    --epochs 50 \
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
    --epochs 50 \
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
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --epochs 100 \
    --data_location ${DATA_ROOT} \
    --config_location ${CONFIG_PATH} \
    --split_location ${SPLIT_PATH} \
    --backbone fenet \
    --exp audio_exp \
    --num_workers 2 \
    --seed 42
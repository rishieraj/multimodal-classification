#!/bin/bash
export PYTHONPATH="$HOME/Documents/Semester-4/CMSC848M/Week-5/Coding Assignment/cmsc848M:${PYTHONPATH}"

DATA_ROOT="dataset/"
SPLIT_PATH="contrastive_split.json"
PRETRAINED_PATH="experiments/contrastive/contrastive_exp/best_model.pth"

# vision -> touch
python models/retrieval.py \
    --pretrained_path ${PRETRAINED_PATH} \
    --data_root ${DATA_ROOT} \
    --split_location ${SPLIT_PATH} \
    --query_modality vision \
    --target_modality touch \
    --batch_size 32 \
    --num_workers 4

# vision -> audio
python models/retrieval.py \
    --pretrained_path ${PRETRAINED_PATH} \
    --data_root ${DATA_ROOT} \
    --split_location ${SPLIT_PATH} \
    --query_modality vision \
    --target_modality audio \
    --batch_size 32 \
    --num_workers 4

# touch -> audio
python models/retrieval.py \
    --pretrained_path ${PRETRAINED_PATH} \
    --data_root ${DATA_ROOT} \
    --split_location ${SPLIT_PATH} \
    --query_modality touch \
    --target_modality audio \
    --batch_size 32 \
    --num_workers 4
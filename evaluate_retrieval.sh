#!/bin/bash
export PYTHONPATH="your project root path:${PYTHONPATH}"

DATA_ROOT="your data root"
SPLIT_PATH="The path to contrastive_split.json"
PRETRAINED_PATH="your pretrained model path"

# vision -> touch
python models/retrieval_answer.py \
    --pretrained_path ${PRETRAINED_PATH} \
    --data_root ${DATA_ROOT} \
    --split_location ${SPLIT_PATH} \
    --query_modality vision \
    --target_modality touch \
    --batch_size 32 \
    --num_workers 4

# vision -> audio
python models/retrieval_answer.py \
    --pretrained_path ${PRETRAINED_PATH} \
    --data_root ${DATA_ROOT} \
    --split_location ${SPLIT_PATH} \
    --query_modality vision \
    --target_modality audio \

# touch -> audio
python models/retrieval_answer.py \
    --pretrained_path ${PRETRAINED_PATH} \
    --data_root ${DATA_ROOT} \
    --split_location ${SPLIT_PATH} \
    --query_modality touch \
    --target_modality audio \
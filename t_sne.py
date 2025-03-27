import os
import os.path as osp
import argparse
import yaml
import json
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE
from models.unimodal import VisionClassifier, TouchClassifier, AudioClassifier
from models.contrastive import ModalityEncoder
from torch.utils.data import Dataset, DataLoader
from models.contrastive import ContrastiveLearning
from data.dataset import MaterialDataset, ContrastiveDataset


def parse_args():
    parser = argparse.ArgumentParser()
    # Task
    parser.add_argument('--task', type=str, required=True,
                      choices=['unimodal', 'multimodal', 'contrastive'])
    parser.add_argument('--modality', type=str, default='vision',
                      choices=['vision', 'touch', 'audio'])
    parser.add_argument('--modality_list', nargs='+', default=['vision', 'touch', 'audio'],
                      help='List of modalities to use')
    parser.add_argument('--fusion_type', type=str, default='late',
                      choices=['early', 'late', 'attention'])
    
    # Model
    parser.add_argument('--backbone', type=str, default='resnet',
                      choices=['resnet', 'fenet'])
    parser.add_argument('--pretrained_path', type=str, default=None)
    
    # Training
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Paths
    parser.add_argument('--data_location', type=str, required=True)
    parser.add_argument('--config_location', type=str, required=True)
    parser.add_argument('--split_location', type=str, required=True)
    parser.add_argument('--exp', type=str, default='default')
    
    # Mode
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--finetune', action='store_true')
    
    # Contrastive learning parameters
    parser.add_argument('--temperature', type=float, default=0.07,
                      help='Temperature parameter for contrastive loss')
    
    return parser.parse_args()

def main():
    # Parse arguments and load config
    args = parse_args()

    dataset = ContrastiveDataset(args, 'train')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=dataset.collate_fn)

    # Load model
    model = ContrastiveLearning(modalities=args.modality_list, feature_dim=128, num_classes=98)
    model.to("cuda")  # <--- Move model and submodules to GPU
    model.eval()

    embeddings_list = []
    labels_list = []

    with torch.no_grad():
        for batch in dataloader:
            for modality in args.modality_list:
                batch[f'{modality}_1'] = batch[f'{modality}_1'].to('cuda')
                batch[f'{modality}_2'] = batch[f'{modality}_2'].to('cuda')
            batch['label'] = batch['label'].to('cuda')

            outputs = model(batch)
            proj_dict = outputs['projections']

            for mod_key, proj_tensor in proj_dict.items():
                embeddings_list.append(proj_tensor.cpu().numpy())
                labels_list.append(batch["label"].cpu().numpy())

    embeddings = np.concatenate(embeddings_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)  

    print("embeddings shape:", embeddings.shape)

    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000, verbose=1)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(8,8))
    scatter = plt.scatter(
        embeddings_2d[:,0], 
        embeddings_2d[:,1], 
        c=labels,        # color by labels
        cmap="tab10",    # or another colormap
        alpha=0.7
    )
    plt.colorbar(scatter)
    plt.title("t-SNE of Contrastive Embeddings")
    plt.xlabel("TSNE-1")
    plt.ylabel("TSNE-2")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
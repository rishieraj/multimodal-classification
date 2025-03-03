# task4_retrieval.py

import os.path as osp
import json
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import argparse
import torch.nn as nn
import torch.nn.functional as F

class RetrievalDataset:
    """Dataset class for cross-modal retrieval evaluation.
    
    This class handles loading and preprocessing data from different modalities
    (vision, touch, audio) for cross-modal retrieval tasks. It creates pairs
    of query and target samples from specified modalities.
    """
    
    def __init__(self, data_root, split_path, modalities, query_modality, target_modality):
        """Initialize the retrieval dataset.
        
        Args:
            data_root (str): Root directory containing all modality data
            split_path (str): Path to the dataset split file
            modalities (list): List of modalities to use
            query_modality (str): Modality to use as query (e.g., 'vision', 'touch', 'audio')
            target_modality (str): Modality to retrieve (e.g., 'vision', 'touch', 'audio')
        """
        self.paths = {
            'vision': osp.join(data_root, 'vision'),
            'touch': osp.join(data_root, 'touch'),
            'audio': osp.join(data_root, 'audio_examples')
        }
        
        # Load dataset splits and labels
        with open(osp.join(data_root, 'label.json')) as f:
            self.label_dict = json.load(f)
        with open(split_path) as f:
            splits = json.load(f)
            self.query_samples = []
            self.target_samples = []
            
            # Build query-target pairs
            for obj in splits['test']:  # Using test set for evaluation
                instances = splits['test'][obj]
                for instance in instances:
                    self.query_samples.append((obj, instance))
                    self.target_samples.append((obj, instance))
                    
        self.query_modality = query_modality
        self.target_modality = target_modality
        
        print(f"\nDataset Statistics:")
        print(f"Number of samples: {len(self.query_samples)}")
        print(f"Query modality: {query_modality}")
        print(f"Target modality: {target_modality}")

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.query_samples)

    def __getitem__(self, idx):
        """Get a query-target pair.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            dict: Contains:
                - query: Query sample tensor
                - target: Target sample tensor
                - query_meta: Tuple of (object, instance) for query
                - target_meta: Tuple of (object, instance) for target
        """
        query_obj, query_instance = self.query_samples[idx]
        target_obj, target_instance = self.target_samples[idx]
        
        query_data = self._load_data(query_obj, query_instance, self.query_modality)
        target_data = self._load_data(target_obj, target_instance, self.target_modality)
        
        return {
            'query': query_data,
            'target': target_data,
            'query_meta': (query_obj, query_instance),
            'target_meta': (target_obj, target_instance)
        }

    def _load_data(self, obj, instance, modality):
        """Load data for a specific modality.
        
        Args:
            obj (str): Object identifier
            instance (str): Instance identifier
            modality (str): Modality to load ('vision', 'touch', or 'audio')
            
        Returns:
            torch.Tensor: Loaded and preprocessed data
        """
        if modality in ['vision', 'touch']:
            img_path = osp.join(self.paths[modality], obj, f'{instance}.png')
            img = Image.open(img_path).convert('RGB')
            return torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        else:  # audio
            spec_path = osp.join(self.paths['audio'], obj, f'{instance}.npy')
            return torch.tensor(np.load(spec_path)).float()

    def collate_fn(self, batch):
        """Custom collate function for batching samples.
        
        Args:
            batch (list): List of samples to collate
            
        Returns:
            dict: Batched samples
        """
        return {
            'query': torch.stack([item['query'] for item in batch]),
            'target': torch.stack([item['target'] for item in batch]),
            'query_meta': [item['query_meta'] for item in batch],
            'target_meta': [item['target_meta'] for item in batch]
        }

class ModalityEncoder(nn.Module):
    """Encoder network for different modalities.
    
    This class should implement:
    1. A backbone network appropriate for the modality
    2. A projection head for mapping features to a shared space
    3. Feature extraction and normalization
    
    Args:
        modality (str): Which modality this encoder handles
        feature_dim (int): Dimension of output features
        num_classes (int): Number of classes for classification
    """
    def __init__(self, modality, feature_dim, num_classes):
        # TODO: Initialize the encoder
        pass

    def _build_backbone(self):
        # TODO: Build an appropriate backbone network
        pass

    def forward(self, x):
        # TODO: Implement the forward pass
        pass

class ModalityRetrieval:
    """Cross-modal retrieval using pretrained encoders and nearest neighbor search.
    
    This class should implement:
    1. Loading pretrained encoders
    2. Building a feature database
    3. Performing nearest neighbor retrieval
    4. Computing retrieval metrics
    
    Args:
        pretrained_model_path (str): Path to pretrained model weights
        modalities (list): List of modalities to use
    """
    
    def __init__(self, pretrained_model_path, modalities):
        # TODO: Initialize the retrieval system
        pass

    def build_database(self, dataloader):
        """Build a database of features for retrieval.
        
        Should:
        1. Extract features from all database samples
        2. Store features and metadata for retrieval
        
        Args:
            dataloader: DataLoader containing database samples
        """
        # TODO: Implement database building
        pass

    def retrieve(self, query_batch):
        """Retrieve nearest neighbors for a batch of queries.
        
        Should:
        1. Extract features from queries
        2. Find nearest neighbors in the database
        3. Return retrieval results
        
        Args:
            query_batch: Batch of query samples
            
        Returns:
            list: Retrieved metadata for each query
        """
        # TODO: Implement retrieval
        pass

    def evaluate(self, query_loader):
        """Evaluate retrieval performance.
        
        Should compute and return:
        - Mean Average Precision (mAP)
        - Recall at K (R@K)
        - Other relevant metrics
        
        Args:
            query_loader: DataLoader containing query samples
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        if not hasattr(self, 'database_features'):
            self.build_database(query_loader)
            
        print("\nEvaluating retrieval...")
        correct_1 = 0
        correct_5 = 0
        total = 0
        aps = []  # check the ap of each query
        
        with torch.no_grad():
            for batch in tqdm(query_loader):
                # Get the query feature
                query = batch['query'].cuda()
                if self.query_modality == 'touch':
                    features = self.model.encoders[self.query_modality].backbone.backbone(query)
                else:  # vision or audio
                    if self.query_modality == 'audio' and query.dim() == 3:
                        query = query.unsqueeze(1)
                    features = self.model.encoders[self.query_modality].backbone.features(query)
            
                if features.dim() > 2:
                    features = features.flatten(1)
                # get the projected feature
                projected = self.model.encoders[self.query_modality].projector(features)
                projected = F.normalize(projected, dim=1)
                
                # get the similarity
                database_features = torch.from_numpy(self.database_features).cuda()
                similarity = torch.mm(projected, database_features.t())
                
                # get the sorted index
                _, indices = similarity.sort(dim=1, descending=True)
                
                # get the accuracy and AP
                for i, query_meta in enumerate(batch['query_meta']):
                    query_obj = query_meta[0]
                    retrieved_objs = [self.database_meta[j][0] for j in indices[i].cpu().numpy()]
                    
                    # get the R@k
                    if query_obj == retrieved_objs[0]:
                        correct_1 += 1
                    if query_obj in retrieved_objs[:5]:
                        correct_5 += 1
                    
                    # get the AP
                    relevant = [obj == query_obj for obj in retrieved_objs]
                    if sum(relevant) > 0:  # if there is a relevant item
                        precisions = []
                        num_relevant = 0
                        for j, is_relevant in enumerate(relevant):
                            if is_relevant:
                                num_relevant += 1
                                precisions.append(num_relevant / (j + 1))
                        ap = sum(precisions) / sum(relevant)
                        aps.append(ap)
                    
                    total += 1
        
        # get the metrics
        r1 = correct_1 / total * 100
        r5 = correct_5 / total * 100
        mAP = np.mean(aps) * 100 if aps else 0.0
        
        print(f"\nRetrieval Results:")
        print(f"mAP: {mAP:.2f}%")
        print(f"R@1: {r1:.2f}%")
        print(f"R@5: {r5:.2f}%")
        
        return {
            'mAP': mAP,
            'R@1': r1,
            'R@5': r5
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_path', type=str, required=True)
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--split_location', type=str, required=True)
    parser.add_argument('--query_modality', type=str, default='vision')
    parser.add_argument('--target_modality', type=str, default='touch')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()
    
    # Evaluate retrieval in both directions
    modalities = [args.query_modality, args.target_modality]
    results = {}
    
    # TODO: Implement the evaluation loop for both directions
    
    # Save results
    with open('retrieval_results.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    main()
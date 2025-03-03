# task3_contrastive.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.unimodal import VisionClassifier, TouchClassifier, AudioClassifier

class ModalityEncoder(nn.Module):
    """Encoder network for each modality
    
    Instructions:
    1. Initialize backbone with pretrained weights
    2. Add projection head for contrastive learning
    3. Freeze backbone parameters
    4. Handle different input modalities
    
    Args:
        modality (str): Modality type ('vision', 'touch', or 'audio')
        output_dim (int): Dimension of output features
        num_classes (int): Number of classes for the classifier
    """
    def __init__(self, modality, output_dim, num_classes):
        super().__init__()
        # TODO: Initialize backbone and projector
        if modality == 'vision':
            self.backbone = VisionClassifier(num_classes, backbone='fenet')
        elif modality == 'touch':
            self.backbone = TouchClassifier(num_classes, backbone='fenet')
        else:
            self.backbone = AudioClassifier(num_classes, backbone='fenet')

        # Load pretrained weights
        self.load_pretrained_weights(modality)

        # Projection head
        self.projector = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        
    def load_pretrained_weights(self, modality):
        """Load pretrained weights for backbone
        
        Instructions:
        1. Load correct checkpoint for each modality
        2. Update model state dict
        3. Handle loading errors
        """
        base_path = 'experiments/unimodal'
        if modality == 'vision':
            ckpt = torch.load(f'{base_path}/visual_exp/best_model.pth', weights_only=False)
        elif modality == 'touch':
            ckpt = torch.load(f'{base_path}/touch_exp/best_model.pth', weights_only=False)
        else:
            ckpt = torch.load(f'{base_path}/audio_exp/best_model.pth', weights_only=False)
        
        # TODO: Load state dict and handle errors
        try:
            self.backbone.load_state_dict(ckpt['model_state_dict'])
        except RuntimeError:
            print('Error loading pretrained weights')
            return

        
    def forward(self, x):
        """Forward pass
        
        Instructions:
        1. Extract features from backbone
        2. Handle different feature dimensions
        3. Project features to contrastive space
        4. Normalize projected features
        
        Returns:
            tuple: (features, projected_features)
        """
        # TODO: Implement forward pass
        features = self.backbone(x)
        projected_features = self.projector(features)
        projected_features = F.normalize(projected_features, dim=-1)
        return (features, projected_features)

class ContrastiveLearning(nn.Module):
    """Contrastive learning framework
    
    Instructions:
    1. Initialize encoders for each modality
    2. Implement contrastive loss calculation
    3. Handle both intra-modal and cross-modal pairs
    4. Support both training and inference modes
    
    Args:
        modalities (list): List of modalities to use
        feature_dim (int): Feature dimension
        temperature (float): Temperature for scaling
        num_classes (int): Number of classes for the classifier
    """
    def __init__(self, modalities, feature_dim, temperature=0.07, num_classes=7):
        super().__init__()
        # TODO: Initialize encoders and parameters
        pass
        
    def forward(self, batch):
        """Forward pass
        
        Instructions:
        1. Process each modality pair
        2. Calculate intra-modal and cross-modal losses
        3. Combine all losses with proper weighting
        4. Return features for downstream tasks
        
        Returns:
            dict: Contains loss, features, projections, and predictions
        """
        # TODO: Implement forward pass
        pass
    
    def info_nce_loss(self, features_1, features_2, labels):
        """Calculate improved InfoNCE loss
        
        Instructions:
        1. Compute similarity matrix
        2. Apply temperature scaling
        3. Handle positive and negative pairs
        4. Implement hard negative mining
        
        Returns:
            torch.Tensor: Calculated contrastive loss
        """
        # TODO: Implement InfoNCE loss
        pass
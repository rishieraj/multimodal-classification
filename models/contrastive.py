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
            self.backbone = VisionClassifier(7, backbone='fenet')
        elif modality == 'touch':
            self.backbone = TouchClassifier(7, backbone='fenet')
        else:
            self.backbone = AudioClassifier(7, backbone='fenet')

        self.modality = modality
        
        # Load pretrained weights
        self.load_pretrained_weights(modality)

        # Projection head
        self.projector = nn.Sequential(
            nn.Linear(7, output_dim),
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
        except RuntimeError as e:
            print(e)
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
        outputs = self.backbone(x)
        features = outputs['pred']
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
        for modality in modalities:
            setattr(self, f'{modality}_encoder', ModalityEncoder(modality, feature_dim, num_classes))
        
        self.temperature = temperature
        self.modalities = modalities
        
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
        modality_outputs = {}

        # Process each modality pair
        for modality in self.modalities:
            x1 = batch[f'{modality}_1']
            x2 = batch[f'{modality}_2']

            encoder = getattr(self, f'{modality}_encoder')

            feats_1, proj_1 = encoder({modality: x1})
            feats_2, proj_2 = encoder({modality: x2})

            modality_outputs[f'{modality}_1'] = (feats_1, proj_1)
            modality_outputs[f'{modality}_2'] = (feats_2, proj_2)

        # Calculate losses
        total_intra_loss = 0.0
        total_cross_loss = 0.0

        # Hyperparameters to weight the losses
        lambda_intra = 1.0
        lambda_cross = 1.0

        # Intra-modal losses
        for modality in self.modalities:
            _, proj_1 = modality_outputs[f'{modality}_1']
            _, proj_2 = modality_outputs[f'{modality}_2']

            loss_intra = self.info_nce_loss(proj_1, proj_2, labels=None)
            total_intra_loss += loss_intra

        # Cross-modal losses
        if len(self.modalities) > 1:
            # Simple approach: all pairs
            for i in range(len(self.modalities)):
                for j in range(i+1, len(self.modalities)):
                    mod_i = self.modalities[i]
                    mod_j = self.modalities[j]
                    
                    _, proj_i = modality_outputs[f'{mod_i}_1']
                    _, proj_j = modality_outputs[f'{mod_j}_1']

                    loss_cross = self.info_nce_loss(proj_i, proj_j, labels=None)
                    total_cross_loss += loss_cross

        # Combine weighted losses
        total_loss = lambda_intra * total_intra_loss + lambda_cross * total_cross_loss

        features_dict = {}
        projections_dict = {}
        for key, (feats, proj) in modality_outputs.items():
            features_dict[key] = feats
            projections_dict[key] = proj

        output = {
            'loss': total_loss,
            'features': features_dict,
            'projections': projections_dict
        }
        return output
    
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
        batch_size = features_1.shape[0]
        sim_matrix = torch.matmul(features_1, features_2.T) / self.temperature
        diag_mask = torch.eye(batch_size, dtype=torch.bool, device=features_1.device)

        # Forward hinge loss
        # Separating positive pairs
        pos_matrix = torch.diag(sim_matrix).unsqueeze(1)
        pos_matrix = pos_matrix.expand_as(sim_matrix)

        # Hinge loss
        loss_12 = torch.clamp(0.2 + sim_matrix - pos_matrix, min=0)
        loss_12.masked_fill_(diag_mask, 0.0)
        loss_12 = loss_12.sum() / (batch_size * (batch_size - 1))

        # Backward hinge loss
        # Separating positive pairs
        pos_matrix_T = torch.diag(sim_matrix.T).unsqueeze(0)
        pos_matrix_T = pos_matrix_T.expand_as(sim_matrix.T)

        # Hinge loss
        loss_21 = torch.clamp(0.2 + sim_matrix.T - pos_matrix_T, min=0)
        loss_21.masked_fill_(diag_mask, 0.0)
        loss_21 = loss_21.sum() / (batch_size * (batch_size - 1))

        return loss_12 + loss_21
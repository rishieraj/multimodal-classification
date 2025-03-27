import torch
import torch.nn as nn
import torch.nn.functional as F
from models.unimodal import VisionClassifier, TouchClassifier, AudioClassifier

class LateFusion(nn.Module):
    """Late fusion strategy for multimodal classification
    
    Instructions:
    1. Load pretrained unimodal models
    2. Freeze their parameters
    3. Implement fusion layer:
       - Concatenate predictions from each modality
       - Add MLP for final classification
    4. Calculate loss using CrossEntropyLoss
    """
    def __init__(self, input_dims, num_classes):
        super().__init__()
        # TODO: Initialize models and fusion layer
        self.vision_model = VisionClassifier(num_classes, backbone='fenet')
        self.touch_model = TouchClassifier(num_classes, backbone='fenet')
        self.audio_model = AudioClassifier(num_classes, backbone='fenet')

        # Load pretrained models
        self.load_pretrained_models()

        # Freeze unimodal models
        for model in [self.vision_model, self.touch_model, self.audio_model]:
            for param in model.parameters():
                param.requires_grad = False

        # Fusion layer
        self.fc = nn.Sequential(
            nn.Linear(11 * 11 * 11, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
        
    def load_pretrained_models(self):
        """Load pretrained unimodal models
        
        Instructions:
        1. Load checkpoints for each modality
        2. Update model state dictionaries
        3. Handle potential key mismatches
        """
        base_path = 'experiments/unimodal'
        
        # Load checkpoints
        vision_ckpt = torch.load(f'{base_path}/visual_exp/best_model.pth', weights_only=False)
        touch_ckpt = torch.load(f'{base_path}/touch_exp/best_model.pth', weights_only=False)
        audio_ckpt = torch.load(f'{base_path}/audio_exp/best_model.pth', weights_only=False)
        
        # TODO: Load state dicts into models
        self.vision_model.load_state_dict(vision_ckpt['model_state_dict'])
        self.touch_model.load_state_dict(touch_ckpt['model_state_dict'])
        self.audio_model.load_state_dict(audio_ckpt['model_state_dict'])

    def forward(self, modalities):
        """Forward pass
        
        Instructions:
        1. Get predictions from each modality
        2. Concatenate predictions
        3. Pass through fusion layer
        4. Calculate and return loss if training
        """
        # TODO: Implement forward pass
        # Get predictions from each modality
        vision_feats = self.vision_model.backbone(modalities['vision'])
        # Add attention layer for touch and audio modalities
        touch_feats = self.touch_model.backbone.forward_head(
            self.touch_model.attention(
                self.touch_model.backbone.forward_features(modalities['touch'])
            )
        )
        audio_feats = self.audio_model.backbone.forward_head(
            self.audio_model.attention(
                self.audio_model.backbone.forward_features(modalities['audio'])
            )
        )

        # Batch size
        batch_size = vision_feats.size(0)

        # Extend feature vectors
        vision_extnd = torch.ones(batch_size, 1, device=vision_feats.device)
        vision_feats = torch.cat([vision_feats, vision_extnd], dim=1)

        touch_extnd = torch.ones(batch_size, 1, device=touch_feats.device)
        touch_feats = torch.cat([touch_feats, touch_extnd], dim=1)
        
        audio_extnd = torch.ones(batch_size, 1, device=audio_feats.device)
        audio_feats = torch.cat([audio_feats, audio_extnd], dim=1)

        # 3D tensor fusion
        fused_feats = torch.einsum('bi,bj,bk->bijk', vision_feats, touch_feats, audio_feats)

        # Flatten and pass through MLP
        logits = self.fc(fused_feats.view(batch_size, -1))

        # Return predictions and loss if training
        output = {'pred': logits}
        if 'label' in modalities:
            output['loss'] = F.cross_entropy(logits, modalities['label'])
        
        return output

class AttentionFusion(nn.Module):
    """Attention-based fusion for multimodal classification
    
    Instructions:
    1. Load pretrained unimodal models
    2. Freeze their parameters
    3. Project each modality to common space
    4. Implement multi-head attention fusion
    5. Add final classification layer
    """
    def __init__(self, input_dims, num_heads, num_classes):
        super().__init__()
        # TODO: Initialize models and fusion components
        self.vision_model = VisionClassifier(num_classes, backbone='fenet')
        self.touch_model = TouchClassifier(num_classes, backbone='fenet')
        self.audio_model = AudioClassifier(num_classes, backbone='fenet')

        self.load_pretrained_models()

        # Freeze unimodal models
        for model in [self.vision_model, self.touch_model, self.audio_model]:
            for param in model.parameters():
                param.requires_grad = False

        # Project to common feature space
        self.feature_proj = nn.Linear(10, num_heads * num_classes)

        # Multi-head attention fusion
        self.attention = nn.MultiheadAttention(embed_dim=num_classes * num_heads, num_heads=num_heads)

        # Fusion layer
        self.fc = nn.Sequential(
            nn.Linear(3 * num_classes * num_heads, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
        
    def load_pretrained_models(self):
        """Load pretrained unimodal models
        
        Instructions:
        1. Load checkpoints for each modality
        2. Update model state dictionaries
        3. Handle potential key mismatches
        """
        base_path = 'experiments/unimodal'
        
        vision_ckpt = torch.load(f'{base_path}/visual_exp/best_model.pth', weights_only=False)
        touch_ckpt = torch.load(f'{base_path}/touch_exp/best_model.pth', weights_only=False)
        audio_ckpt = torch.load(f'{base_path}/audio_exp/best_model.pth', weights_only=False)
        
        # TODO: Load state dicts into models
        self.vision_model.load_state_dict(vision_ckpt['model_state_dict'])
        self.touch_model.load_state_dict(touch_ckpt['model_state_dict'])
        self.audio_model.load_state_dict(audio_ckpt['model_state_dict'])

    def forward(self, modalities):
        """Forward pass
        
        Instructions:
        1. Get predictions from each modality
        2. Project to common feature space
        3. Apply multi-head attention
        4. Fuse attended features
        5. Calculate and return loss if training
        """
        # TODO: Implement forward pass
        # Get predictions from each modality
        vision_feats = self.feature_proj(self.vision_model.backbone(modalities['vision']))
        # Add attention layer for touch and audio modalities
        touch_feats = self.feature_proj(
            self.touch_model.backbone.forward_head(
                self.touch_model.attention(
                    self.touch_model.backbone.forward_features(modalities['touch'])
                )
            )
        )
        audio_feats = self.feature_proj(
            self.audio_model.backbone.forward_head(
                self.audio_model.attention(
                    self.audio_model.backbone.forward_features(modalities['audio'])
                )
            )
        )

        # Project to common feature space
        kv = torch.stack([vision_feats, touch_feats, audio_feats], dim=0)

        # Apply multi-head attention
        attended, _ = self.attention(kv, kv, kv)

        # Fuse attended features
        fusion_preds = torch.cat([attended[0], attended[1], attended[2]], dim=1)
        logits = self.fc(fusion_preds)

        # Return predictions and loss if training
        output = {'pred': logits}
        if 'label' in modalities:
            output['loss'] = F.cross_entropy(logits, modalities['label'])
        
        return output



        
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class FAP(nn.Module):
    """Fractal Analysis Pooling Module"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # Adaptive pooling to ensure consistent input size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((15, 15))
        self.fc = nn.Linear(225, dim)  # 15x15 -> dim
        
    def forward(self, x):
        # First ensure input is 15x15
        x = self.adaptive_pool(x)  # (B, C, 15, 15)
        bs, c, h, w = x.shape
        x = x.view(bs * c, -1)  # Flatten to (B*C, 225)
        x = self.fc(x)  # Project to (B*C, dim)
        x = x.view(bs, c * self.dim)  # Reshape to (B, C*dim)
        return x

class ModifiedResNet(nn.Module):
    """Modified ResNet18 for material classification"""
    def __init__(self, in_channels=3, pretrained=True):
        super().__init__()
        resnet = models.resnet18(weights='IMAGENET1K_V1') if pretrained else models.resnet18(weights=None)
        resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])

    def forward(self, x):
        return self.resnet(x)

class UpsampleBlock(nn.Module):
    """Upsampling block with depthwise separable convolution"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.upsample = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.relu(x)
        x = self.upsample(x)
        return x

class FENet(nn.Module):
    """Feature Enhancement Network (FENet)

    This network is designed as an alternative to ResNet for material classification.
    Key motivations and features:
    1. Enhanced Feature Learning:
       - Uses fractal analysis to capture material texture patterns
       - Combines global and local feature representations
       - Better suited for material-specific characteristics
    
    2. Architecture Components:
       a) ResNet18 Backbone:
          - Provides strong base feature extraction
          - Pretrained on ImageNet for transfer learning
       
       b) Fractal Analysis Pooling (FAP):
          - Captures multi-scale texture information
          - Computes fractal dimensions of feature maps
          - More sensitive to material surface patterns
       
       c) Bilinear Pooling:
          - Combines FAP and global features
          - Models feature interactions
          - Enhances discriminative power
    
    3. Advantages over ResNet:
       - Better texture representation
       - Material-specific feature enhancement
       - Explicit multi-scale analysis
    
    Instructions:
    1. Implement a lightweight network focusing on feature enhancement
    2. Use depthwise separable convolutions for efficiency
    3. Add squeeze-and-excitation blocks for channel attention
    4. Structure should have:
       - Initial conv layer
       - 4 feature enhancement blocks
       - Global average pooling
    """
    def __init__(self, in_channels=3, fap_dim=15, pretrained=True):
        super().__init__()
        # TODO: Initialize backbone (ResNet18)
        # Hint: Use pretrained ResNet18, modify first conv layer based on input channels, 
        # and remove FC and avgpool
        self.backbone = ModifiedResNet(in_channels, pretrained)
        
        # TODO: Initialize upsampling layer
        # Hint: Use ConvTranspose2d for upsampling features
        self.upsample_blocks = nn.Sequential(
            UpsampleBlock(512, 64)
        )
        
        # TODO: Initialize pre-FAP convolution
        # Hint: Reduce channels to 3 with Conv2d before FAP
        self.pre_fap_conv = nn.Conv2d(64, 3, kernel_size=1)
        
        # TODO: Initialize FAP module
        # Hint: Implement or use provided FAP class
        self.fap = FAP(fap_dim)
        
        # TODO: Initialize pooling and feature processing
        # Hint: Combine average pooling and bilinear models
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.bilinear = nn.Bilinear(64, 3*fap_dim, 64)
        
        # TODO: Initialize final MLP
        # Hint: Use multiple FC layers with normalization
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 10),
            nn.Dropout(0.5)
        )

    def forward_features(self, x):
        x = self.backbone(x)
        x = self.upsample_blocks(x)
        return x
    
    def forward_head(self, x):
        global_feat = self.global_pool(x).view(x.size(0), -1)
        x_fap = self.pre_fap_conv(x)
        fap_feat = self.fap(x_fap)
        feat = self.bilinear(global_feat, fap_feat)
        out = self.fc(feat)
        return out

    def forward(self, x):
        # TODO: Implement forward pass
        # 1. Get backbone features
        # 2. Apply upsampling to the features
        # 3. Process with FAP
        # 4. Apply bilinear pooling
        # 5. Final MLP processing
        x = self.forward_features(x)
        out = self.forward_head(x)
        return out

class SpatialAttention(nn.Module):
    """Spatial Attention Mechanism for touch input"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg_pool, max_pool], dim=1)
        attn = self.conv(attn)
        attn = self.sigmoid(attn)
        return x * attn

class FrequencyAttention(nn.Module):
    """Frequency Attention Mechanism for audio input"""
    def __init__(self, channels):
        super().__init__()
        self.depthwise_conv = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1, groups=channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1. 
        x_avg = torch.mean(x, dim=3)
        attn = self.depthwise_conv(x_avg)
        attn = self.sigmoid(attn)
        attn = attn.unsqueeze(3)
        return x * attn

class VisionClassifier(nn.Module):
    """Vision-based material classifier
    
    Instructions:
    1. Support both ResNet and FENet backbones
    2. Add classification head with dropout (p=0.5)
    3. Return predictions and loss in a dictionary
    4. Handle both training and inference modes
    """
    def __init__(self, num_classes, backbone='resnet'):
        super().__init__()
        # TODO: Initialize backbone and classifier
        if backbone == 'resnet':
            self.backbone = ModifiedResNet(in_channels=3, pretrained=True)
            self.feature_dim = 512
        else:
            self.backbone = FENet(in_channels=3, fap_dim=15, pretrained=True)
            self.feature_dim = 10
        
        # Add classifier with dropout
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, num_classes),
            nn.Dropout(p=0.5)
        )

    def forward(self, batch):
        # TODO: Implement forward pass
        x = batch['vision']
        feat = self.backbone(x)
        logits = self.classifier(feat)

        output = {'pred': logits}
        if 'label' in batch:
            output['loss'] = F.cross_entropy(logits, batch['label'])
        return output

class TouchClassifier(nn.Module):
    """Touch-based material classifier
    
    Instructions:
    1. Similar structure to VisionClassifier
    2. Modify first conv layer for tactile input
    3. Add spatial attention mechanism
    4. Support both ResNet and FENet backbones
    """
    def __init__(self, num_classes, backbone='resnet'):
        super().__init__()
        # TODO: Initialize network
        if backbone == 'resnet':
            self.backbone = ModifiedResNet(in_channels=3, pretrained=True)
            self.feature_dim = 512
        else:
            self.backbone = FENet(in_channels=3, fap_dim=15, pretrained=True)
            self.feature_dim = 10
        
        # Add spatial attention
        self.attention = SpatialAttention()

        # Add classifier with dropout
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, num_classes),
            nn.Dropout(p=0.5)
        )

    def forward(self, batch):
        # TODO: Implement forward pass
        x = batch['touch']
        # 1. Get the feature map from the backbone 
        feat_map = self.backbone.forward_features(x)

        # 2. Apply spatial attention mechanism
        feat_map = self.attention(feat_map)
        
        # 3. Get the feature embeddings from the FENet forward head
        feat = self.backbone.forward_head(feat_map)

        # 4. Get the logits from the classifier
        logits = self.classifier(feat)

        output = {'pred': logits}
        if 'label' in batch:
            output['loss'] = F.cross_entropy(logits, batch['label'])
        return output

class AudioClassifier(nn.Module):
    """Audio-based material classifier
    
    Instructions:
    1. Similar structure to VisionClassifier
    2. Modify first conv layer for spectrogram input (1 channel)
    3. Add frequency attention mechanism
    4. Support both ResNet and FENet backbones
    """
    def __init__(self, num_classes, backbone='resnet'):
        super().__init__()
        # TODO: Initialize network
        if backbone == 'resnet':
            self.backbone = ModifiedResNet(in_channels=1, pretrained=True)
            self.feature_dim = 512
        else:
            self.backbone = FENet(in_channels=1, fap_dim=15, pretrained=True)
            self.feature_dim = 10

        self.attention = FrequencyAttention(64)
        
        # Add classifier with dropout
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, num_classes),
            nn.Dropout(p=0.5)
        )

    def forward(self, batch):
        # TODO: Implement forward pass
        x = batch['audio']
        # 1. Get the feature map from the backbone 
        feat_map = self.backbone.forward_features(x)

        # 2. Apply spatial attention mechanism
        feat_map = self.attention(feat_map)
        
        # 3. Get the feature embeddings from the FENet forward head
        feat = self.backbone.forward_head(feat_map)

        # 4. Get the logits from the classifier
        logits = self.classifier(feat)

        output = {'pred': logits}
        if 'label' in batch:
            output['loss'] = F.cross_entropy(logits, batch['label'])
        return output
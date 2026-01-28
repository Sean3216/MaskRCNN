import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

import torchvision
from torchvision.models.detection import (
    maskrcnn_resnet50_fpn_v2,
    MaskRCNN_ResNet50_FPN_V2_Weights,
    mask_rcnn,
    faster_rcnn,
)

import numpy as np


def get_maskrcnn_model(num_classes, pretrained=True):
    model = maskrcnn_resnet50_fpn_v2(
        weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None,
        min_size=480,
        max_size=800,
    )
    # replace box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    # replace mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 128
    model.roi_heads.mask_predictor = mask_rcnn.MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model

class AnomalyAwareMaskRCNN(nn.Module):
    """
    Mask R-CNN wrapper that prepends a learnable adapter layer.
    Input:  4 channels (3 RGB + 1 Heatmap)
    Output: 3 channels (Passed to standard Pre-trained ResNet)
    """
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        self.model = get_maskrcnn_model(num_classes=num_classes, pretrained=pretrained)
        
        # 4 channels in -> 3 channels out. 
        # Using 1x1 convolution acts as a pixel-wise weighted channel mixer
        # to fuse the heatmap into the RGB channels while preserving spatial dims.
        self.input_adapter = nn.Conv2d(4, 3, kernel_size=1)
        
        # Initialize weights for stability (optional but recommended)
        # We start with weights close to identity for RGB and small for Heatmap
        # so training doesn't diverge at the start.
        nn.init.kaiming_normal_(self.input_adapter.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, images, heatmaps, targets=None):
        """
        images: List[Tensor] of shape (3, H, W)
        heatmaps: List[Tensor] of shape (1, H, W) or (H, W)
        targets: List[Dict] (Standard Mask R-CNN targets)
        """
        adapted_images = []

        # Process each image individually to handle varying image sizes 
        # (A standard feature of torchvision detection models)
        for img, hmap in zip(images, heatmaps):
            # 1. Ensure consistency
            # Move heatmap to same device/dtype as image
            hmap = hmap.to(device=img.device, dtype=img.dtype)
            
            # Ensure hmap has channel dimension: (H, W) -> (1, H, W)
            if hmap.dim() == 2:
                hmap = hmap.unsqueeze(0)
            
            # Check for size mismatch and resize heatmap if necessary
            if hmap.shape[-2:] != img.shape[-2:]:
                hmap = F.interpolate(hmap.unsqueeze(0), size=img.shape[-2:], mode='bilinear', align_corners=False).squeeze(0)

            # 2. Concatenate: (3, H, W) + (1, H, W) -> (4, H, W)
            x = torch.cat([img, hmap], dim=0)

            # 3. Apply Adapter
            # Conv2d expects (B, C, H, W), so we unsqueeze -> conv -> squeeze
            x = x.unsqueeze(0)       # (1, 4, H, W)
            x = self.input_adapter(x) # (1, 3, H, W)
            x = x.squeeze(0)         # (3, H, W)

            adapted_images.append(x)

        # 4. Pass adapted 3-channel images to the original backbone
        return self.model(adapted_images, targets)
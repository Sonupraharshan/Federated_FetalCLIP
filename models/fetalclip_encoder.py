# models/fetalclip_encoder.py
import torch
import torch.nn as nn
import json
import open_clip
from pathlib import Path

class FetalCLIPEncoder(nn.Module):
    """
    FetalCLIP Encoder - Vision encoder from the FetalCLIP foundation model
    
    Uses open_clip to load the official FetalCLIP model.
    Reference: https://github.com/BioMedIA-MBZUAI/FetalCLIP
    
    To use:
    1. Download FetalCLIP_weights.pt from SharePoint link in GitHub repo
    2. Download FetalCLIP_config.json from the GitHub repo
    3. Place both files in project root or specify paths in config.py
    """
    
    def __init__(self, feature_dim=512, pretrained_path=None, config_path=None, device='cpu'):
        super().__init__()
        self.device = device
        self.feature_dim = feature_dim
        self.model = None
        self.preprocess_train = None
        self.preprocess_test = None
        
        # Try to load FetalCLIP model
        if pretrained_path and config_path:
            self._load_fetalclip(pretrained_path, config_path)
        else:
            # Fallback to placeholder if paths not provided
            self._init_placeholder_backbone()
        
        self.to(device)
    
    def _load_fetalclip(self, pretrained_path, config_path):
        """
        Load actual FetalCLIP model from pretrained weights and config
        """
        try:
            # Load configuration
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            
            # Register model configuration with open_clip
            open_clip.factory._MODEL_CONFIGS["FetalCLIP"] = config_dict
            
            # Load pretrained FetalCLIP model and preprocessing
            self.model, self.preprocess_train, self.preprocess_test = open_clip.create_model_and_transforms(
                "FetalCLIP",
                pretrained=pretrained_path,
                device=self.device
            )
            self.model.eval()
            self.out_dim = self.model.visual.output_dim  # Get actual output dimension
            print(f"✅ FetalCLIP loaded successfully!")
            print(f"   Config: {config_path}")
            print(f"   Weights: {pretrained_path}")
            print(f"   Visual encoder output dim: {self.out_dim}")
            
        except Exception as e:
            print(f"⚠️  Failed to load FetalCLIP: {e}")
            print(f"   Falling back to placeholder backbone")
            self._init_placeholder_backbone()
    
    def _init_placeholder_backbone(self):
        """
        Placeholder backbone for when FetalCLIP weights are not available
        Replace this with actual FetalCLIP architecture when weights are ready
        """
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.proj = nn.Linear(64, self.feature_dim)
        self.out_dim = self.feature_dim
        print(f"⚠️  Using placeholder backbone (feature_dim={self.feature_dim})")
    
    def forward(self, x):
        """
        Forward pass through encoder
        
        Args:
            x: (B, C, H, W) - Batch of images (C can be 1 for grayscale or 3 for RGB)
        
        Returns:
            embeddings: (B, output_dim) - Image embeddings
        """
        if self.model is not None:
            # Use actual FetalCLIP model
            # FetalCLIP expects 3-channel RGB images, so convert grayscale to RGB if needed
            if x.size(1) == 1:
                # Grayscale (1-channel) -> RGB (3-channel) by repeating
                x = x.repeat(1, 3, 1, 1)
            
            with torch.no_grad():
                embeddings = self.model.encode_image(x)
            return embeddings
        else:
            # Use placeholder backbone
            feats = self.backbone(x)           # (B, C, 1, 1)
            feats = feats.view(feats.size(0), -1)  # (B, C)
            emb = self.proj(feats)             # (B, feature_dim)
            return emb
    
    def freeze(self):
        """Freeze all parameters (for inference only)"""
        for p in self.parameters():
            p.requires_grad = False
    
    def unfreeze(self):
        """Unfreeze all parameters (for fine-tuning)"""

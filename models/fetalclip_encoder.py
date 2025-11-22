# models/fetalclip_encoder.py
import torch
import torch.nn as nn

class FetalCLIPEncoder(nn.Module):
    def __init__(self, feature_dim=512, pretrained_path=None, device='cpu'):
        super().__init__()
        # Example small backbone - replace with your actual FetalCLIP implementation
        # The key is: encoder(x) -> (B, D) or (B, N, D) depending on your FetalCLIP design
        self.device = device
        self.feature_dim = feature_dim
        # For example purposes a small conv backbone - replace with FetalCLIP model body
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.proj = nn.Linear(64, feature_dim)
        self.out_dim = feature_dim

        if pretrained_path:
            self.load_pretrained(pretrained_path)
        self.to(device)

    def forward(self, x):
        # expects x: (B,1,H,W)
        feats = self.backbone(x)      # (B, C, 1, 1)
        feats = feats.view(feats.size(0), -1)  # (B, C)
        emb = self.proj(feats)        # (B, feature_dim)
        return emb

    def load_pretrained(self, path):
        sd = torch.load(path, map_location=self.device)
        # adjust according to saved structure; attempt to load safely
        try:
            self.load_state_dict(sd)
            print(f"Loaded pretrained encoder from {path}")
        except Exception as e:
            print(f"Warning: couldn't fully load pretrained encoder: {e}")

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True

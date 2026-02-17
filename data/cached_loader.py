"""
Cached Features Data Loader
Loads pre-computed FetalCLIP embeddings instead of raw images
Much faster than computing features on-the-fly
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json


class CachedFeaturesDataset(Dataset):
    """Load pre-computed features and labels from .npz files"""
    
    def __init__(self, features, labels):
        self.features = torch.from_numpy(features).float()
        self.labels = torch.from_numpy(labels).long()
        self.df = None  # Placeholder for compatibility
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def load_cached_features(num_clients=4, batch_size=128):
    """
    Load pre-computed features from disk
    
    Returns:
        client_loaders: list of DataLoaders for each client
        test_loader: DataLoader for test set
        label_map: dict mapping class names to IDs
    """
    
    features_dir = Path("./data/cached_features")
    
    # Check if features exist
    if not features_dir.exists():
        raise FileNotFoundError(
            f"❌ Features not found at {features_dir}\n"
            f"   Run: python extract_features.py"
        )
    
    # Load label map
    label_map_path = features_dir / "label_map.json"
    with open(label_map_path, 'r') as f:
        label_map = json.load(f)
    
    print(f"\n✅ Loaded label map: {label_map}")
    
    # Load client features
    client_loaders = []
    for client_id in range(num_clients):
        npz_path = features_dir / f"client_{client_id}_features.npz"
        
        if not npz_path.exists():
            print(f"⚠️  Warning: {npz_path} not found, skipping client {client_id}")
            continue
        
        data = np.load(npz_path)
        features = data['features']  # (N, 768)
        labels = data['labels']      # (N,)
        
        dataset = CachedFeaturesDataset(features, labels)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        client_loaders.append(loader)
        
        print(f"   📱 Client {client_id}: {len(dataset)} samples")
    
    # Load test features
    test_npz_path = features_dir / "test_features.npz"
    if not test_npz_path.exists():
        raise FileNotFoundError(f"❌ Test features not found at {test_npz_path}")
    
    test_data = np.load(test_npz_path)
    test_features = test_data['features']
    test_labels = test_data['labels']
    
    test_dataset = CachedFeaturesDataset(test_features, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"   📊 Test set: {len(test_dataset)} samples")
    
    return client_loaders, test_loader, label_map

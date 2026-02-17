"""
Feature Extraction Script
Pre-compute FetalCLIP embeddings for all clients to speed up training
Usage: python extract_features.py
"""

import os
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import config as cfg
from data.dataset_loader import make_dataloaders
from models.fetalclip_encoder import FetalCLIPEncoder


def extract_and_save_features():
    """Extract FetalCLIP features for each client and save as cached embeddings"""
    
    print("\n" + "="*70)
    print("🚀 FetalCLIP Feature Extraction")
    print("="*70)
    
    # Create output directory
    features_dir = Path("./data/cached_features")
    features_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Features will be saved to: {features_dir}")
    
    # Load encoder
    print("\n📦 Loading FetalCLIP encoder...")
    encoder = FetalCLIPEncoder(
        feature_dim=cfg.FEATURE_DIM,
        pretrained_path=cfg.FETALCLIP_WEIGHTS_PATH,
        config_path=cfg.FETALCLIP_CONFIG_PATH,
        device=cfg.DEVICE
    )
    encoder.eval()
    encoder.freeze()
    print(f"   ✅ Encoder loaded (output_dim={encoder.out_dim})")
    
    # Load data
    print("\n📊 Loading datasets...")
    client_loaders, test_loader, label_map = make_dataloaders(
        cfg.DATA_PATH,
        num_clients=cfg.NUM_CLIENTS,
        iid=True,
        batch_size=cfg.BATCH_SIZE,
        test_split=0.2
    )
    print(f"   ✅ Loaded {len(client_loaders)} clients")
    print(f"   ✅ Label map: {label_map}")
    
    # Save label map
    import json
    with open(features_dir / "label_map.json", "w") as f:
        json.dump(label_map, f)
    
    # Extract features for each client
    print("\n🔄 Extracting features...")
    for client_id, client_loader in enumerate(client_loaders):
        print(f"\n   📱 Client {client_id + 1}/{len(client_loaders)}...")
        
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(tqdm(client_loader, desc=f"Client {client_id+1}", leave=False)):
                images = images.to(cfg.DEVICE)
                
                # Extract features
                features = encoder(images)  # (B, 768)
                
                all_features.append(features.cpu().numpy())
                all_labels.append(labels.numpy())
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"      Batch {batch_idx + 1}/{len(client_loader)}")
        
        # Concatenate all batches
        features_array = np.concatenate(all_features, axis=0)  # (N, 768)
        labels_array = np.concatenate(all_labels, axis=0)      # (N,)
        
        # Save as .npz
        save_path = features_dir / f"client_{client_id}_features.npz"
        np.savez_compressed(
            save_path,
            features=features_array,
            labels=labels_array
        )
        
        print(f"   ✅ Saved: {save_path}")
        print(f"      Features shape: {features_array.shape}")
        print(f"      Labels shape: {labels_array.shape}")
    
    # Extract features for test set
    print(f"\n   📊 Test set...")
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(test_loader, desc="Test set", leave=False)):
            images = images.to(cfg.DEVICE)
            features = encoder(images)
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())
    
    features_array = np.concatenate(all_features, axis=0)
    labels_array = np.concatenate(all_labels, axis=0)
    
    save_path = features_dir / f"test_features.npz"
    np.savez_compressed(save_path, features=features_array, labels=labels_array)
    
    print(f"   ✅ Saved: {save_path}")
    print(f"      Features shape: {features_array.shape}")
    print(f"      Labels shape: {labels_array.shape}")
    
    print("\n" + "="*70)
    print("✅ Feature extraction complete!")
    print(f"📁 All features saved to: {features_dir}")
    print("\n🚀 Next step: Run training with cached features")
    print("   python main.py --use_cached_features --head mlp --rounds 10")
    print("="*70 + "\n")


if __name__ == '__main__':
    extract_and_save_features()

#!/usr/bin/env python
"""
🚀 COMPLETE END-TO-END WORKFLOW
===============================
This script shows the complete workflow:
1. Data is ALREADY split into clients (by make_dataloaders)
2. Extract FetalCLIP features for EACH client
3. Train & Finetune on cached features with CFL, QFL, DP support

Usage:
    python setup_cached_training.py          # Show status
    python setup_cached_training.py --extract  # Extract features
    python setup_cached_training.py --train    # Train on cached features
"""

import argparse
import subprocess
import sys
from pathlib import Path


def show_status():
    """Show what features are cached"""
    features_dir = Path("./data/cached_features")
    
    print("\n" + "="*70)
    print("📊 CACHED FEATURES STATUS")
    print("="*70)
    
    if not features_dir.exists():
        print("❌ No cached features found")
        print("   Run: python setup_cached_training.py --extract")
        return False
    
    # List cached files
    cached_files = list(features_dir.glob("*.npz"))
    print(f"\n✅ Found {len(cached_files)} cached feature files:")
    for f in sorted(cached_files):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"   📦 {f.name} ({size_mb:.1f} MB)")
    
    print("\n✅ Ready to train with: python setup_cached_training.py --train")
    return True


def extract_features():
    """Step 1: Extract FetalCLIP features for each client"""
    print("\n" + "="*70)
    print("⚙️  STEP 1: EXTRACT FETALCLIP FEATURES FOR EACH CLIENT")
    print("="*70)
    
    print("""
This process:
1. Loads FetalCLIP encoder
2. For EACH client:
   - Takes their training data
   - Extracts 768-dim embeddings
   - Saves as client_X_features.npz
3. Extracts test set features
4. Saves label mappings

⏱️  Estimated time: 5-10 minutes
""")
    
    try:
        subprocess.run([sys.executable, "extract_features.py"], check=True)
        print("\n✅ Feature extraction complete!")
        show_status()
    except subprocess.CalledProcessError:
        print("\n❌ Feature extraction failed!")
        sys.exit(1)


def train_with_cached_features():
    """Step 2: Train on cached features with CFL, QFL, DP support"""
    print("\n" + "="*70)
    print("⚙️  STEP 2: TRAIN ON CACHED FEATURES")
    print("="*70)
    
    print("""
Example commands (choose one):

🚀 BASIC TRAINING (Fast, ~10-15 min):
    python main.py --use_cached_features --head linear --rounds 10

🚀 WITH MLP HEAD (Medium speed):
    python main.py --use_cached_features --head mlp --rounds 10 --local_epochs 2

🚀 WITH CLUSTERED FEDERATED LEARNING (CFL):
    python main.py --use_cached_features --head mlp --use_cfl --rounds 10

🚀 WITH QUANTUM FEDERATED LEARNING (QFL):
    python main.py --use_cached_features --head mlp --use_qfl --rounds 10

🚀 WITH DIFFERENTIAL PRIVACY (DP):
    python main.py --use_cached_features --head mlp --use_dp --rounds 10

🚀 ALL FEATURES COMBINED (CFL + DP):
    python main.py --use_cached_features --head mlp --use_cfl --use_dp --rounds 10

🚀 NON-IID DATA DISTRIBUTION:
    python main.py --use_cached_features --head mlp --split noniid --rounds 10
""")
    
    # Ask user which command to run
    cmd = input("\n📝 Enter your training command (or press Enter for default): ").strip()
    
    if not cmd:
        cmd = "python main.py --use_cached_features --head linear --rounds 5"
        print(f"Using default: {cmd}")
    
    try:
        subprocess.run(cmd, shell=True, check=True)
        print("\n✅ Training complete!")
    except subprocess.CalledProcessError:
        print("\n❌ Training failed!")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Federated Learning with Cached Features")
    parser.add_argument("--extract", action="store_true", help="Extract FetalCLIP features")
    parser.add_argument("--train", action="store_true", help="Train on cached features")
    parser.add_argument("--all", action="store_true", help="Extract then train")
    
    args = parser.parse_args()
    
    if args.all:
        extract_features()
        print("\n")
        train_with_cached_features()
    elif args.extract:
        extract_features()
    elif args.train:
        if not show_status():
            print("\n⚠️  Please extract features first!")
            sys.exit(1)
        train_with_cached_features()
    else:
        # Show menu
        print("\n" + "="*70)
        print("🚀 FEDERATED LEARNING WITH CACHED FEATURES")
        print("="*70)
        print("""
WORKFLOW:
1️⃣  Extract Features  → python setup_cached_training.py --extract
2️⃣  Train Models     → python setup_cached_training.py --train
3️⃣  Or do both      → python setup_cached_training.py --all

CURRENT STATUS:
""")
        show_status()
        print("\nUSAGE:")
        print("    python setup_cached_training.py --extract    # Extract FetalCLIP features")
        print("    python setup_cached_training.py --train      # Train on cached features")
        print("    python setup_cached_training.py --all        # Extract then train")



if __name__ == '__main__':
    main()

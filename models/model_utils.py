# models/model_utils.py
import torch
import os
from pathlib import Path
from config import CHECKPOINT_DIR

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def save_checkpoint(state_dict, name):
    path = Path(CHECKPOINT_DIR) / f"{name}.pt"
    torch.save(state_dict, path)
    return str(path)

def load_checkpoint_to(model, path, device='cpu'):
    model.load_state_dict(torch.load(path, map_location=device))
    return model

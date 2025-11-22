# data/dataset_loader.py
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
import random

# ============================================================
# 📦 Dataset Class
# ============================================================
class XLSXImageDataset(Dataset):
    """
    Reads labels.xlsx with at least these columns:
        ['Image_name', 'Train', 'Plane'] or ['Image_name', 'label']
    Expects all images in <data_path>/images/
    """
    def __init__(self, df, img_dir, transform=None, label_map=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.label_map = label_map or self._build_label_map()
        self.labels = [self.label_map[lbl] for lbl in self.df['Plane']]

    def _build_label_map(self):
        unique_labels = sorted(self.df['Plane'].unique())
        return {lbl: i for i, lbl in enumerate(unique_labels)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['Image_name']
        label = self.labels[idx]
        img_path = os.path.join(self.img_dir, f"{img_name}.png")

        # Load grayscale image
        image = Image.open(img_path).convert("L")

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        return image, torch.tensor(label, dtype=torch.long)


# ============================================================
# 🧩 Data Loading + Client Split
# ============================================================
def make_dataloaders(data_path, num_clients=5, iid=True, batch_size=32, test_split=0.2):
    """
    Loads dataset from data_path containing:
        - images/ folder
        - labels.xlsx (with 'Image_name' and 'Plane' columns)
    Returns:
        client_loaders: list of DataLoaders (train per client)
        test_loader: global test DataLoader
        label_map: {id: label_name}
    """

    labels_path = os.path.join(data_path, "labels.xlsx")
    img_dir = os.path.join(data_path, "images")

    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Missing labels.xlsx at {labels_path}")
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Missing image folder: {img_dir}")

    df = pd.read_excel(labels_path)
    print(f"Loading data from: {data_path}")
    print(f"Initial DataFrame size: {len(df)}")
    print(f"Available columns: {list(df.columns)}")

    # Remove invalid rows
    df = df.dropna(subset=['Image_name', 'Plane'])
    print(f"Cleaned DataFrame size: {len(df)}")

    # Split train/test
    train_df, test_df = train_test_split(df, test_size=test_split, stratify=df['Plane'], random_state=42)

    # Label map (string -> int)
    unique_labels = sorted(train_df['Plane'].unique())
    label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
    inv_label_map = {v: k for k, v in label_map.items()}
    print("Label map:", inv_label_map)

    # ============================================================
    # 🔁 Augmentations
    # ============================================================
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # ============================================================
    # 🧠 Dataset creation
    # ============================================================
    train_dataset = XLSXImageDataset(train_df, img_dir, transform=train_transforms, label_map=label_map)
    test_dataset = XLSXImageDataset(test_df, img_dir, transform=val_transforms, label_map=label_map)

    # ============================================================
    # 🤝 Federated Split
    # ============================================================
    if iid:
        # IID split: random shuffle across clients
        indices = np.arange(len(train_dataset))
        np.random.shuffle(indices)
        splits = np.array_split(indices, num_clients)
    else:
        # Non-IID: group by class, then assign by class clusters
        class_indices = {c: np.where(np.array(train_dataset.labels) == c)[0] for c in np.unique(train_dataset.labels)}
        splits = [[] for _ in range(num_clients)]
        for c, idxs in class_indices.items():
            np.random.shuffle(idxs)
            chunks = np.array_split(idxs, num_clients)
            for i in range(num_clients):
                splits[i].extend(chunks[i])
        splits = [np.array(s) for s in splits]

    # Create per-client DataLoaders
    client_loaders = []
    for i, split in enumerate(splits):
        subset = Subset(train_dataset, split)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=0)
        client_loaders.append(loader)
        print(f"Client {i+1}: {len(subset)} samples")

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return client_loaders, test_loader, inv_label_map

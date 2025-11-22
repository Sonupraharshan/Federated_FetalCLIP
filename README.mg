# Federated_FetalCLIP — Stage 1

## Purpose
Federated fine-tuning (FedAvg) of a FetalCLIP encoder with several classification heads.
Supports IID and Non-IID (grouped by Patient_num) client splits. Uses images in ./data/images/ and labels.xlsx.

## Data format expected
- `data/images/` folder containing .png images.
- `data/labels.xlsx` with columns (at least): `image_name`, `Patient_num`, `Plane`, `Train`.
  - `image_name` matches the image filename (e.g., 0001.png)
  - `Plane` is the target label (text)
  - `Train` is optional (1/0) — if present we will respect it for train/test split; otherwise automatic split used.

## Install
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

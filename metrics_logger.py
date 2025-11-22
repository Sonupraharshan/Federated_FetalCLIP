# metrics_logger.py
import csv
from pathlib import Path

METRICS_CSV = Path('metrics.csv')

def init_metrics():
    with open(METRICS_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['head', 'split', 'round', 'acc', 'f1'])

def log_metric(head, split, round_idx, acc, f1):
    with open(METRICS_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([head, split, round_idx, f"{acc:.4f}", f"{f1:.4f}"])

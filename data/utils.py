# data/utils.py
import numpy as np
from collections import defaultdict

def iid_split_indices(n_samples, num_clients, seed=42):
    np.random.seed(seed)
    idx = np.arange(n_samples)
    np.random.shuffle(idx)
    return np.array_split(idx, num_clients)

def noniid_split_indices_by_patient(train_df, num_clients):
    """
    train_df: pandas DataFrame corresponding to the training rows (ordered same as master_dataset)
    We'll group by Patient_num and distribute patient groups to clients (each client gets some patients)
    Returns: list of arrays of indices (into train_df)
    """
    patient_groups = train_df.groupby('Patient_num').indices  # dict: patient_num -> list of indices
    # sort patients by size to balance
    patients = sorted(patient_groups.keys(), key=lambda k: len(patient_groups[k]), reverse=True)
    client_idxs = [[] for _ in range(num_clients)]
    # round-robin assign biggest patient groups first
    for i, p in enumerate(patients):
        client_idxs[i % num_clients].extend(patient_groups[p].tolist())
    # convert to numpy arrays
    return [np.array(lst) for lst in client_idxs]

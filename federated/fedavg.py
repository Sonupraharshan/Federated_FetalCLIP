# federated/fedavg.py
import copy
import torch

def fedavg(client_states, weights):
    """
    client_states: list of state_dicts
    weights: list of floats (same length)
    Returns aggregated state_dict
    """
    total = float(sum(weights))
    if total == 0:
        raise ValueError("Sum of weights must be > 0")

    agg = {}
    # accumulate only floating tensors; copy non-float tensors once
    for sd, w in zip(client_states, weights):
        for k, v in sd.items():
            if not v.is_floating_point():
                # copy first seen non-floating buffer/value
                if k not in agg:
                    agg[k] = v.clone()
                continue
            if k not in agg:
                agg[k] = torch.zeros_like(v, dtype=torch.float32)
            agg[k] += v.to(torch.float32) * (w / total)

    # cast aggregated floats back to original dtype of first client's tensor
    first = client_states[0]
    for k in list(agg.keys()):
        if agg[k].is_floating_point() and k in first:
            agg[k] = agg[k].to(first[k].dtype)

    return agg

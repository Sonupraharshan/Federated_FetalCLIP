# federated/server.py
import torch
import numpy as np
from copy import deepcopy
from sklearn.cluster import KMeans
from federated.fedavg import fedavg
from federated.client import local_train_feature, _unpack_state_dicts
from metrics_logger import log_metric
from models.heads import get_head
from models.model_utils import save_checkpoint
from sklearn.metrics import accuracy_score, f1_score


def compute_client_summary(encoder, loader, device):
    """Compute client summary from encoder or cached features"""
    if encoder is None:
        # Using cached features: compute mean of embeddings directly
        feats_list = []
        with torch.no_grad():
            for xb, _ in loader:  # xb is already embeddings when using cached features
                xb = xb.to(device)
                feats_list.append(xb.mean(dim=0).cpu().numpy())
        if not feats_list:
            return None
        return np.mean(feats_list, axis=0)
    else:
        # Using raw images: extract features with encoder
        encoder.eval()
        feats_list = []
        total_batches = len(loader) if hasattr(loader, '__len__') else None
        with torch.no_grad():
            for idx, (xb, _) in enumerate(loader):
                try:
                    xb = xb.to(device)
                    f = encoder(xb)
                except Exception as e:
                    print(f"   ⚠️  Error while running encoder for client summary: {e}")
                    raise

                if f.dim() > 2:
                    f = torch.nn.functional.adaptive_avg_pool2d(f, (1, 1))
                    f = f.view(f.size(0), -1)
                feats_list.append(f.mean(dim=0).cpu().numpy())

                # progress print every 50 batches (or on last batch)
                if total_batches is not None:
                    if (idx + 1) % 50 == 0 or (idx + 1) == total_batches:
                        print(f"      Extracted features from {idx+1}/{total_batches} batches for this client")
                else:
                    if (idx + 1) % 50 == 0:
                        print(f"      Extracted features from {idx+1} batches for this client")

        if not feats_list:
            return None
        return np.mean(feats_list, axis=0)


def run_federated(encoder, head_name, client_loaders, test_loader, config, split_name='iid', encoder_out_dim=None):
    device = config.DEVICE
    head_name = head_name.lower()
    
    # Determine encoder output dimension
    if encoder_out_dim is None:
        encoder_out_dim = encoder.out_dim if encoder is not None else 768

    finetune = getattr(config, "FINETUNE_ENCODER", True)
    freeze_rounds = getattr(config, "FREEZE_ENCODER_ROUNDS", 2)
    use_dp = getattr(config, "USE_DP", False)
    use_cfl = getattr(config, "USE_CFL", False)
    use_qfl = getattr(config, "USE_QFL", False)
    use_cached = getattr(config, "USE_CACHED_FEATURES", False)

    q_qubits = getattr(config, "QFL_QUBITS", 4)
    q_out_dim = getattr(config, "QFL_OUTPUT_DIM", 8)
    
    # When using cached features, disable encoder finetuning
    if use_cached:
        finetune = False
        print("   ⚠️  Using cached features → encoder finetuning disabled")

    dp_cfg = {
        "clip_norm": 3.0,
        "noise_multiplier": 0.08,
        "accum_steps": 2,         # simulate twice the batch size
        "warmup_epochs": 1,       # 1 epoch warmup without DP
        "per_layer_scaling": True
    }


    try:
        num_classes = len(test_loader.dataset.df['label_id'].unique())
    except Exception:
        num_classes = 6

    prototype_head = get_head(head_name, input_dim=encoder_out_dim, num_classes=num_classes,
                              use_qfl=use_qfl, q_qubits=q_qubits, q_out_dim=q_out_dim).to(device)

    # Initialize global states
    if use_cfl:
        # compute summaries and cluster clients
        summaries = []
        for idx, loader in enumerate(client_loaders):
            try:
                ds_len = len(loader.dataset) if hasattr(loader, 'dataset') else 'unknown'
            except Exception:
                ds_len = 'unknown'
            print(f"   🔄 Computing summary for client {idx+1}/{len(client_loaders)} (samples={ds_len})...")
            s = compute_client_summary(encoder, loader, device)
            print(f"   ✅ Summary for client {idx+1} computed. shape={None if s is None else s.shape}")
            summaries.append(s)
        summaries = np.stack(summaries)
        k = getattr(config, "CFL_NUM_CLUSTERS", 2)
        kmeans = KMeans(n_clusters=k, random_state=0).fit(summaries)
        cluster_ids = kmeans.labels_.tolist()
        print("📌 Cluster assignments:", cluster_ids)

        # init cluster states
        cluster_states = []
        for cid in range(k):
            head = deepcopy(prototype_head)
            if finetune:
                enc_sd = deepcopy(encoder.state_dict())
                head_sd = deepcopy(head.state_dict())
                combined = {**{f"encoder.{kk}": vv for kk, vv in enc_sd.items()},
                            **{f"head.{kk}": vv for kk, vv in head_sd.items()}}
                cluster_states.append(combined)
            else:
                cluster_states.append(deepcopy(head.state_dict()))
    else:
        # single global
        if finetune:
            head = deepcopy(prototype_head)
            enc_sd = deepcopy(encoder.state_dict())
            head_sd = deepcopy(head.state_dict())
            global_state = {**{f"encoder.{k}": v for k, v in enc_sd.items()},
                            **{f"head.{k}": v for k, v in head_sd.items()}}
        else:
            global_state = deepcopy(prototype_head.state_dict())

    # Federated rounds
    for rnd in range(1, config.ROUNDS + 1):
        print(f"\n=== Federated Round {rnd}/{config.ROUNDS} | Head={head_name} | Split={split_name} ===")
        finetune_round = finetune and rnd > freeze_rounds
        if finetune and rnd <= freeze_rounds:
            print("🧊 Encoder frozen (warmup)")
        elif finetune_round:
            print("🔥 Encoder unfrozen (finetuning)")

        if use_cfl:
            k = getattr(config, "CFL_NUM_CLUSTERS", 2)
            new_cluster_states = [None] * k
            for cid in range(k):
                print(f"\n--- Training cluster {cid} ---")
                client_sds = []
                client_ws = []
                for i, loader in enumerate(client_loaders):
                    if cluster_ids[i] != cid:
                        continue
                    sd = local_train_feature(
                        encoder, cluster_states[cid], loader,
                        config.LOCAL_EPOCHS, config.LR, config.DEVICE,
                        head_name, num_classes,
                        encoder_out_dim=encoder_out_dim,
                        finetune=finetune_round,
                        encoder_lr=getattr(config, "ENCODER_LR", None),
                        head_lr=getattr(config, "HEAD_LR", None),
                        weight_decay=getattr(config, "WEIGHT_DECAY", 1e-4),
                        class_weights=getattr(config, "CLASS_WEIGHTS", None),
                        use_dp=use_dp, dp_config=dp_cfg,
                        use_qfl=use_qfl, q_qubits=q_qubits, q_out_dim=q_out_dim
                    )
                    client_sds.append(sd)
                    client_ws.append(len(loader.dataset))
                    print(f"Client {i} finished local training ({len(loader.dataset)} samples)")

                if not client_sds:
                    new_cluster_states[cid] = cluster_states[cid]
                else:
                    cluster_agg = fedavg(client_sds, client_ws)
                    new_cluster_states[cid] = cluster_agg

            cluster_states = new_cluster_states

            # Evaluate each cluster model and average metrics
            accs = []
            f1s = []
            for cid in range(k):
                if finetune:
                    enc_sd, head_sd = _unpack_state_dicts(cluster_states[cid])
                    encoder.load_state_dict(enc_sd, strict=False)
                    head = get_head(head_name, input_dim=encoder_out_dim, num_classes=num_classes,
                                    use_qfl=use_qfl, q_qubits=q_qubits, q_out_dim=q_out_dim).to(device)
                    head.load_state_dict(head_sd, strict=False)
                    eval_model = torch.nn.Sequential(encoder, head).to(device)
                else:
                    head = get_head(head_name, input_dim=encoder_out_dim, num_classes=num_classes,
                                    use_qfl=use_qfl, q_qubits=q_qubits, q_out_dim=q_out_dim).to(device)
                    head.load_state_dict(cluster_states[cid], strict=False)
                    eval_model = torch.nn.Sequential(encoder, head) if encoder is not None else head

                preds, trues = [], []
                eval_model.eval()
                with torch.no_grad():
                    for xb, yb in test_loader:
                        xb, yb = xb.to(device), yb.to(device)
                        logits = eval_model(xb)
                        preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
                        trues.extend(yb.cpu().tolist())
                acc = accuracy_score(trues, preds)
                f1 = f1_score(trues, preds, average='macro')
                print(f"Cluster {cid} Eval | Acc={acc:.4f} | F1={f1:.4f}")
                accs.append(acc)
                f1s.append(f1)

            mean_acc = float(np.mean(accs))
            mean_f1 = float(np.mean(f1s))
            print(f"\n🎯 Round {rnd} (CFL mean) | Acc={mean_acc:.4f} | F1={mean_f1:.4f}")
            log_metric(head_name, split_name, rnd, mean_acc, mean_f1)

        else:
            # normal federated (single global)
            client_states = []
            weights = []
            for i, loader in enumerate(client_loaders):
                print(f"\n   📱 Client {i+1}/{len(client_loaders)} training...")
                sd = local_train_feature(
                    encoder, global_state, loader,
                    config.LOCAL_EPOCHS, config.LR, config.DEVICE,
                    head_name, num_classes,
                    encoder_out_dim=encoder_out_dim,
                    finetune=finetune_round,
                    encoder_lr=getattr(config, "ENCODER_LR", None),
                    head_lr=getattr(config, "HEAD_LR", None),
                    weight_decay=getattr(config, "WEIGHT_DECAY", 1e-4),
                    class_weights=getattr(config, "CLASS_WEIGHTS", None),
                    use_dp=use_dp, dp_config=dp_cfg,
                    use_qfl=use_qfl, q_qubits=q_qubits, q_out_dim=q_out_dim
                )
                client_states.append(sd)
                weights.append(len(loader.dataset))
                print(f"   ✅ Client {i+1} done ({len(loader.dataset)} samples)")

            print(f"\n   🔄 Aggregating models (FedAvg)...")
            global_state = fedavg(client_states, weights)
            print(f"   ✅ Aggregation complete")

            # evaluate global
            print(f"   📊 Evaluating on test set...")
            if finetune:
                enc_sd, head_sd = _unpack_state_dicts(global_state)
                encoder.load_state_dict(enc_sd, strict=False)
                head = get_head(head_name, input_dim=encoder_out_dim, num_classes=num_classes,
                                use_qfl=use_qfl, q_qubits=q_qubits, q_out_dim=q_out_dim).to(device)
                head.load_state_dict(head_sd, strict=False)
                eval_model = torch.nn.Sequential(encoder, head).to(device)
            else:
                head = get_head(head_name, input_dim=encoder_out_dim, num_classes=num_classes,
                                use_qfl=use_qfl, q_qubits=q_qubits, q_out_dim=q_out_dim).to(device)
                head.load_state_dict(global_state, strict=False)
                # When using cached features, encoder is None, so use only the head
                eval_model = head if encoder is None else torch.nn.Sequential(encoder, head).to(device)

            preds, trues = [], []
            eval_model.eval()
            with torch.no_grad():
                for xb, yb in test_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    logits = eval_model(xb)
                    preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
                    trues.extend(yb.cpu().tolist())
            acc = accuracy_score(trues, preds)
            f1 = f1_score(trues, preds, average='macro')
            print(f"\n🎯 Round {rnd}/{config.ROUNDS} | Acc={acc:.4f} | F1={f1:.4f}")
            log_metric(head_name, split_name, rnd, float(acc), float(f1))

    # Save final models
    if use_cfl:
        k = getattr(config, "CFL_NUM_CLUSTERS", 2)
        for cid in range(k):
            save_checkpoint(cluster_states[cid], f"{head_name}_{split_name}_cluster{cid}")
        print("💾 Saved cluster models.")
        return cluster_states
    else:
        save_checkpoint(global_state, f"{head_name}_{split_name}")
        print("💾 Saved single global model.")
        return global_state

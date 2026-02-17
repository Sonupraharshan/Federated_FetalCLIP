# federated/client.py
import torch
import torch.nn as nn
from copy import deepcopy
from models.heads import get_head
from typing import Optional
import config
from torch.cuda.amp import autocast, GradScaler

# --------------------- pack/unpack helpers ---------------------
def _pack_state_dicts(encoder_state, head_state):
    combined = {}
    for k, v in encoder_state.items():
        combined[f"encoder.{k}"] = v
    for k, v in head_state.items():
        combined[f"head.{k}"] = v
    return combined


def _unpack_state_dicts(combined):
    enc = {}
    head = {}
    for k, v in combined.items():
        if k.startswith("encoder."):
            enc[k.replace("encoder.", "", 1)] = v
        elif k.startswith("head."):
            head[k.replace("head.", "", 1)] = v
    return enc, head


# --------------------- BatchNorm -> GroupNorm helper (kept for safety if needed) ---------------------
def replace_batchnorm_with_groupnorm(module: nn.Module, gn_groups: int = 8):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm1d):
            ch = child.num_features
            setattr(module, name, nn.GroupNorm(num_groups=min(gn_groups, ch), num_channels=ch))
        elif isinstance(child, nn.BatchNorm2d):
            ch = child.num_features
            setattr(module, name, nn.GroupNorm(num_groups=min(gn_groups, ch), num_channels=ch))
        else:
            replace_batchnorm_with_groupnorm(child, gn_groups)


# --------------------- Combined wrapper model ---------------------
class FederatedModel(nn.Module):
    def __init__(self, encoder: nn.Module, head: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, x):
        feats = self.encoder(x)
        if feats.dim() > 2:
            feats = torch.nn.functional.adaptive_avg_pool2d(feats, (1, 1))
            feats = feats.view(feats.size(0), -1)
        return self.head(feats)


# --------------------- Utility: sanitize gradients and params ---------------------
def sanitize_tensor_(t: torch.Tensor, name="tensor"):
    if t is None:
        return
    if torch.any(torch.isnan(t)) or torch.any(torch.isinf(t)):
        # In-place replace NaN/Inf with finite numbers
        t.data = torch.nan_to_num(t.data, nan=0.0, posinf=1e3, neginf=-1e3)


# --------------------- utility: compute total grad-norm ---------------------
def _compute_grad_norm(params):
    total = 0.0
    for p in params:
        if p.grad is None:
            continue
        total += float(torch.norm(p.grad.detach(), p=2).item() ** 2)
    return total ** 0.5


# --------------------- local training (manual DP, QFL preserved) ---------------------
def local_train_feature(
    encoder,
    global_state,
    dataloader,
    local_epochs,
    lr,
    device,
    head_name,
    num_classes,
    encoder_out_dim: Optional[int] = None,         # <-- ADD THIS
    finetune: Optional[bool] = None,                # <-- default None; use config if not provided
    encoder_lr: Optional[float] = None,
    head_lr: Optional[float] = None,
    weight_decay: float = 1e-4,
    class_weights=None,
    use_dp: Optional[bool] = None,                  # <-- default None => use config
    dp_config: Optional[dict] = None,
    use_qfl: Optional[bool] = None,
    q_qubits: int = 4,
    q_out_dim: int = 8
):
    """
    Manual-DP local trainer.
    - finetune: if None, uses config.FINETUNE_ENCODER
    - use_dp: if None, uses config.USE_DP
    - use_qfl: if None, uses config.USE_QFL
    All other DP defaults are taken from dp_config or config (so everything follows config.py).
    """

    # ----------------------------
    # Resolve defaults from config
    # ----------------------------
    if finetune is None:
        finetune = bool(getattr(config, "FINETUNE_ENCODER", True))
    if use_dp is None:
        use_dp = bool(getattr(config, "USE_DP", False))
    if use_qfl is None:
        use_qfl = bool(getattr(config, "USE_QFL", False))

    # DP defaults: prefer dp_config values, else read from config.py
    dp_cfg = dp_config or {}
    clip_norm = float(dp_cfg.get("clip_norm", getattr(config, "DP_CLIP_NORM", 3.0)))
    noise_mult = float(dp_cfg.get("noise_multiplier", getattr(config, "DP_NOISE_MULTIPLIER", 0.08)))
    accum_steps = int(dp_cfg.get("accum_steps", 1))
    warmup_epochs = int(dp_cfg.get("warmup_epochs", 0))
    per_layer_scaling = bool(dp_cfg.get("per_layer_scaling", False))

    # Determine encoder output dimension
    if encoder_out_dim is None:
        encoder_out_dim = encoder.out_dim if encoder is not None else 768
    
    # Build head (QFL handled inside get_head)
    head = get_head(
        head_name,
        input_dim=encoder_out_dim,
        num_classes=num_classes,
        use_qfl=use_qfl,
        q_qubits=q_qubits,
        q_out_dim=q_out_dim
    )

    # Load states (do not mutate finetune here)
    if finetune:
        enc_sd, head_sd = _unpack_state_dicts(global_state)
        encoder.load_state_dict(enc_sd, strict=False)
        head.load_state_dict(head_sd, strict=False)
    else:
        head.load_state_dict(global_state, strict=False)

    # Build model & optimizer
    if finetune:
        model = FederatedModel(encoder, head)
        model.to(device)
        enc_lr = encoder_lr if encoder_lr is not None else lr * 0.2
        hd_lr = head_lr if head_lr is not None else lr
        optimizer = torch.optim.AdamW([
            {'params': model.encoder.parameters(), 'lr': enc_lr},
            {'params': model.head.parameters(), 'lr': hd_lr}
        ], weight_decay=weight_decay)
    else:
        model = head.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Loss
    criterion = nn.CrossEntropyLoss(
        weight=(torch.tensor(class_weights, device=device) if class_weights is not None else None)
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, local_epochs))
    
    # Initialize AMP GradScaler
    scaler = GradScaler(enabled=(device == "cuda"))

    # Print config summary (accurate)
    print(f"\n📌 Local training start | finetune={finetune} | use_dp={use_dp} | qfl={use_qfl}")
    if use_dp:
        print(f"    DP config: clip_norm={clip_norm}, noise_mult={noise_mult}, accum_steps={accum_steps}, warmup_epochs={warmup_epochs}")

    # Training loop with accumulation and robust manual DP
    for epoch in range(1, local_epochs + 1):
        model.train()
        total_loss = 0.0
        batch_count = 0
        optimizer.zero_grad()

        dp_active = use_dp and (epoch > warmup_epochs)
        if use_dp and not dp_active and warmup_epochs > 0:
            print(f"   Epoch {epoch}: DP warmup (DP disabled this epoch)")

        for i, (xb, yb) in enumerate(dataloader):
            xb, yb = xb.to(device), yb.to(device)
            batch_count += 1

            # Extract features (encoder used for features; gradients flow if finetune True)
            # When using cached features, encoder is None and xb already contains embeddings
            if encoder is None:
                feats = xb
            elif finetune:
                feats = model.encoder(xb)
            else:
                with torch.no_grad():
                    feats = encoder(xb)

            if feats.dim() > 2:
                feats = torch.nn.functional.adaptive_avg_pool2d(feats, (1, 1))
                feats = feats.view(feats.size(0), -1)

            # Forward through head with AMP
            with autocast(enabled=(device == "cuda")):
                if finetune:
                    logits = model.head(feats)
                else:
                    logits = model(feats) if (isinstance(model, nn.Module) and model is not head) else head(feats)

                loss = criterion(logits, yb) / accum_steps

            # Scale loss and backward
            scaler.scale(loss).backward()

            # accumulate
            if (i + 1) % accum_steps == 0:
                # Unscale gradients before clipping/DP
                scaler.unscale_(optimizer)

                # Manual DP step (fixed clip_norm)
                if use_dp and dp_active:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)

                    # sanitize grads before noise
                    for p in model.parameters():
                        if p.grad is not None:
                            sanitize_tensor_(p.grad, name=f"grad:{p.shape}")

                    # add Gaussian noise scaled to clip_norm
                    for p in model.parameters():
                        if p.grad is not None:
                            noise = noise_mult * clip_norm * torch.randn_like(p.grad)
                            p.grad.add_(noise)

                    # re-sanitize
                    for p in model.parameters():
                        if p.grad is not None:
                            sanitize_tensor_(p.grad, name=f"grad_after_noise:{p.shape}")
                
                # Step with scaler
                scaler.step(optimizer)
                scaler.update()

                # sanitize parameters to prevent NaN/Inf propagation
                for p in model.parameters():
                    if p is not None:
                        sanitize_tensor_(p.data, name=f"param:{p.shape}")

                optimizer.zero_grad()
            
            # Progress indicator every 10 batches
            if (i + 1) % 10 == 0:
                avg_loss = total_loss / (i + 1)
                print(f"      Epoch {epoch}/{local_epochs} | Batch {i+1}/{len(dataloader)} | Loss: {avg_loss:.4f}")

            total_loss += (loss.item() * accum_steps)

        # leftover batches
        if (len(dataloader) % accum_steps) != 0:
            scaler.unscale_(optimizer)
            if use_dp and dp_active:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                for p in model.parameters():
                    if p.grad is not None:
                        sanitize_tensor_(p.grad, name=f"grad:{p.shape}")
                        p.grad.add_(noise_mult * clip_norm * torch.randn_like(p.grad))
                        sanitize_tensor_(p.grad, name=f"grad_after_noise:{p.shape}")
            scaler.step(optimizer)
            scaler.update()
            for p in model.parameters():
                sanitize_tensor_(p.data, name=f"param:{p.shape}")
            optimizer.zero_grad()

        scheduler.step()
        avg_loss = total_loss / max(1, batch_count)
        print(f"✅ Local Epoch {epoch} done | Avg loss = {avg_loss:.4f}")

    # Return packed states
    if finetune:
        enc_state = deepcopy(model.encoder.state_dict())
        head_state = deepcopy(model.head.state_dict())
        return _pack_state_dicts(enc_state, head_state)
    else:
        return deepcopy(model.state_dict())

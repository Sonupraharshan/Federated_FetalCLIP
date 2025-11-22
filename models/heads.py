# models/heads.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional quantum layer import (if installed)
try:
    from models.quantum_layer import QuantumLayer
except Exception:
    QuantumLayer = None

# Optional Opacus DP-compatible MultiheadAttention
try:
    from opacus.layers import DPMultiheadAttention
    OPACUS_AVAILABLE = True
except ImportError:
    DPMultiheadAttention = None
    OPACUS_AVAILABLE = False


class MLPHead(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dims=(1024, 512), dropout=0.3):
        super().__init__()
        layers = []
        dims = [input_dim] + list(hidden_dims)
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU(inplace=False))
            layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ResidualMLPHead(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=512, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x1 = F.relu(self.bn1(self.fc1(x)), inplace=False)
        x2 = F.relu(self.bn2(self.fc2(self.dropout(x1))), inplace=False)
        x = x1 + x2
        x = self.fc_out(self.dropout(x))
        return x


class AttentionHead(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=256, heads=4, dropout=0.2, use_dp_compatible=False):
        super().__init__()
        # This head accepts (B, D) or (B, N, D)
        self.proj = nn.Linear(input_dim, hidden_dim)
        
        # Use DPMultiheadAttention if available and requested, otherwise standard
        if use_dp_compatible and OPACUS_AVAILABLE:
            self.attn = DPMultiheadAttention(embed_dim=hidden_dim, num_heads=heads, batch_first=True)
            print("✅ Using DPMultiheadAttention for Opacus compatibility")
        else:
            self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=heads, batch_first=True)
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=False),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # x: (B, D) or (B, N, D)
        if x.dim() == 2:
            # vector input - need to project first
            x_proj = self.proj(x)  # (B, D) -> (B, hidden_dim)
            x_proj = self.norm(x_proj)
            return self.classifier(x_proj)
        elif x.dim() == 3:
            # sequence of tokens
            x_proj = self.proj(x)  # (B, N, H)
            z, _ = self.attn(x_proj, x_proj, x_proj)  # (B, N, H)
            pooled = z.mean(dim=1)  # (B, H)
            pooled = self.norm(pooled)
            return self.classifier(pooled)
        else:
            raise ValueError("AttentionHead expects 2D or 3D input")


class LinearProbe(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


class HeadWrapper(nn.Module):
    """
    Wraps a base head and optionally applies a small quantum layer fusion.
    use_qfl True => requires QuantumLayer implemented in models/quantum_layer.py
    """
    def __init__(self, base_head: nn.Module, use_qfl: bool = False, q_qubits: int = 4, q_out_dim: int = 8):
        super().__init__()
        self.base_head = base_head
        self.use_qfl = use_qfl and (QuantumLayer is not None)
        self.quantum_failed = False  # Track if quantum layer failed to avoid repeated messages
        
        if self.use_qfl:
            # Get input dimension from the base head
            if hasattr(base_head, 'proj'):  # AttentionHead
                input_dim = base_head.proj.in_features
            elif hasattr(base_head, 'fc'):  # LinearProbe
                input_dim = base_head.fc.in_features
            elif hasattr(base_head, 'fc1'):  # ResidualMLPHead
                input_dim = base_head.fc1.in_features
            elif hasattr(base_head, 'net') and hasattr(base_head.net[0], 'in_features'):  # MLPHead
                input_dim = base_head.net[0].in_features
            else:
                input_dim = 512  # fallback
                
            # Get output dimension (num_classes)
            if hasattr(base_head, 'classifier') and hasattr(base_head.classifier[-1], 'out_features'):  # AttentionHead
                output_dim = base_head.classifier[-1].out_features
            elif hasattr(base_head, 'fc') and hasattr(base_head.fc, 'out_features'):  # LinearProbe
                output_dim = base_head.fc.out_features
            elif hasattr(base_head, 'fc_out') and hasattr(base_head.fc_out, 'out_features'):  # ResidualMLPHead
                output_dim = base_head.fc_out.out_features
            elif hasattr(base_head, 'net') and hasattr(base_head.net[-1], 'out_features'):  # MLPHead
                output_dim = base_head.net[-1].out_features
            else:
                output_dim = 6  # fallback to NUM_CLASSES
            
            try:
                # Fix: Use proper parameter order for QuantumLayer
                self.q_layer = QuantumLayer(n_qubits=q_qubits, out_dim=q_out_dim)
                # small fusion MLP: concat base head output with q_out
                fusion_input_dim = output_dim + q_out_dim
                self.fusion = nn.Sequential(
                    nn.Linear(fusion_input_dim, 256),
                    nn.ReLU(inplace=False),
                    nn.LayerNorm(256),
                    nn.Dropout(0.2),
                    nn.Linear(256, output_dim)
                )
                print(f"✅ Quantum layer initialized: {input_dim}→{q_out_dim}, fusion: {fusion_input_dim}→{output_dim}")
            except Exception as e:
                print(f"❌ Failed to initialize quantum components: {e}")
                self.use_qfl = False
                self.q_layer = None
                self.quantum_failed = True
        else:
            self.q_layer = None

    def forward(self, x):
        base_out = self.base_head(x)  # (B, num_classes)
        if not self.use_qfl or self.q_layer is None or self.quantum_failed:
            return base_out
        
        try:
            # Call quantum layer with proper input
            q_out = self.q_layer(x)  # Should return (B, q_out_dim)
            
            # Handle different quantum layer output formats
            if isinstance(q_out, (list, tuple)):
                # Handle nested lists/tuples
                if isinstance(q_out[0], (list, tuple)):
                    # Flatten nested structure
                    flat_out = []
                    for batch_item in q_out:
                        if isinstance(batch_item, (list, tuple)):
                            flat_out.extend(batch_item)
                        else:
                            flat_out.append(batch_item)
                    q_out = torch.tensor(flat_out, device=x.device, dtype=x.dtype).view(x.size(0), -1)
                else:
                    q_out = torch.stack([torch.tensor(item, device=x.device, dtype=x.dtype) if not isinstance(item, torch.Tensor) 
                                        else item for item in q_out])
                    if q_out.dim() == 1:
                        q_out = q_out.unsqueeze(0).expand(x.size(0), -1)
            elif not isinstance(q_out, torch.Tensor):
                # Single value output
                q_out = torch.full((x.size(0), self.q_layer.n_qubits if hasattr(self.q_layer, 'n_qubits') else 8), 
                                  float(q_out), device=x.device, dtype=x.dtype)
            
            # Ensure proper batch size
            if q_out.size(0) != x.size(0):
                if q_out.size(0) == 1:
                    q_out = q_out.expand(x.size(0), -1)
                else:
                    # Reshape/repeat to match batch size
                    q_out = q_out.view(-1)[:x.size(0) * q_out.size(-1)].view(x.size(0), -1)
            
            # Concatenate and fuse
            fused = torch.cat([base_out, q_out], dim=1)
            return self.fusion(fused)
                
        except Exception as e:
            if not self.quantum_failed:  # Only print once
                print(f"❌ Quantum layer failed: {e}. Disabling for this session.")
                self.quantum_failed = True
            return base_out


def get_head(name: str, input_dim: int = 512, num_classes: int = 6,
             use_qfl: bool = False, q_qubits: int = 4, q_out_dim: int = 8,
             use_dp_compatible: bool = False):
    name = name.lower()
    if name == "mlp":
        base = MLPHead(input_dim, num_classes)
    elif name == "residual_mlp":
        base = ResidualMLPHead(input_dim, num_classes)
    elif name == "attention":
        base = AttentionHead(input_dim, num_classes, use_dp_compatible=use_dp_compatible)
    elif name == "linear":
        base = LinearProbe(input_dim, num_classes)
    else:
        raise ValueError(f"Unknown head type: {name}")
    return HeadWrapper(base, use_qfl=use_qfl, q_qubits=q_qubits, q_out_dim=q_out_dim)

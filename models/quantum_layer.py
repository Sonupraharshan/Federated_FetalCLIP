# models/quantum_layer.py
import torch
import torch.nn as nn
import pennylane as qml


class QuantumLayer(nn.Module):
    """
    A safe, batch-compatible quantum layer:
    - Takes (B, D)
    - Projects to n_qubits inputs
    - Runs a quantum circuit per sample
    - Returns (B, out_dim)
    """

    def __init__(self, n_qubits=4, out_dim=8):
        super().__init__()
        self.n_qubits = n_qubits
        self.out_dim = out_dim

        self.dev = qml.device("default.qubit", wires=n_qubits)

        # quantum weights
        self.weights = nn.Parameter(0.01 * torch.randn(n_qubits, n_qubits, dtype=torch.float32))

        # classical projection (set after seeing input shape)
        self.input_proj = None

        # classical post layer
        self.post = nn.Linear(n_qubits, out_dim)

    def _make_circuit(self):
        n = self.n_qubits

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            """inputs: (n_qubits,)"""
            # encode classical values
            for i in range(n):
                qml.RY(inputs[i], wires=i)

            # entangle + trainable weights
            for i in range(n):
                for j in range(n):
                    qml.RZ(weights[i, j], wires=i)

            return [qml.expval(qml.PauliZ(k)) for k in range(n)]

        return circuit

    def forward(self, x):
        """
        x: (B, D)
        """
        # Ensure input is float32
        x = x.float()
        B, D = x.shape

        # lazy-create projection layer once
        if self.input_proj is None:
            self.input_proj = nn.Linear(D, self.n_qubits, dtype=torch.float32).to(x.device)

        # project → [-π, π]
        proj = torch.tanh(self.input_proj(x)) * 3.14159265  # (B, n_qubits)

        circuit = self._make_circuit()

        outputs = []
        for b in range(B):
            raw = circuit(proj[b], self.weights)

            # SAFE CONVERSION: properly handle gradients and ensure float32
            if isinstance(raw, torch.Tensor):
                # Keep gradient tracking and ensure float32
                raw = raw.to(dtype=torch.float32, device=x.device)
            elif isinstance(raw, (list, tuple)):
                # Convert list/tuple to tensor while preserving gradients
                raw = torch.stack([
                    r.to(dtype=torch.float32, device=x.device) if isinstance(r, torch.Tensor) 
                    else torch.tensor(r, dtype=torch.float32, device=x.device) 
                    for r in raw
                ])
            else:
                # Single value - ensure float32
                raw = torch.tensor(float(raw), dtype=torch.float32, device=x.device, requires_grad=True)

            outputs.append(raw)

        outs = torch.stack(outputs)  # (B, n_qubits)

        return self.post(outs)

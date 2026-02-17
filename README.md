# 🏥 Federated FetalCLIP - Distributed Learning for Fetal Ultrasound Classification

## 📋 Project Overview

**Federated FetalCLIP** is a sophisticated federated learning framework designed for **distributed training of deep learning models on fetal ultrasound images**. This project combines:

- **FetalCLIP Encoder**: A CLIP-based encoder specialized for fetal ultrasound classification
- **Federated Learning (FedAvg)**: Distributed model training across multiple clients without centralizing data
- **Clustered Federated Learning (CFL)**: Grouping similar clients for improved convergence
- **Quantum Federated Learning (QFL)**: Integration of quantum neural networks for advanced feature extraction
- **Differential Privacy (DP)**: Privacy-preserving training using gradient clipping and noise injection
- **Multiple Head Architectures**: MLPHead, ResidualMLPHead, AttentionHead, Linear, ResNet18, EfficientNet-B0, DenseNet121, ViT-Small

---

## 📁 Project Structure

```
Federated_FetalCLIP/
│
├── 📄 main.py                    # Entry point: orchestrates entire federated training pipeline
├── 📄 config.py                  # Global configuration (hyperparameters, paths, flags)
├── 📄 metrics_logger.py          # Logging accuracy/F1 metrics to CSV
├── 📄 requirements.txt           # Python package dependencies
│
├── 📂 data/                      # Data handling and loading
│   ├── __init__.py
│   ├── dataset_loader.py         # Dataset class & DataLoader creation
│   ├── utils.py                  # Data utility functions
│   └── images/                   # Fetal ultrasound images (grayscale, 1 channel)
│
├── 📂 models/                    # Model architectures
│   ├── __init__.py
│   ├── fetalclip_encoder.py      # FetalCLIP encoder backbone
│   ├── heads.py                  # Classification head architectures
│   ├── model_utils.py            # Checkpoint saving/loading utilities
│   └── quantum_layer.py          # Quantum neural network layer (for QFL)
│
├── 📂 federated/                 # Federated learning implementation
│   ├── __init__.py
│   ├── server.py                 # Central server orchestrating rounds
│   ├── client.py                 # Local client training logic
│   └── fedavg.py                 # FedAvg aggregation algorithm
│
├── 📂 results/                   # Output files
│   ├── metrics.csv               # Logged metrics (head, split, round, acc, f1)
│   └── checkpoints/              # Saved model weights
│
└── 📂 rounds/                    # Per-round analysis
    └── accuracy&F1Score.csv      # Round-wise performance summary
```

---

## 🔄 Code Flow & Execution Sequence

### **Phase 1: Initialization (main.py)**

```python
# Step 1: Parse command-line arguments
python main.py --clients 4 --rounds 20 --head attention --split iid [--use_dp] [--use_cfl] [--use_qfl]
```

**Arguments explained:**

- `--clients`: Number of federated clients (default: 4)
- `--rounds`: Number of federated training rounds (default: 20)
- `--local_epochs`: Local training epochs per client per round (default: 10)
- `--batch_size`: Batch size for training (default: 128)
- `--split`: 'iid' or 'noniid' - data distribution across clients
- `--head`: Classification head type (mlp, residual_mlp, attention, linear, resnet18, efficientnet_b0, densenet121, vit_small)
- `--finetune`: Enable encoder fine-tuning during federated training
- `--use_dp`: Enable Differential Privacy (Opacus-based)
- `--use_cfl`: Enable Clustered Federated Learning (groups similar clients)
- `--use_qfl`: Enable Quantum Federated Learning (quantum layer in head)

**Step 2: Create necessary directories**

```python
ensure_dirs()
# Creates: ./results/checkpoints/, ./results/
```

**Step 3: Initialize metrics logging**

```python
init_metrics()
# Creates ./metrics.csv with headers: [head, split, round, acc, f1]
```

---

### **Phase 2: Data Loading (data/dataset_loader.py → XLSXImageDataset)**

**Step 1: Load dataset from Excel**

```python
df = pd.read_excel("./data/labels.xlsx")
# Expected columns: ['Image_name', 'Plane'] (or similar label column)
```

**Step 2: Data Cleaning**

```python
df = df.dropna(subset=['Image_name', 'Plane'])
# Remove rows with missing image names or labels
```

**Step 3: Train-Test Split**

```python
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['Plane'], random_state=42)
# Stratified split ensures each class is represented in train/test
```

**Step 4: Create Label Mapping**

```python
unique_labels = sorted(train_df['Plane'].unique())
label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
# Example: {'Sagittal': 0, 'Transverse': 1, 'Coronal': 2, ...}
```

**Step 5: Data Augmentation (only on training data)**

```python
train_transforms = [
    Resize((224, 224)),              # Normalize image size
    RandomHorizontalFlip(),          # Horizontal flip augmentation
    RandomRotation(10),              # Small rotations
    RandomAffine(...),               # Affine transformations
    ColorJitter(...),                # Brightness/contrast variation
    ToTensor(),                      # Convert to torch.Tensor
    Normalize(...)                   # Normalize pixel values
]
```

**Step 6: Client Data Distribution**

```python
# IID (Independent Identical Distribution)
if iid:
    # Shuffle data randomly, split equally among clients
    shuffled_indices = random.shuffle(train_indices)
    client_data = [shuffled_indices[i::num_clients] for i in range(num_clients)]
else:
    # Non-IID: Clients get samples from different label distributions
    # (simulates real federated scenarios with heterogeneous data)
```

**Step 7: Create DataLoaders**

```python
client_loaders = [
    DataLoader(ClientDataset[i], batch_size=128, shuffle=True)
    for i in range(num_clients)
]
test_loader = DataLoader(TestDataset, batch_size=128, shuffle=False)
```

---

### **Phase 3: Model Initialization (main.py → FetalCLIPEncoder)**

**Step 1: Create Encoder**

```python
encoder = FetalCLIPEncoder(
    feature_dim=512,                           # Output embedding dimension
    pretrained_path=None,                      # Path to pretrained weights (optional)
    device=device                              # 'cuda' or 'cpu'
)
encoder.to(device)
```

**FetalCLIP Encoder Architecture:**

```
Input: (B, 1, H, W)                    # Batch of grayscale ultrasound images
    ↓
Backbone (Conv2d blocks):
    - Conv2d(1, 32, 3, stride=2)       # Reduce spatial dims
    - BatchNorm2d + ReLU
    - Conv2d(32, 64, 3, stride=2)
    - BatchNorm2d + ReLU
    - AdaptiveAvgPool2d((1, 1))        # Global average pooling
    ↓
Linear Projection:
    - Linear(64, 512)                  # Project to feature dimension
    ↓
Output: (B, 512)                       # Embedding for each image
```

**Step 2: Get Output Dimension**

```python
encoder_out_dim = encoder.out_dim  # 512
```

---

### **Phase 4: Federated Learning Server (federated/server.py → run_federated)**

#### **4.1: Initialize Global Model State**

```python
# If NOT using CFL (single global model):
if not use_cfl:
    global_state = {
        "encoder.layer1.weight": tensor,   # Encoder parameters
        "encoder.layer1.bias": tensor,
        ...
        "head.fc1.weight": tensor,         # Head parameters
        "head.fc1.bias": tensor,
        ...
    }

# If using CFL (separate model per cluster):
else:
    # Compute client summaries
    for each client:
        summary = compute_client_summary(encoder, client_loader, device)
        # Extract mean feature vector from encoder

    # Cluster clients using K-means
    kmeans = KMeans(n_clusters=2, random_state=0).fit(summaries)
    cluster_ids = kmeans.labels_  # [0, 1, 0, 1] for 4 clients, 2 clusters

    # Initialize separate state for each cluster
    cluster_states = [deepcopy(prototype_head) for _ in range(num_clusters)]
```

#### **4.2: Federated Training Rounds Loop**

```python
for rnd in range(1, config.ROUNDS + 1):
    # ===== CLIENT TRAINING =====
    print(f"Round {rnd}/{ROUNDS}")

    if use_cfl:
        # Train each cluster separately
        for cluster_id in range(num_clusters):
            client_sds = []      # State dicts from clients in this cluster
            client_ws = []       # Weights (data sizes) for aggregation

            for client_id in range(num_clients):
                if cluster_ids[client_id] != cluster_id:
                    continue

                # Send cluster state to client
                client_state = cluster_states[cluster_id]
```

#### **4.3: Local Client Training (federated/client.py → local_train_feature)**

**For each client:**

```python
def local_train_feature(encoder, global_state, client_loader, local_epochs, lr, device, head_name, ...):

    # Step 1: Unpack global state
    encoder_state, head_state = _unpack_state_dicts(global_state)

    # Step 2: Load global state into local encoder & head
    encoder.load_state_dict(encoder_state, strict=False)
    head = get_head(head_name, input_dim=encoder.out_dim, num_classes=num_classes)
    head.load_state_dict(head_state, strict=False)

    # Step 3: Create federated model (encoder + head)
    model = FederatedModel(encoder, head)
    model.to(device)

    # Step 4: Setup optimizer
    if finetune:
        # Encoder and head both trainable
        params = [
            {'params': encoder.parameters(), 'lr': encoder_lr},
            {'params': head.parameters(), 'lr': head_lr}
        ]
        optimizer = torch.optim.Adam(params, weight_decay=weight_decay)
    else:
        # Only head trainable (encoder frozen)
        optimizer = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)

    # Step 5: Optional - Setup Differential Privacy (if use_dp=True)
    if use_dp:
        # Wrap optimizer with privacy engine
        privacy_engine = PrivacyEngine(
            model=model,
            batch_size=batch_size,
            sample_size=len(client_dataset),
            noise_multiplier=dp_noise_multiplier,
            max_grad_norm=dp_clip_norm
        )
        privacy_engine.attach(optimizer)

    # Step 6: Optional - Replace BatchNorm with GroupNorm for DP compatibility
    if use_dp:
        replace_batchnorm_with_groupnorm(model)

    # Step 7: Local training loop
    criterion = CrossEntropyLoss(weight=class_weights)  # Optional weighted loss

    for epoch in range(local_epochs):
        model.train()
        for batch_idx, (images, labels) in enumerate(client_loader):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            logits = model(images)
            loss = criterion(logits, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Optional - Gradient clipping (DP)
            if use_dp and not use_opacus:
                grad_norm = _compute_grad_norm(model.parameters())
                max_norm = dp_clip_norm
                if grad_norm > max_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            # Optimization step (includes noise injection if using DP)
            optimizer.step()

            print(f"Epoch {epoch+1}/{local_epochs}, Batch {batch_idx+1}, Loss: {loss.item():.4f}")

    # Step 8: Extract and return updated state
    enc_state = encoder.state_dict()
    head_state = head.state_dict()
    client_sd = _pack_state_dicts(enc_state, head_state)

    return client_sd  # Send back to server
```

#### **4.4: Server Aggregation (federated/fedavg.py → fedavg)**

**After all clients in a round complete training:**

```python
def fedavg(client_sds, client_ws):
    """
    Performs Federated Averaging aggregation.

    Inputs:
    - client_sds: list of state_dicts from each client
    - client_ws: list of weights (dataset sizes) for each client

    Weighted average:
    global_param = sum(w_i * client_param_i) / sum(w_i)
    """

    # Normalize weights
    total_w = sum(client_ws)
    normalized_ws = [w / total_w for w in client_ws]

    # Initialize aggregated state
    aggregated_state = {}

    for param_name in client_sds[0].keys():
        # Weighted average across all clients
        aggregated_param = torch.zeros_like(client_sds[0][param_name])

        for client_idx, client_sd in enumerate(client_sds):
            aggregated_param += normalized_ws[client_idx] * client_sd[param_name]

        aggregated_state[param_name] = aggregated_param

    return aggregated_state  # Return to server
```

#### **4.5: Server Evaluation**

**After aggregation, evaluate on global test set:**

```python
# Load aggregated state into model
encoder.load_state_dict(enc_state, strict=False)
head.load_state_dict(head_state, strict=False)
model = torch.nn.Sequential(encoder, head)
model.eval()

# Inference on test set
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        logits = model(images)
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# Compute metrics
accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average='weighted')

# Log metrics
log_metric(head_name, split_name, rnd, accuracy, f1)
print(f"Round {rnd}: Accuracy={accuracy:.4f}, F1={f1:.4f}")
```

#### **4.6: Optional - Clustered Federated Learning (CFL)**

```python
if use_cfl:
    # Periodically (every CFL_REASSIGN_EVERY_ROUND rounds):
    # 1. Recompute client summaries using current encoder
    # 2. Re-cluster using K-means
    # 3. Redistribute cluster membership
    # This helps adapt cluster assignments as encoder evolves
```

#### **4.7: Optional - Differential Privacy (DP)**

```python
if use_dp:
    # Noise injection during local training
    # Each client's gradients are:
    # 1. Clipped to max norm (default: dp_clip_norm=1.0)
    # 2. Gaussian noise added (scale: noise_multiplier * clip_norm)

    # Privacy budget accounting (if using Opacus):
    eps, delta = privacy_engine.get_privacy_budget()
    print(f"Privacy budget: eps={eps:.2f}, delta={delta}")
    # Converged epsilon indicates privacy level:
    # - Lower epsilon = more private (higher noise)
    # - Higher epsilon = less private (lower noise)
```

#### **4.8: Optional - Quantum Federated Learning (QFL)**

```python
if use_qfl:
    # Quantum layer replaces final dense layer in head
    # Input: (B, head_input_dim)
    #   ↓
    # Classical-to-quantum encoding (angle encoding)
    #   ↓
    # Parameterized quantum circuit (q_qubits qubits)
    #   ↓
    # Measurement → classical output (q_out_dim dimensions)
    #   ↓
    # Output: (B, q_out_dim)

    # Quantum circuit is trained like a classical neural network layer
    # Gradients computed via parameter shift rule
```

---

### **Phase 5: Model Head Architectures (models/heads.py)**

**1. MLPHead**

```python
Input (B, 512)
    ↓
Linear(512, 1024) + ReLU + LayerNorm + Dropout
    ↓
Linear(1024, 512) + ReLU + LayerNorm + Dropout
    ↓
Linear(512, num_classes)
    ↓
Output (B, num_classes)
```

**2. ResidualMLPHead**

```python
Input (B, 512)
    ↓
FC1(512, 512) + BatchNorm + ReLU
    ↓
FC2(512, 512) + Dropout + BatchNorm + ReLU (with residual connection)
    ↓
FC_out(512, num_classes)
    ↓
Output (B, num_classes)
```

**3. AttentionHead**

```python
Input (B, 512)
    ↓
Linear projection to query/key/value
    ↓
Multi-head attention (8 heads)
    ↓
Feed-forward network (MLP)
    ↓
Linear(512, num_classes)
    ↓
Output (B, num_classes)
```

**4. Linear Head**

```python
Input (B, 512)
    ↓
Linear(512, num_classes)
    ↓
Output (B, num_classes)
```

**5. ResNet18/EfficientNet-B0/DenseNet121/ViT-Small Heads**

```python
Input (B, 512)
    ↓
Use backbone architecture with num_classes output
    ↓
Output (B, num_classes)
```

---

## 🚀 Installation & Setup

### **1. Clone Repository**

```bash
cd "C:/Users/sonup/Desktop/Final Year Project"
git clone <repo_url>
cd Federated_FetalCLIP
```

### **2. Create Virtual Environment**

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### **3. Install Dependencies**

```bash
pip install -r requirements.txt
```

**Key packages:**

- `torch>=1.12`: Deep learning framework
- `torchvision`: Image processing utilities
- `timm`: Transformer models (ViT, etc.)
- `scikit-learn`: Clustering & metrics
- `pandas`: Data handling
- `numpy`: Numerical computing
- `Pillow`: Image loading
- `tqdm`: Progress bars
- `openpyxl`: Excel file reading
- `opacus`: Differential Privacy
- `pennylane`: Quantum computing

### **4. Prepare Dataset**

```
data/
├── labels.xlsx          # Excel file with columns: ['Image_name', 'Plane']
└── images/
    ├── image1.png       # Grayscale ultrasound images (1 channel)
    ├── image2.png
    └── ...
```

---

## ▶️ Running the Project

### **Basic Run (IID split, MLP head)**

```bash
python main.py
```

### **Advanced Configuration**

```bash
python main.py \
    --clients 4 \
    --rounds 20 \
    --local_epochs 10 \
    --batch_size 128 \
    --split iid \
    --head attention \
    --finetune \
    --data_path "./data" \
    --device cuda
```

### **With Differential Privacy**

```bash
python main.py --head mlp --use_dp
```

### **With Clustered Federated Learning**

```bash
python main.py --head attention --use_cfl
```

### **With Quantum Federated Learning**

```bash
python main.py --head attention --use_qfl --q_qubits 4 --q_out_dim 8
```

### **Non-IID Data Distribution**

```bash
python main.py --split noniid --head residual_mlp
```

---

## 📊 Configuration (config.py)

```python
NUM_CLASSES = 6                                    # Number of output classes
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Federated Learning
NUM_CLIENTS = 4                                    # Number of clients
ROUNDS = 20                                        # Number of federated rounds
LOCAL_EPOCHS = 10                                  # Local training epochs per round
BATCH_SIZE = 128                                   # Batch size

# Model Architecture
FEATURE_DIM = 512                                  # Encoder output dimension
LR = 1e-4                                          # Learning rate (for head)
WEIGHT_DECAY = 1e-4                               # L2 regularization
ENCODER_LR = 5e-5                                  # Encoder-specific learning rate (if finetuning)
HEAD_LR = 1e-4                                     # Head-specific learning rate
FINETUNE_ENCODER = True                            # Fine-tune encoder weights
CLASS_WEIGHTS = None                               # Optional: [w0, w1, ...] for imbalanced classes

# Paths
DATA_PATH = "./data"                               # Dataset path
CHECKPOINT_DIR = "./results/checkpoints/"          # Model checkpoint directory

# Differential Privacy (DP)
USE_DP = False                                     # Enable DP training
DP_METHOD = "opacus"                               # "opacus" or "manual"
DP_CLIP_NORM = 1.0                                 # Gradient clipping threshold
DP_NOISE_MULTIPLIER = 0.5                          # Noise scale (higher = more private)
DP_DELTA = 1e-5                                    # Privacy budget parameter

# Clustered Federated Learning (CFL)
USE_CFL = False                                    # Enable CFL
CFL_NUM_CLUSTERS = 2                               # Number of clusters
CFL_REASSIGN_EVERY_ROUND = False                   # Re-cluster each round

# Quantum Federated Learning (QFL)
USE_QFL = False                                    # Enable QFL
QFL_QUBITS = 4                                     # Number of qubits
QFL_OUTPUT_DIM = 8                                 # Quantum layer output dimension
```

---

## 📈 Monitoring & Outputs

### **Metrics CSV (results/metrics.csv)**

```csv
head,split,round,acc,f1
attention,iid,1,0.7234,0.7189
attention,iid,2,0.7567,0.7521
attention,iid,3,0.7812,0.7789
...
```

**Fields:**

- `head`: Classification head type
- `split`: Data distribution (iid/noniid)
- `round`: Federated round number
- `acc`: Test set accuracy
- `f1`: Weighted F1-score

### **Checkpoints (results/checkpoints/)**

```
attention_iid_cluster0.pt     # Cluster 0 model (if using CFL)
attention_iid_cluster1.pt     # Cluster 1 model (if using CFL)
attention_iid.pt              # Final global model (if not using CFL)
```

### **Console Output Example**

```
Loading data from: ./data
Initial DataFrame size: 5000
Cleaned DataFrame size: 4980
Label map: {0: 'Sagittal', 1: 'Transverse', 2: 'Coronal', ...}

Encoder out_dim = 512

=== Federated Round 1/20 | Head=attention | Split=iid ===
🧊 Encoder frozen (warmup)
Client 0 finished local training (1245 samples)
Client 1 finished local training (1256 samples)
Client 2 finished local training (1243 samples)
Client 3 finished local training (1236 samples)
Round 1: Accuracy=0.7234, F1=0.7189

=== Federated Round 2/20 | Head=attention | Split=iid ===
🔥 Encoder unfrozen (finetuning)
[Client training continues...]
Round 2: Accuracy=0.7567, F1=0.7521

[... continues for 20 rounds ...]

Done.
```

---

## 🔑 Key Concepts Explained

### **Federated Learning**

- **Concept**: Train ML models on decentralized data without centralizing it
- **Process**:
  1. Server sends global model to clients
  2. Each client trains locally on their data
  3. Clients send updated models back to server
  4. Server aggregates using FedAvg
  5. Repeat for multiple rounds

### **FedAvg (Federated Averaging)**

- Weighted average of client model updates
- Weights are proportional to client dataset sizes
- Ensures larger datasets have more influence on global model

### **Clustered Federated Learning (CFL)**

- Clients are grouped into clusters based on feature similarity
- Each cluster has separate global model
- Useful when clients have heterogeneous data distributions

### **Differential Privacy (DP)**

- Adds noise to gradients to protect individual data privacy
- Gradient clipping limits influence of any single sample
- Privacy-utility tradeoff: more noise = more private but lower accuracy

### **Quantum Federated Learning (QFL)**

- Uses quantum circuits as neural network layers
- Potential advantage: quantum speedup for certain tasks
- Currently simulated on classical hardware via PennyLane

### **Non-IID Data**

- "Non-Independent and Identically Distributed"
- Different clients have different label distributions
- More challenging and realistic federated scenario
- Simulates real-world where different hospitals have different patient populations

---

## 🛠️ Advanced Features

### **Encoder Fine-tuning**

```python
# First FREEZE_ENCODER_ROUNDS rounds: encoder frozen (only head training)
# After that: encoder unfrozen (both encoder and head trainable)
# This improves convergence by starting with stable encoder
```

### **Weighted Loss (Class Imbalance)**

```python
# If dataset has imbalanced classes:
class_weights = [1.0, 1.5, 2.0, ...]  # Higher weight for minority classes
criterion = CrossEntropyLoss(weight=torch.tensor(class_weights))
```

### **Opacus Differential Privacy**

```python
# Opacus-based DP provides formal privacy guarantees
# Uses per-sample gradient computation and sophisticated accounting
# More advanced than manual gradient clipping
```

---

## 📚 Important Classes & Functions

| File                          | Class/Function                   | Purpose                         |
| ----------------------------- | -------------------------------- | ------------------------------- |
| `data/dataset_loader.py`      | `XLSXImageDataset`               | Load images & labels from Excel |
| `data/dataset_loader.py`      | `make_dataloaders()`             | Create client DataLoaders       |
| `models/fetalclip_encoder.py` | `FetalCLIPEncoder`               | Fetal ultrasound encoder        |
| `models/heads.py`             | `MLPHead`, `AttentionHead`, etc. | Classification heads            |
| `federated/server.py`         | `run_federated()`                | Main federated training loop    |
| `federated/client.py`         | `local_train_feature()`          | Client-side local training      |
| `federated/fedavg.py`         | `fedavg()`                       | Federated averaging aggregation |
| `metrics_logger.py`           | `log_metric()`                   | Log accuracy/F1 to CSV          |

---

## 🎯 Example Workflow

```bash
# 1. Setup
python -m venv venv
source venv/bin/activate  # or: venv\Scripts\activate (Windows)
pip install -r requirements.txt

# 2. Prepare data in ./data/ directory (images/ + labels.xlsx)

# 3. Run federated training with attention head and differential privacy
python main.py \
    --clients 4 \
    --rounds 20 \
    --head attention \
    --split iid \
    --use_dp \
    --finetune

# 4. Check results
# - View metrics in ./results/metrics.csv
# - Check model checkpoints in ./results/checkpoints/
# - Analyze console output for training progress
```

---

## 📝 Expected Results

When training converges well, you should see:

- **Accuracy increasing** across federated rounds (e.g., 60% → 75% → 80%)
- **F1-score improving** in sync with accuracy
- **Smooth training curves** (not too noisy)
- **Final accuracy typically** in range: 70-85% depending on:
  - Dataset quality and size
  - Head architecture complexity
  - Number of federated rounds
  - Data distribution (IID vs Non-IID)
  - Presence of DP (slightly lower accuracy)

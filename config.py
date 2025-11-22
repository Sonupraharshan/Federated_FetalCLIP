import os

NUM_CLASSES = 6

DEVICE = "cuda" if __import__('torch').cuda.is_available() else "cpu"

NUM_CLIENTS = 4
ROUNDS = 20
LOCAL_EPOCHS = 10
BATCH_SIZE = 128
FEATURE_DIM = 512
LR = 1e-4
WEIGHT_DECAY = 1e-4
ENCODER_LR = 5e-5
HEAD_LR = 1e-4
PRETRAINED_ENCODER_PATH = None  # set path to fetalclip pretrained if available
FINETUNE_ENCODER = True          # set to True to fine-tune encoder
CLASS_WEIGHTS = None             # or a list / numpy array of weights
DATA_PATH = os.path.abspath("./data")  # or use absolute path like "D:/EC22b1011/Federated_FetalCLIP/data"
HEADS = [ "mlp", "resnet18", "efficientnet_b0", "densenet121", "vit_small"]
LOG_DIR = "./logs/"
CHECKPOINT_DIR = "./results/checkpoints/"

# DP (Differential Privacy)
USE_DP = False
DP_METHOD = "opacus"           # "opacus" or "manual"
DP_CLIP_NORM = 1.0
DP_NOISE_MULTIPLIER = 0.5
DP_TARGET_EPS = None           # optional, used with Opacus accounting
DP_DELTA = 1e-5


# CFL (Clustered Federated Learning)
USE_CFL = False
CFL_NUM_CLUSTERS = 2
CFL_REASSIGN_EVERY_ROUND = False  # recompute clusters each round if True

# QFL (Quantum Federated Learning)
USE_QFL = False
QFL_QUBITS = 4
QFL_OUTPUT_DIM = 8              # quantum layer output dim
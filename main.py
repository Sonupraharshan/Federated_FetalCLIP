# main.py
import argparse
import os
import config as cfg
from data.dataset_loader import make_dataloaders
from data.cached_loader import load_cached_features
from models.fetalclip_encoder import FetalCLIPEncoder
from federated.server import run_federated
from metrics_logger import init_metrics

def ensure_dirs():
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs("./results", exist_ok=True)
    open("./results/metrics.csv", "a").close()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clients', type=int, default=cfg.NUM_CLIENTS)
    parser.add_argument('--rounds', type=int, default=cfg.ROUNDS)
    parser.add_argument('--local_epochs', type=int, default=cfg.LOCAL_EPOCHS)
    parser.add_argument('--batch_size', type=int, default=cfg.BATCH_SIZE)
    parser.add_argument('--split', type=str, default='iid', choices=['iid','noniid'])
    parser.add_argument('--data_path', type=str, default=cfg.DATA_PATH)
    parser.add_argument('--device', type=str, default=cfg.DEVICE)
    parser.add_argument('--head', type=str, default='attention', choices=['mlp','residual_mlp','attention','linear'])
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--use_dp', action='store_true')
    parser.add_argument('--use_cfl', action='store_true')
    parser.add_argument('--use_qfl', action='store_true')
    parser.add_argument('--q_qubits', type=int, default=getattr(cfg,'QFL_QUBITS',4))
    parser.add_argument('--q_out_dim', type=int, default=getattr(cfg,'QFL_OUTPUT_DIM',8))
    parser.add_argument('--use_cached_features', action='store_true', help='Use pre-computed FetalCLIP embeddings')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    cfg.NUM_CLIENTS = args.clients
    cfg.ROUNDS = args.rounds
    cfg.LOCAL_EPOCHS = args.local_epochs
    cfg.BATCH_SIZE = args.batch_size
    cfg.DATA_PATH = args.data_path
    cfg.DEVICE = args.device
    cfg.FINETUNE_ENCODER = bool(args.finetune)
    cfg.USE_DP = bool(args.use_dp)
    cfg.USE_CFL = bool(args.use_cfl)
    cfg.USE_QFL = bool(args.use_qfl)
    cfg.QFL_QUBITS = int(args.q_qubits)
    cfg.QFL_OUTPUT_DIM = int(args.q_out_dim)
    cfg.USE_CACHED_FEATURES = bool(args.use_cached_features)

    # Print effective configuration for visibility
    print("\n-- Effective Config --")
    print(f" FINETUNE_ENCODER={cfg.FINETUNE_ENCODER} | USE_DP={cfg.USE_DP} | USE_CFL={cfg.USE_CFL} | USE_QFL={cfg.USE_QFL} | USE_CACHED_FEATURES={cfg.USE_CACHED_FEATURES}")
    print(f" NUM_CLIENTS={cfg.NUM_CLIENTS} | ROUNDS={cfg.ROUNDS} | LOCAL_EPOCHS={cfg.LOCAL_EPOCHS} | BATCH_SIZE={cfg.BATCH_SIZE}")

    ensure_dirs()
    init_metrics()
    
    # Load data (either from cache or raw images)
    if cfg.USE_CACHED_FEATURES:
        print("\n[*] Loading pre-computed FetalCLIP embeddings...")
        client_loaders, test_loader, label_map = load_cached_features(
            num_clients=cfg.NUM_CLIENTS,
            batch_size=cfg.BATCH_SIZE
        )
        encoder = None  # Not needed when using cached features
        encoder_out_dim = 768  # FetalCLIP output dimension
        print("[OK] Cached features loaded!")
    else:
        print("\n[DATA] Loading raw images from disk...")
        client_loaders, test_loader, label_map = make_dataloaders(
            cfg.DATA_PATH,
            num_clients=cfg.NUM_CLIENTS,
            iid=(args.split=='iid'),
            batch_size=cfg.BATCH_SIZE
        )
        print("Label map:", label_map)

        print("\n📦 Loading FetalCLIP encoder...")
        encoder = FetalCLIPEncoder(
            feature_dim=getattr(cfg,'FEATURE_DIM',512),
            pretrained_path=getattr(cfg,'FETALCLIP_WEIGHTS_PATH',None),
            config_path=getattr(cfg,'FETALCLIP_CONFIG_PATH',None),
            device=cfg.DEVICE
        )
        encoder.to(cfg.DEVICE)
        encoder_out_dim = encoder.out_dim
        print(f"✅ Encoder loaded (out_dim={encoder_out_dim})")

    final_state = run_federated(encoder, args.head, client_loaders, test_loader, cfg, split_name=args.split, encoder_out_dim=encoder_out_dim)
    print("Done.")

# main.py
import argparse
import os
import config as cfg
from data.dataset_loader import make_dataloaders
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

    ensure_dirs()
    init_metrics()

    client_loaders, test_loader, label_map = make_dataloaders(cfg.DATA_PATH, num_clients=cfg.NUM_CLIENTS,
                                                              iid=(args.split=='iid'),
                                                              batch_size=cfg.BATCH_SIZE)
    print("Label map:", label_map)

    encoder = FetalCLIPEncoder(feature_dim=getattr(cfg,'FEATURE_DIM',512),
                               pretrained_path=getattr(cfg,'PRETRAINED_ENCODER_PATH',None),
                               device=cfg.DEVICE)
    encoder.to(cfg.DEVICE)
    print(f"Encoder out_dim = {encoder.out_dim}")

    final_state = run_federated(encoder, args.head, client_loaders, test_loader, cfg, split_name=args.split)
    print("Done.")

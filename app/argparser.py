import argparse
import os
from typing import Optional, Tuple


def get_args():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--interaction", default="ml-1m.csv", type=str)
    parser.add_argument("--item-meta", default=None, type=Optional[str])
    parser.add_argument("--user-key", default="user_id", type=str)
    parser.add_argument("--item-key", default="item_id", type=str)
    parser.add_argument("--time-key", default="timestamp", type=str)

    # preprocess
    parser.add_argument("--min-session-interval", default=60*60, type=int, help="1hour=60*60sec")
    parser.add_argument("--min-item-pop", default=5, type=int)
    parser.add_argument("--min-session-size", default=3, type=int)
    parser.add_argument("--min-num-sessions", default=5, type=int)
    parser.add_argument("--session-per-user", default=(5, 99), type=Tuple[int])
    parser.add_argument("--leave-out-session", default=1, type=int)

    # model
    parser.add_argument("--model-name", default="HierTCN.pt", type=str)
    parser.add_argument("--embedding-dim", default=256, type=int, help="dimension of item embedding layer")
    parser.add_argument("--hidden-dim", default=256, type=int, help="dimension of user hidden state in GRU")
    parser.add_argument("--dropout-rate", default=0.1, type=float, help="dropout rate in GRU")
    parser.add_argument("--kernel-size", default=3, type=int, help="kernel size in TCN")
    parser.add_argument("--dilation-rate", default=2, type=int, help="dilation rate in TCN")
    parser.add_argument("--sigma", default=-1, type=float,
                        help="init weight, -1: range [-sigma, sigma], -2: range [0, sigma]")
    parser.add_argument("--seed", default=777, type=int, help="seed for random initialization")

    # learning parameters
    parser.add_argument("--num_samples", default=-1, type=int)
    parser.add_argument("--loss_type", default="HingeLoss", type=str,
                        help="L2Loss, NCE, BPR, HingeLoss")
    parser.add_argument("--optimizer_type", default="Adam", type=str,
                        help="Adagrad, Adam, RMSProp, SGD")
    parser.add_argument("--n_epochs", default=10, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--momentum", default=0.0, type=float)
    parser.add_argument("--eps", default=1e-6, type=float)
    parser.add_argument("--patience", default=3, type=int, help="early stopping patience")
    parser.add_argument("--delta", default=0.0, type=float, help="early stopping threshold")
    parser.add_argument("--verbose", default=True, type=bool)

    # inference
    parser.add_argument("--eval_k", default=20, type=int, help="how many items you recommend")
    parser.add_argument("--user_id", type=int, help="user id")

    args = parser.parse_args()
    return args


def set_env(data_root="data", save_root="trained", checkpoint_root="checkpoint"):
    args = get_args()

    # raw data
    os.environ["interaction_dir"] = os.path.join(data_root, args.interaction)
    if args.item_meta:
        os.environ["item_meta_dir"] = os.path.join(data_root, args.item_meta)

    # data
    os.environ["inter_dir"] = os.path.join(data_root, "inter_data.csv")
    os.environ["train_dir"] = os.path.join(data_root, "train_data.csv")
    os.environ["valid_dir"] = os.path.join(data_root, "valid_data.csv")
    os.environ["test_dir"] = os.path.join(data_root, "test_data.csv")
    os.environ["item_dir"] = os.path.join(data_root, "item_data.csv")

    # pre-trained
    os.environ["save_dir"] = os.path.join(save_root, args.model_name)
    os.environ["checkpoint_dir"] = os.path.join(checkpoint_root, args.model_name)

    return args

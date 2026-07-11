"""Open-loop evaluation of a trained steering model (Fase 2).

Runs the model over a dataset split and reports steering MAE, the variance ratio
std(pred)/std(target) (collapse detector for "always straight"), and MAE broken
down by regime (straight vs curve) — the global mean hides what matters.

Usage:
    python src/ai/eval_openloop.py --ckpt runs/cam_v1.pt --data D:/tcc_data/dataset_v1 --split val
"""
import os
import sys

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from ai.dataset import SteeringDataset
from ai.dataset_index import build_index, list_episodes, split_episodes
from ai.metrics import mae, variance_ratio
from ai.model import CameraSteeringNet


def evaluate(ckpt, data_dir, split="val", val_frac=0.2, seed=0, batch=128, workers=4, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    episodes = list_episodes(data_dir)
    train_eps, val_eps = split_episodes(episodes, val_frac, seed)
    chosen = {"val": val_eps, "train": train_eps}.get(split, episodes)
    index = build_index(chosen)
    loader = DataLoader(SteeringDataset(index), batch_size=batch, shuffle=False, num_workers=workers)

    model = CameraSteeringNet().to(device)
    model(torch.zeros(1, 3, 66, 200, device=device))
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    preds, tgts = [], []
    with torch.no_grad():
        for x, y in loader:
            preds.append(model(x.to(device)).cpu().numpy().ravel())
            tgts.append(y.numpy().ravel())
    p, t = np.concatenate(preds), np.concatenate(tgts)

    straight = np.abs(t) < 0.05
    print("split=%s  frames=%d" % (split, len(t)))
    print("  MAE_steer   = %.4f   (gate: < 0.07)" % mae(p, t))
    print("  var_ratio   = %.2f   (gate: >= 0.60)" % variance_ratio(p, t))
    print("  MAE straight= %.4f   MAE curve= %.4f"
          % (mae(p[straight], t[straight]) if straight.any() else 0.0,
             mae(p[~straight], t[~straight]) if (~straight).any() else 0.0))
    return mae(p, t), variance_ratio(p, t)


def main():
    p = argparse.ArgumentParser(description="Open-loop evaluation of a steering model")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--split", default="val", choices=["val", "train", "all"])
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default=None)
    a = p.parse_args()
    evaluate(a.ckpt, a.data, a.split, a.val_frac, a.seed, a.batch, a.workers, a.device)


if __name__ == "__main__":
    main()

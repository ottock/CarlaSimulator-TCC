"""Train the camera-only steering model (Fase 2).

Splits by episode, upweights curves with a WeightedRandomSampler (the data is
~88% straight), trains a PilotNet CNN with SmoothL1 loss, and checkpoints the
lowest val steering MAE.

Usage (from the repo root):
    python src/ai/train.py --data D:/tcc_data/dataset_v1 --out runs/cam_v1.pt --epochs 40
"""
import os
import sys

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from ai.dataset import SteeringDataset
from ai.dataset_index import build_index, list_episodes, sample_weights, split_episodes
from ai.metrics import mae, variance_ratio
from ai.model import CameraSteeringNet


def _evaluate(model, loader, device):
    model.eval()
    preds, tgts = [], []
    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(device))
            preds.append(out.cpu().numpy().ravel())
            tgts.append(y.numpy().ravel())
    p, t = np.concatenate(preds), np.concatenate(tgts)
    return mae(p, t), variance_ratio(p, t)


def train(data_dir, out_path, epochs=40, batch=128, lr=1e-4, weight_decay=1e-5,
          dropout=0.3, val_frac=0.2, seed=0, workers=4, limit=0, patience=6, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    episodes = list_episodes(data_dir)
    if not episodes:
        raise SystemExit("No ep_* episodes found in %s" % data_dir)
    train_eps, val_eps = split_episodes(episodes, val_frac, seed)
    train_index = build_index(train_eps)
    val_index = build_index(val_eps)
    if limit:
        train_index = train_index[:limit]
        val_index = val_index[:max(1, limit // 5)]
    print("device=%s | episodes: %d train / %d val | frames: %d train / %d val"
          % (device, len(train_eps), len(val_eps), len(train_index), len(val_index)))

    weights = torch.as_tensor(sample_weights([r["steer"] for r in train_index]), dtype=torch.double)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    train_loader = DataLoader(SteeringDataset(train_index), batch_size=batch, sampler=sampler,
                              num_workers=workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(SteeringDataset(val_index), batch_size=batch, shuffle=False,
                            num_workers=workers, pin_memory=True)

    model = CameraSteeringNet(dropout=dropout).to(device)
    model(torch.zeros(1, 3, 66, 200, device=device))  # initialize LazyLinear
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.SmoothL1Loss()

    best, best_epoch, since_best = float("inf"), 0, 0
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        t0, running, nb = time.time(), 0.0, 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item()
            nb += 1
        sched.step()
        val_mae, var_ratio = _evaluate(model, val_loader, device)
        flag = ""
        if val_mae < best:
            best, best_epoch, since_best = val_mae, epoch, 0
            torch.save({"model_state_dict": model.state_dict(), "val_mae": val_mae,
                        "var_ratio": var_ratio, "epoch": epoch, "arch": "CameraSteeringNet"}, out_path)
            flag = " *"
        else:
            since_best += 1
        print("epoch %2d/%d  train_loss=%.4f  val_MAE=%.4f  var_ratio=%.2f  (%.1fs)%s"
              % (epoch, epochs, running / max(1, nb), val_mae, var_ratio, time.time() - t0, flag))
        if since_best >= patience:
            print("early stop (no val improvement for %d epochs)" % patience)
            break

    print("best val_MAE=%.4f at epoch %d  ->  %s" % (best, best_epoch, out_path))


def main():
    p = argparse.ArgumentParser(description="Train camera-only steering model")
    p.add_argument("--data", required=True)
    p.add_argument("--out", default="runs/cam_v1.pt")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=0, help="cap train frames (quick smoke runs)")
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--device", default=None)
    a = p.parse_args()
    train(a.data, a.out, a.epochs, a.batch, a.lr, a.weight_decay, a.dropout,
          a.val_frac, a.seed, a.workers, a.limit, a.patience, a.device)


if __name__ == "__main__":
    main()

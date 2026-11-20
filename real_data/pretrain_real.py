"""Pretrain Attn-LOB on REAL LOBSTER data (AAPL) mid-price direction.

Same task as pretrain.py (paper Sec. III-B): classify the future
price trend (up / stationary / down) from the past 50 LOB snapshots.
Labels are the .npz labels (mid direction 10 steps ahead, +/- $0.02
deadband), remapped -1/0/+1 -> 2/1/0 to match the FI-2010 convention
(0 = up, 1 = stationary, 2 = down).

Split: train steps [0, 24000), validation [24000, 30029) (last 20% of
the train slice), test = the held-out afternoon [30029, 40039).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from encoder import AttnLOB, count_parameters
from lob_data import LOBWindows
from pretrain import WindowDataset, evaluate

torch.set_num_threads(1)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "AAPL_10.npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--train-stride", type=int, default=5)
    ap.add_argument("--max-train", type=int, default=12000)
    ap.add_argument("--max-val", type=int, default=2000)
    ap.add_argument("--max-test", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/attn_lob_real_pretrained.pt")
    ap.add_argument("--metrics", default="results/pretrain_real_h10.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cpu")

    print("loading AAPL ...", flush=True)
    d = dict(np.load(DATA))
    feats, labels = d["feats"], d["labels"]
    y = (1 - labels).astype(np.int64)  # +1 -> 0 (up), 0 -> 1, -1 -> 2 (down)
    n = len(labels)
    n_train = int(d["train_mask"].sum())
    tr_cut = 24000
    print(f"steps: train [0,{tr_cut}) val [{tr_cut},{n_train}) "
          f"test [{n_train},{n})", flush=True)
    for name, sl in (("train", slice(0, tr_cut)),
                     ("val", slice(tr_cut, n_train)),
                     ("test", slice(n_train, n))):
        shares = [round(float(np.mean(y[sl] == c)), 3) for c in (0, 1, 2)]
        print(f"  {name} label shares (up/flat/down): {shares}", flush=True)
    train_w = LOBWindows(feats[:, :tr_cut], y[:tr_cut], seq_size=50,
                         stride=args.train_stride, max_windows=args.max_train)
    val_w = LOBWindows(feats[:, tr_cut:n_train], y[tr_cut:n_train],
                       seq_size=50, stride=10, max_windows=args.max_val)
    test_w = LOBWindows(feats[:, n_train:], y[n_train:], seq_size=50,
                        stride=10, max_windows=args.max_test)
    print(f"windows: train {len(train_w)}, val {len(val_w)}, "
          f"test {len(test_w)}", flush=True)
    train_dl = DataLoader(WindowDataset(train_w), batch_size=args.batch,
                          shuffle=True, num_workers=0)
    val_dl = DataLoader(WindowDataset(val_w), batch_size=args.batch,
                        num_workers=0)
    test_dl = DataLoader(WindowDataset(test_w), batch_size=args.batch,
                         num_workers=0)

    model = AttnLOB()
    print(f"Attn-LOB parameters: {count_parameters(model)}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    best_val = -1.0
    for ep in range(args.epochs):
        model.train()
        losses = []
        for x, yb in train_dl:
            opt.zero_grad()
            loss = loss_fn(model(x), yb)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        val = evaluate(model, val_dl, device)
        print(f"epoch {ep}: train_loss={np.mean(losses):.4f} "
              f"val_macro_f1={val['macro_f1']:.4f} "
              f"val_acc={val['accuracy']:.4f}", flush=True)
        if val["macro_f1"] > best_val:
            best_val = val["macro_f1"]
            torch.save(model.state_dict(), os.path.join(HERE, args.out))
    test = evaluate(model, test_dl, device)
    print(f"TEST macro_f1={test['macro_f1']:.4f} acc={test['accuracy']:.4f} "
          f"per_class={[round(v, 3) for v in test['per_class_f1']]}",
          flush=True)
    metrics = {
        "model": "attn-lob", "data": "LOBSTER AAPL 2012-06-21",
        "horizon_steps": 10, "seq_size": 50, "n_features": 40,
        "epochs": args.epochs, "batch": args.batch, "lr": args.lr,
        "seed": args.seed, "n_params": count_parameters(model),
        "train_windows": len(train_w), "val_windows": len(val_w),
        "test_windows": len(test_w),
        "best_val_macro_f1": best_val, "test": test,
    }
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, args.metrics), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"saved {args.out} + {args.metrics}", flush=True)


if __name__ == "__main__":
    main()

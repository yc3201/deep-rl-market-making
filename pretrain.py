"""Pretrain Attn-LOB on FI-2010 mid-price direction prediction.

The paper's pretraining task (Sec. III-B): classify the future price
trend (up / stationary / down) from the past T=50 LOB snapshots. The
paper trains on Ping An Bank (Shenzhen) with horizon k=10 and
threshold alpha=1e-5 and reports Attn-LOB F1 0.7284 vs DeepLOB 0.7118
(Table I). Here the same task is run on FI-2010 with the file's
standard labels at horizon 10 — the same labels the TLOB project uses
— so the number is directly comparable to the TLOB project's h=10
results (3-epoch: DeepLOB macro F1 0.312, TLOB 0.276). It is NOT
directly comparable to the paper's 0.7284 (different market, different
labelling); the comparison that matters is against the baselines on
the same data.

Split (paper / repo convention): train file first 80% -> train, last
20% -> validation; test = the three test files (days 8-10).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset

from encoder import AttnLOB, count_parameters
from lob_data import LOBWindows, class_shares, load_lob, standard_split

torch.set_num_threads(1)


class WindowDataset(Dataset):
    def __init__(self, windows: LOBWindows):
        self.w = windows

    def __len__(self):
        return len(self.w)

    def __getitem__(self, i):
        x, y = self.w[i]
        return torch.from_numpy(x), y


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    for x, y in loader:
        x = x.to(device)
        preds.append(model(x).argmax(dim=1).cpu().numpy())
        trues.append(y.numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    macro_f1 = float(f1_score(trues, preds, average="macro", zero_division=0))
    per_class = [float(v) for v in
                 f1_score(trues, preds, average=None, zero_division=0)]
    return {
        "macro_f1": macro_f1,
        "accuracy": float(accuracy_score(trues, preds)),
        "per_class_f1": per_class,
        "n": len(trues),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--max-train", type=int, default=20000)
    ap.add_argument("--max-val", type=int, default=5000)
    ap.add_argument("--max-test", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/attn_lob_pretrained.pt")
    ap.add_argument("--metrics", default="results/pretrain_h10.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cpu")
    here = os.path.dirname(os.path.abspath(__file__))

    print("loading FI-2010 ...", flush=True)
    d = load_lob(horizon=args.horizon)
    tr_f, tr_y = d["train_feats"], d["train_labels"]
    te_f, te_y = d["test_feats"], d["test_labels"]
    print(f"train shares: {[round(class_shares(tr_y)[c], 3) for c in (0, 1, 2)]}",
          flush=True)
    tr_slice, va_slice = standard_split(tr_f.shape[1])
    train_w = LOBWindows(tr_f[:, tr_slice], tr_y[tr_slice], seq_size=50,
                         stride=10, max_windows=args.max_train)
    val_w = LOBWindows(tr_f[:, va_slice], tr_y[va_slice], seq_size=50,
                       stride=10, max_windows=args.max_val)
    test_w = LOBWindows(te_f, te_y, seq_size=50, stride=10,
                        max_windows=args.max_test)
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
    opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    best_val = -1.0
    for ep in range(args.epochs):
        model.train()
        losses = []
        for x, y in train_dl:
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        val = evaluate(model, val_dl, device)
        print(f"epoch {ep}: train_loss={np.mean(losses):.4f} "
              f"val_macro_f1={val['macro_f1']:.4f} "
              f"val_acc={val['accuracy']:.4f}", flush=True)
        if val["macro_f1"] > best_val:
            best_val = val["macro_f1"]
            torch.save(model.state_dict(), os.path.join(here, args.out))
    test = evaluate(model, test_dl, device)
    print(f"TEST macro_f1={test['macro_f1']:.4f} acc={test['accuracy']:.4f} "
          f"per_class={[round(v, 3) for v in test['per_class_f1']]}",
          flush=True)
    metrics = {
        "model": "attn-lob", "horizon": args.horizon, "seq_size": 50,
        "n_features": 40, "epochs": args.epochs, "batch": args.batch,
        "lr": args.lr, "seed": args.seed,
        "n_params": count_parameters(model),
        "train_windows": len(train_w), "val_windows": len(val_w),
        "test_windows": len(test_w),
        "best_val_macro_f1": best_val, "test": test,
    }
    with open(os.path.join(here, args.metrics), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"saved {args.out} + {args.metrics}", flush=True)


if __name__ == "__main__":
    main()

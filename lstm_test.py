"""
lstm_test.py - does an LSTM predict direction better than XGBoost?

An LSTM is a neural network that reads a SEQUENCE (here the last 30 candles of the 12
features) instead of one row at a time. We keep the evaluation identical to the other models:
same chronological split, same target, same AUC metric, same rows for both models.

Needs PyTorch once:   pip install torch
Run:                  python lstm_test.py          (a few minutes on CPU)
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
try:
    import torch
    import torch.nn as nn
except ImportError:
    sys.exit("PyTorch is not installed. Run:  pip install torch   then run this again.")
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from model_utils import FEATURE_COLUMNS, load_split

SEQ, HIDDEN, EPOCHS, PATIENCE, BATCH = 30, 32, 12, 3, 256
INTERVAL = "5m"
torch.manual_seed(42)
np.random.seed(42)


def sequences(X, y, seq):
    idx = np.arange(seq - 1, len(X))
    Xs = np.stack([X[i - seq + 1:i + 1] for i in idx]).astype("float32")
    return Xs, y[idx].astype("float32")


class Net(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, HIDDEN, batch_first=True)
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(HIDDEN, 1)

    def forward(self, x):
        h, _ = self.lstm(x)
        return self.out(self.drop(h[:, -1, :])).squeeze(-1)


def predict(model, X):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(X))).numpy()


def block_bootstrap_auc(y, p, n=300, block=100, seed=42):
    rng = np.random.default_rng(seed)
    N = len(y)
    starts = np.arange(0, N - block + 1)
    aucs = []
    for _ in range(n):
        idx = np.concatenate([np.arange(s, s + block) for s in rng.choice(starts, N // block)])
        if len(np.unique(y[idx])) == 2:
            aucs.append(roc_auc_score(y[idx], p[idx]))
    return np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)


rows = []
for sym in ["BTCUSDT", "XRPUSDT", "SOLUSDT"]:
    print(f"\n=== {sym} {INTERVAL} ===")
    tr, va, te = (load_split(sym, INTERVAL, s) for s in ("train", "val", "test"))
    mu, sd = tr[FEATURE_COLUMNS].mean(), tr[FEATURE_COLUMNS].std().replace(0, 1)
    f = lambda d: ((d[FEATURE_COLUMNS] - mu) / sd).clip(-5, 5).to_numpy()
    Xtr, ytr = sequences(f(tr), tr["target"].to_numpy(), SEQ)
    Xva, yva = sequences(f(va), va["target"].to_numpy(), SEQ)
    Xte, yte = sequences(f(te), te["target"].to_numpy(), SEQ)

    net = Net(len(FEATURE_COLUMNS))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    best_auc, best_state, bad = 0.0, None, 0
    Xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    for ep in range(1, EPOCHS + 1):
        net.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad()
            loss_fn(net(Xt[b]), yt[b]).backward()
            opt.step()
        auc = roc_auc_score(yva, predict(net, Xva))
        print(f"  epoch {ep:2d}  validation AUC {auc:.4f}")
        if auc > best_auc:
            best_auc, bad = auc, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    net.load_state_dict(best_state)          # best epoch chosen on VALIDATION only

    # XGBoost on exactly the same rows, for a fair comparison
    xgb = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8,
                        colsample_bytree=0.8, min_child_weight=50, eval_metric="logloss",
                        n_jobs=-1, random_state=42).fit(tr[FEATURE_COLUMNS], tr["target"])
    for split, X, y, df in (("val", Xva, yva, va), ("test", Xte, yte, te)):
        p_l = predict(net, X)
        p_x = xgb.predict_proba(df[FEATURE_COLUMNS].iloc[SEQ - 1:])[:, 1]
        lo, hi = block_bootstrap_auc(y, p_l)
        rows.append(dict(symbol=sym, interval=INTERVAL, split=split,
                         lstm_auc=roc_auc_score(y, p_l), lstm_low=lo, lstm_high=hi,
                         xgb_auc=roc_auc_score(y, p_x)))
        print(f"  {split}: LSTM {rows[-1]['lstm_auc']:.4f} ({lo:.3f}-{hi:.3f})   XGBoost {rows[-1]['xgb_auc']:.4f}")

out = pd.DataFrame(rows)
os.makedirs("results", exist_ok=True)
out.to_csv("results/lstm_comparison.csv", index=False)
print("\n", out.round(4).to_string(index=False))
print("\nMean AUC by split:")
print(out.groupby("split")[["lstm_auc", "xgb_auc"]].mean().round(4).to_string())
print("\nSaved results/lstm_comparison.csv  (LSTM is 'better' only if it wins clearly AND its range stays above XGBoost's)")

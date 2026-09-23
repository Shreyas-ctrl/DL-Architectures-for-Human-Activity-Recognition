"""Shared training + evaluation. Trains MLP and/or CNN (and LSTM on request) on the
same subject-wise split, saves metrics JSON + confusion-matrix PNGs + model weights.

Usage:
  python src/train.py --models mlp cnn --epochs 40 --batch 64
  python src/train.py --models lstm --epochs 30
"""
import os, json, argparse, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from preprocessing import load_uci_har
from models import MLP, CNN1D, LSTMClassifier, count_params

ACTIVITIES = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def get_loaders(data, batch=64):
    def mk(X, y, shuffle):
        return DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
                          batch_size=batch, shuffle=shuffle)
    return (mk(data["X_train"], data["y_train"], True),
            mk(data["X_val"], data["y_val"], False),
            mk(data["X_test"], data["y_test"], False))


def train_one(model, train_loader, val_loader, epochs=40, lr=1e-3, patience=7, device="cpu"):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
    loss_fn = nn.CrossEntropyLoss()
    best_val, best_state, wait, history = float("inf"), None, 0, {"train_loss": [], "val_loss": []}
    for ep in range(1, epochs + 1):
        model.train()
        tl = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            tl += loss.item() * len(xb)
        tl /= len(train_loader.dataset)
        model.eval()
        vl = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                vl += loss_fn(model(xb), yb).item() * len(xb)
        vl /= len(val_loader.dataset)
        sched.step(vl)
        history["train_loss"].append(tl); history["val_loss"].append(vl)
        print(f"  epoch {ep:02d}/{epochs} train_loss={tl:.4f} val_loss={vl:.4f}")
        if vl < best_val - 1e-4:
            best_val, best_state, wait = vl, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  early stop at epoch {ep}")
                break
    model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def evaluate(model, loader, device="cpu"):
    model.eval().to(device)
    preds, trues = [], []
    for xb, yb in loader:
        preds += model(xb.to(device)).argmax(1).cpu().tolist()
        trues += yb.tolist()
    acc = accuracy_score(trues, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(trues, preds, average="macro", zero_division=0)
    cm = confusion_matrix(trues, preds, labels=list(range(6)))
    return {"accuracy": acc, "precision_macro": prec, "recall_macro": rec, "f1_macro": f1,
            "confusion_matrix": cm.tolist(), "y_true": trues, "y_pred": preds}


def plot_cm(cm, title, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(6), ACTIVITIES, rotation=30, ha="right", fontsize=7)
    ax.set_yticks(range(6), ACTIVITIES, fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, cm[i][j], ha="center", va="center", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["mlp", "cnn"], choices=["mlp", "cnn", "lstm"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--data_dir", default="data/UCI HAR Dataset")
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()

    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    data = load_uci_har(args.data_dir)
    train_loader, val_loader, test_loader = get_loaders(data, args.batch)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    builders = {"mlp": MLP, "cnn": CNN1D, "lstm": LSTMClassifier}
    summary = {}
    for name in args.models:
        print(f"\n=== Training {name.upper()} ({builders[name].__name__}) ===")
        model = builders[name]()
        print(f"Params: {count_params(model):,}")
        model, hist = train_one(model, train_loader, val_loader, epochs=args.epochs, device=device)
        res = evaluate(model, test_loader, device)
        print(f"{name.upper()} TEST — acc={res['accuracy']:.4f} prec={res['precision_macro']:.4f} "
              f"rec={res['recall_macro']:.4f} f1={res['f1_macro']:.4f}")
        torch.save(model.state_dict(), os.path.join("models", f"{name}.pt"))
        plot_cm(np.array(res["confusion_matrix"]),
                f"{name.upper()} confusion matrix (UCI HAR test)",
                os.path.join(args.out_dir, f"cm_{name}.png"))
        summary[name] = {k: v for k, v in res.items() if k in
                         ("accuracy", "precision_macro", "recall_macro", "f1_macro", "confusion_matrix")}
        summary[name]["params"] = count_params(builders[name]())
        summary[name]["history"] = hist

    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved metrics ->", os.path.join(args.out_dir, "metrics.json"))


if __name__ == "__main__":
    main()

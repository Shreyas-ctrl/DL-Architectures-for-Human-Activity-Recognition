"""
LSTM-specific training entry point for the group HAR project.

Run from the repository root:
    python src/lstm_train.py --epochs 30

This uses the shared preprocessing pipeline and LSTMClassifier from src/models.py
so the LSTM remains comparable with the MLP/CNN experiments.
"""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models import LSTMClassifier, count_params
from preprocessing import load_uci_har


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(X, y, batch_size, shuffle):
    x_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    return DataLoader(
        TensorDataset(x_tensor, y_tensor),
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=torch.cuda.is_available(),
    )


def evaluate_loss(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * y.size(0)
            total += y.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()

    return total_loss / total, correct / total


def main():
    args = parse_args()
    set_seed(args.seed)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Shared project preprocessing: official train/test split, validation
    # created from training subjects only, and training-fitted normalization.
    data = load_uci_har("data/UCI HAR Dataset")
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    assert X_train.shape[1:] == (128, 9)
    assert X_val.shape[1:] == (128, 9)
    assert X_test.shape[1:] == (128, 9)

    train_loader = make_loader(X_train, y_train, args.batch_size, True)
    val_loader = make_loader(X_val, y_val, args.batch_size, False)
    test_loader = make_loader(X_test, y_test, args.batch_size, False)

    model = LSTMClassifier().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    print(f"Parameters: {count_params(model):,}")
    print(
        f"Train: {len(X_train)} | Validation: {len(X_val)} | "
        f"Test: {len(X_test)}"
    )

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_total = 0
        train_correct = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * y.size(0)
            train_total += y.size(0)
            train_correct += (logits.argmax(dim=1) == y).sum().item()

        train_loss = train_loss_sum / train_total
        train_acc = train_correct / train_total
        val_loss, val_acc = evaluate_loss(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "learning_rate": current_lr,
            }
        )

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
            f"lr {current_lr:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    elapsed = time.time() - start

    if best_state is None:
        raise RuntimeError("No best model state was recorded.")

    model.load_state_dict(best_state)
    test_loss, test_acc = evaluate_loss(model, test_loader, criterion, device)

    model_path = MODEL_DIR / "lstm_aryan.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "architecture": {
                "input_size": 9,
                "hidden_size": 128,
                "num_layers": 2,
                "dropout": 0.3,
                "num_classes": 6,
            },
            "best_epoch": best_epoch,
            "seed": args.seed,
        },
        model_path,
    )

    history_path = RESULT_DIR / "lstm_training_history.json"
    history_path.write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )

    summary = {
        "model": "lstm",
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "parameters": count_params(model),
        "epochs_requested": args.epochs,
        "epochs_run": len(history),
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "patience": args.patience,
        "seed": args.seed,
        "device": str(device),
        "training_seconds": elapsed,
    }
    summary_path = RESULT_DIR / "lstm_train_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nBest epoch:", best_epoch)
    print(f"Final test accuracy: {test_acc:.4f}")
    print(f"Saved model: {model_path}")
    print(f"Saved history: {history_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()

"""
Final LSTM evaluation without scikit-learn.

Run from repository root:
    python src/lstm_evaluate.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from models import LSTMClassifier, count_params
from preprocessing import load_uci_har


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "lstm_aryan.pt"
RESULT_DIR = ROOT / "results"
PLOT_DIR = RESULT_DIR / "plots"

CLASS_NAMES = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
]


def calculate_metrics(y_true, y_pred, num_classes=6):
    """
    Calculate accuracy, macro precision, recall and F1
    using NumPy only.
    """

    # Confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=int)

    for true, pred in zip(y_true, y_pred):
        cm[true, pred] += 1

    # Accuracy
    accuracy = np.mean(y_true == y_pred)

    precisions = []
    recalls = []
    f1_scores = []

    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    macro_precision = np.mean(precisions)
    macro_recall = np.mean(recalls)
    macro_f1 = np.mean(f1_scores)

    return (
        accuracy,
        macro_precision,
        macro_recall,
        macro_f1,
        cm,
        precisions,
        recalls,
        f1_scores,
    )


def main():

    print("Starting LSTM evaluation...", flush=True)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{MODEL_PATH} not found. Run src/lstm_train.py first."
        )

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}", flush=True)

    # Load the same preprocessing used during training
    data = load_uci_har("data/UCI HAR Dataset")

    X_test = data["X_test"]
    y_test = data["y_test"]

    print(f"Test samples: {len(X_test)}", flush=True)

    assert X_test.shape[1:] == (128, 9)

    # IMPORTANT:
    # Use the same constructor that worked during training.
    model = LSTMClassifier().to(device)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False,
    )

    state = (
        checkpoint["model_state_dict"]
        if "model_state_dict" in checkpoint
        else checkpoint
    )

    model.load_state_dict(state)
    model.eval()

    print("Model loaded successfully.", flush=True)

    loader = DataLoader(
        TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.long),
        ),
        batch_size=128,
        shuffle=False,
    )

    y_true = []
    y_pred = []

    print("Running test evaluation...", flush=True)

    with torch.no_grad():

        for x, y in loader:

            x = x.to(device)

            logits = model(x)

            pred = logits.argmax(dim=1).cpu().numpy()

            y_pred.extend(pred.tolist())
            y_true.extend(y.numpy().tolist())

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    print("Predictions complete.", flush=True)

    (
        accuracy,
        precision,
        recall,
        f1,
        cm,
        class_precisions,
        class_recalls,
        class_f1s,
    ) = calculate_metrics(y_true, y_pred)

    parameters = count_params(model)

    # Save metrics
    metrics = {
        "model": "lstm",
        "test_accuracy": float(accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "parameters": parameters,
        "class_names": CLASS_NAMES,
        "confusion_matrix": cm.tolist(),
        "per_class": {
            CLASS_NAMES[i]: {
                "precision": float(class_precisions[i]),
                "recall": float(class_recalls[i]),
                "f1": float(class_f1s[i]),
            }
            for i in range(len(CLASS_NAMES))
        },
    }

    metrics_path = RESULT_DIR / "lstm_metrics.json"

    metrics_path.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    # Confusion matrix plot
    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(cm)

    ax.set_title("LSTM Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    ax.set_xticks(np.arange(len(CLASS_NAMES)))
    ax.set_yticks(np.arange(len(CLASS_NAMES)))

    ax.set_xticklabels(
        CLASS_NAMES,
        rotation=45,
        ha="right",
    )

    ax.set_yticklabels(CLASS_NAMES)

    threshold = cm.max() / 2.0

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):

            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white"
                if cm[i, j] > threshold
                else "black",
            )

    fig.colorbar(im, ax=ax)

    fig.tight_layout()

    plot_path = PLOT_DIR / "lstm_confusion_matrix.png"

    fig.savefig(
        plot_path,
        dpi=200,
    )

    plt.close(fig)

    # Final output
    print()
    print("=" * 45)
    print("LSTM FINAL TEST RESULTS")
    print("=" * 45)

    print(f"Test accuracy   : {accuracy:.4f}")
    print(f"Macro precision : {precision:.4f}")
    print(f"Macro recall    : {recall:.4f}")
    print(f"Macro F1        : {f1:.4f}")
    print(f"Parameters      : {parameters:,}")

    print()
    print("Confusion Matrix:")
    print(cm)

    print()
    print(f"Saved metrics : {metrics_path}")
    print(f"Saved plot    : {plot_path}")


if __name__ == "__main__":
    main()
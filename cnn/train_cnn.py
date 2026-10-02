"""1D-CNN for UCI HAR (raw inertial signals). Run: python train_cnn.py"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from tensorflow import keras
from tensorflow.keras import layers

from data_utils import CLASS_NAMES, get_data

SEED = 42
EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

keras.utils.set_random_seed(SEED)


def build_cnn(input_shape=(128, 9), n_classes=6):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(64, 5, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),

        layers.Conv1D(128, 5, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),

        layers.Conv1D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),

        layers.Dropout(0.4),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(n_classes, activation="softmax"),
    ], name="cnn1d")
    model.compile(optimizer=keras.optimizers.Adam(LEARNING_RATE),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def plot_history(history):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(history.history["loss"], label="train")
    ax[0].plot(history.history["val_loss"], label="val")
    ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(history.history["accuracy"], label="train")
    ax[1].plot(history.history["val_accuracy"], label="val")
    ax[1].set_title("Accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "cnn_training_curves.png", dpi=150)
    plt.close(fig)


def main():
    X_tr, y_tr, X_val, y_val, X_test, y_test = get_data(seed=SEED)
    print(f"train {X_tr.shape}  val {X_val.shape}  test {X_test.shape}")

    model = build_cnn(input_shape=X_tr.shape[1:])
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=8,
                                      restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=3, verbose=1),
    ]
    history = model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
                        epochs=EPOCHS, batch_size=BATCH_SIZE,
                        callbacks=callbacks, verbose=2)
    plot_history(history)

    # ---- Evaluation on the official test subjects ----
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0)
    print(f"\nTest accuracy: {acc:.4f}  macro-P {prec:.4f}  "
          f"macro-R {rec:.4f}  macro-F1 {f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4))

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(
        ax=ax, xticks_rotation=45, colorbar=False)
    fig.tight_layout()
    fig.savefig(RESULTS / "cnn_confusion_matrix.png", dpi=150)
    plt.close(fig)

    # Same JSON format for all three models -> easy comparison later
    metrics = {
        "model": "1D-CNN",
        "test_accuracy": float(acc),
        "macro_precision": float(prec),
        "macro_recall": float(rec),
        "macro_f1": float(f1),
        "params": int(model.count_params()),
        "epochs_run": len(history.history["loss"]),
        "confusion_matrix": cm.tolist(),
    }
    (RESULTS / "cnn_metrics.json").write_text(json.dumps(metrics, indent=2))
    model.save(RESULTS / "cnn_model.keras")

    # TODO(Ansh): hyperparameter tuning (kernel size, filters, dropout, lr, batch size)
    # TODO(Ansh): error analysis - why SITTING/STANDING and UPSTAIRS/DOWNSTAIRS get confused
    # TODO(Ansh): inference-time / model-size measurement for the complexity comparison


if __name__ == "__main__":
    main()

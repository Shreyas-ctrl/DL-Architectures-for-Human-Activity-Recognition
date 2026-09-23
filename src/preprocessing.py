# UCI HAR — shared preprocessing pipeline
# Owner (as per synopsis Table I): Shreyas Bibhuty — preprocessing checks
# Input : UCI HAR Dataset raw inertial signals (9 channels x 128 timesteps)
# Output: standardized train/val/test tensors with subject-wise split, no test leakage
import os
import numpy as np

SIGNAL_FILES = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]
ACTIVITIES = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
              "SITTING", "STANDING", "LAYING"]
# Original UCI subject partition (21 train / 9 test). We keep test untouched.
# Validation is formed ONLY from training subjects (no test leakage).
DEFAULT_VAL_SUBJECTS = [27, 28, 29, 30]


def _load_signals(base_dir, split):
    """Load 9 inertial signals -> array (N, 128, 9)."""
    folder = os.path.join(base_dir, split, "Inertial Signals")
    prefix = "train" if split == "train" else "test"
    arrays = []
    for sig in SIGNAL_FILES:
        path = os.path.join(folder, f"{sig}_{prefix}.txt")
        arr = np.loadtxt(path)  # (N, 128)
        arrays.append(arr)
    X = np.stack(arrays, axis=-1)  # (N, 128, 9)
    return X


def _load_labels(base_dir, split):
    prefix = "train" if split == "train" else "test"
    y = np.loadtxt(os.path.join(base_dir, split, f"y_{prefix}.txt")).astype(int) - 1  # 0-5
    subj = np.loadtxt(os.path.join(base_dir, split, f"subject_{prefix}.txt")).astype(int)
    return y, subj


def load_uci_har(base_dir, val_subjects=None):
    """Full pipeline: load -> subject-wise train/val split -> standardize (fit on train only).

    Returns dict with X_train, y_train, X_val, y_val, X_test, y_test, scaler, meta.
    """
    if val_subjects is None:
        val_subjects = DEFAULT_VAL_SUBJECTS
    X_train_full = _load_signals(base_dir, "train")
    y_train_full, subj_train = _load_labels(base_dir, "train")
    X_test = _load_signals(base_dir, "test")
    y_test, subj_test = _load_labels(base_dir, "test")

    # Subject-wise validation mask (only from training subjects)
    train_subjs = sorted(map(int, np.unique(subj_train)))
    val_subjects = [s for s in val_subjects if s in train_subjs]
    if len(val_subjects) == 0:  # fallback: last 3 train subjects
        val_subjects = train_subjs[-3:]
    val_mask = np.isin(subj_train, val_subjects)
    train_mask = ~val_mask

    X_train, y_train = X_train_full[train_mask], y_train_full[train_mask]
    X_val, y_val = X_train_full[val_mask], y_train_full[val_mask]

    # Standardize per channel using TRAIN stats only
    mean = X_train.mean(axis=(0, 1), keepdims=True)  # (1,1,9)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    print(f"Train subjects ({len(train_subjs)-len(val_subjects)}): "
          f"{[s for s in train_subjs if s not in val_subjects]}")
    print(f"Val subjects ({len(val_subjects)}): {val_subjects}")
    print(f"Test subjects: {sorted(map(int, np.unique(subj_test)))}")
    print(f"Shapes: train {X_train.shape} val {X_val.shape} test {X_test.shape}")
    print(f"Channels: {SIGNAL_FILES} | window=128")
    for name, yy in [("train", y_train), ("val", y_val), ("test", y_test)]:
        uniq, cnt = np.unique(yy, return_counts=True)
        print(name, dict(zip([ACTIVITIES[i] for i in uniq], cnt)))

    return {
        "X_train": X_train.astype(np.float32), "y_train": y_train.astype(np.int64),
        "X_val": X_val.astype(np.float32), "y_val": y_val.astype(np.int64),
        "X_test": X_test.astype(np.float32), "y_test": y_test.astype(np.int64),
        "mean": mean.astype(np.float32), "std": std.astype(np.float32),
        "val_subjects": val_subjects,
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/UCI HAR Dataset")
    ap.add_argument("--out_dir", default="models")
    args = ap.parse_args()
    out = load_uci_har(args.data_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    np.savez(os.path.join(args.out_dir, "scaler.npz"), mean=out["mean"], std=out["std"])
    print("Saved scaler to", os.path.join(args.out_dir, "scaler.npz"))

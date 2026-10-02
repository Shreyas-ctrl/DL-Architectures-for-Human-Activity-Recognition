"""Shared data loading for the UCI HAR project (raw inertial signals).

Every model (MLP / 1D-CNN / LSTM) should use get_data() so all three
are compared on the SAME split and preprocessing.
"""
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

URL = ("https://archive.ics.uci.edu/static/public/240/"
       "human+activity+recognition+using+smartphones.zip")
DATA_DIR = Path("data")
ROOT = DATA_DIR / "UCI HAR Dataset"

# 9 raw channels: body acc, body gyro, total acc (x, y, z each)
SIGNALS = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]
CLASS_NAMES = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
               "SITTING", "STANDING", "LAYING"]


def download_dataset():
    """Download + unzip the dataset into ./data (skipped if already there)."""
    if ROOT.exists():
        return
    DATA_DIR.mkdir(exist_ok=True)
    zpath = DATA_DIR / "uci_har.zip"
    if not zpath.exists():
        print("Downloading UCI HAR dataset...")
        urllib.request.urlretrieve(URL, zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(DATA_DIR)
    # The UCI archive contains a nested zip; extract it too if needed
    if not ROOT.exists():
        for inner in DATA_DIR.glob("*.zip"):
            if inner != zpath:
                with zipfile.ZipFile(inner) as z:
                    z.extractall(DATA_DIR)
    if not ROOT.exists():
        raise FileNotFoundError(
            "Could not find 'UCI HAR Dataset' folder. Download the zip manually "
            "from https://archive.ics.uci.edu/dataset/240 and extract it into ./data")


def _load_split(split):
    """Return X (N, 128, 9), y (N,) in 0..5, subjects (N,) for 'train' or 'test'."""
    base = ROOT / split
    channels = [np.loadtxt(base / "Inertial Signals" / f"{s}_{split}.txt")
                for s in SIGNALS]                       # each (N, 128)
    X = np.stack(channels, axis=-1).astype("float32")   # (N, 128, 9)
    y = np.loadtxt(base / f"y_{split}.txt").astype(int) - 1   # labels are 1..6
    subjects = np.loadtxt(base / f"subject_{split}.txt").astype(int)
    return X, y, subjects


def get_data(val_size=0.2, seed=42):
    """Official subject-wise train/test split + validation carved out of
    the TRAIN subjects only (split by subject, so no subject leaks)."""
    download_dataset()
    X_train, y_train, subj_train = _load_split("train")
    X_test, y_test, _ = _load_split("test")

    gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    tr_idx, val_idx = next(gss.split(X_train, y_train, groups=subj_train))
    X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]

    # Standardise per channel using TRAIN statistics only
    mean = X_tr.mean(axis=(0, 1), keepdims=True)
    std = X_tr.std(axis=(0, 1), keepdims=True) + 1e-8
    X_tr, X_val, X_test = [(a - mean) / std for a in (X_tr, X_val, X_test)]

    return X_tr, y_tr, X_val, y_val, X_test, y_test

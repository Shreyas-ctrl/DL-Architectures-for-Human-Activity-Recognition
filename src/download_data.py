"""Download + extract UCI HAR Dataset (id 240). Run once from repo root:
python src/download_data.py
"""
import os, urllib.request, zipfile

URLS = [
    "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip",
]
DEST = os.path.join("data", "uci_har.zip")
OUTDIR = "data"

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    if os.path.isdir(os.path.join(OUTDIR, "UCI HAR Dataset")):
        print("Dataset already present at data/UCI HAR Dataset")
        return
    for url in URLS:
        try:
            print(f"Downloading {url} ...")
            urllib.request.urlretrieve(url, DEST)
            print("Downloaded ->", DEST)
            break
        except Exception as e:
            print(f"Failed {url}: {e}")
    else:
        raise SystemExit("All download URLs failed. Download manually from "
                         "https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones "
                         "and extract to data/")
    with zipfile.ZipFile(DEST, "r") as z:
        z.extractall(OUTDIR)
    print("Extracted to", OUTDIR)

if __name__ == "__main__":
    main()

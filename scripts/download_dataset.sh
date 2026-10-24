#!/usr/bin/env bash
# Download and unzip ECG-Image-Database (Kaggle: physionet/ecg-image-database, ~36 GB).
# Requires a Kaggle token in ~/.kaggle/access_token (or KAGGLE_API_TOKEN) and the kaggle CLI
# from the ecg-bench conda env. Usage: scripts/download_dataset.sh [TARGET_DIR]
set -euo pipefail
TARGET="${1:-$HOME/Documents/ecg-image-database}"
mkdir -p "$TARGET"
kaggle datasets download -d physionet/ecg-image-database -p "$TARGET" --unzip
echo "Done. Dataset at: $TARGET"
du -sh "$TARGET"

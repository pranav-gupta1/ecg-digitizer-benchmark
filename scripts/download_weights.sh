#!/usr/bin/env bash
# Download the two Open-ECG-Digitizer checkpoints (shared by V1 and V2) into weights/.
# Source: Ahus-AIM/Open-ECG-Digitizer (git LFS), served by media.githubusercontent.com.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p weights
BASE="https://media.githubusercontent.com/media/Ahus-AIM/Open-ECG-Digitizer/main/weights"
for f in unet_weights_07072025.pt lead_name_unet_weights_07072025.pt; do
  if [ ! -s "weights/$f" ]; then
    echo "downloading $f"
    curl -L --fail -o "weights/$f" "$BASE/$f"
  fi
done
echo "expected sha256:"
echo "17fe7071ef270102631306127262fc08c250d79d4e3aeb572ab1719dd34d320b  unet_weights_07072025.pt"
echo "840bd6bf2433ee6c22db67f57c861d9d427f29e10a32eeb334f0bcf061b175a2  lead_name_unet_weights_07072025.pt"
echo "actual sha256:"
(cd weights && shasum -a 256 unet_weights_07072025.pt lead_name_unet_weights_07072025.pt)

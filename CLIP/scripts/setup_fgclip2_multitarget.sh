#!/usr/bin/env bash
set -euo pipefail

# Fresh PyTorch-CUDA VM -> uploaded data, dependencies, FG-CLIP2, smoke, research.
# Run from anywhere after cloning the repository. No torch/torchvision upgrade.
# Examples:
#   IMAGES=/data/images RATINGS=/data/ratings.pkl bash CLIP/scripts/setup_fgclip2_multitarget.sh
#   PROTOCOL=holdout VARIANTS="chain-fixed chain-pool chain-prompt" bash ...
#   IMAGES=/data/images RATINGS=/data/ratings.pkl RUN_SMOKE=0 bash ...
#   RUN_DIR=/workspace/run1 RESUME=1 bash ...
#   bash ... --tail-target attractive --protocol holdout

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON="${PYTHON:-python3}"
PACKAGE="$ROOT/CLIP/fgclip2_multitarget"
RUN_DIR="${RUN_DIR:-$PACKAGE/runs/cuda96}"
MODEL_CACHE="${MODEL_CACHE:-$ROOT/models/fgclip2}"
FEATURE_CACHE="${FEATURE_CACHE:-$PACKAGE/.cache}"
DATA_DIR="${DATA_DIR:-$PACKAGE/data/omi}"
DATA_SOURCE="${DATA_SOURCE:-uploaded}"
IMAGES="${IMAGES:-$DATA_DIR/images}"
RATINGS="${RATINGS:-$DATA_DIR/ratings.pkl}"
DATA_URL="${DATA_URL:-}"
DATA_ARCHIVE="${DATA_ARCHIVE:-}"
DATA_SHA256="${DATA_SHA256:-}"
INSTALL_REQUIREMENTS="${INSTALL_REQUIREMENTS:-1}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
RUN_SMOKE="${RUN_SMOKE:-1}"
RUN_TRAINING="${RUN_TRAINING:-1}"
RESUME="${RESUME:-1}"
FINAL_FIT="${FINAL_FIT:-1}"
MODEL="${MODEL:-so400m}"
PRECISION="${PRECISION:-bf16}"
PROTOCOL="${PROTOCOL:-cv}"
VARIANTS="${VARIANTS:-frozen-kernel independent chain-fixed chain-pool chain-prompt chain-lora chain-joint}"
PATCHES="${PATCHES:-576}"
BATCH_SIZE="${BATCH_SIZE:-32}"
ENCODE_BATCH="${ENCODE_BATCH:-16}"
ACCUMULATE="${ACCUMULATE:-2}"
TEXT_CHUNK="${TEXT_CHUNK:-16}"
FOLDS="${FOLDS:-5}"
INNER_FOLDS="${INNER_FOLDS:-4}"
SEED="${SEED:-20260921}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SETUP_DIR="${RUN_DIR}-setup-$STAMP"
mkdir -p "$SETUP_DIR"
cd "$ROOT"

if [[ "$DATA_SOURCE" != "uploaded" && "$DATA_SOURCE" != "omi" ]]; then
  echo "DATA_SOURCE must be uploaded (default) or omi (explicit public-data download)." >&2
  exit 1
fi
if [[ ! -d "$IMAGES" || ! -f "$RATINGS" ]]; then
  if [[ -z "$DATA_ARCHIVE" && -z "$DATA_URL" && "$DATA_SOURCE" != "omi" ]]; then
    echo "Upload your data first, then set IMAGES=/path/to/images RATINGS=/path/to/dim_to_photo_to_ratings.pkl." >&2
    exit 1
  fi
fi

"$PYTHON" - "$SETUP_DIR/torch-constraints.txt" <<'PY'
import sys
from importlib.metadata import version
from pathlib import Path
if sys.version_info < (3, 10):
    raise SystemExit('Python >=3.10 is required. Use a current PyTorch-CUDA container.')
import torch, torchvision
if not torch.cuda.is_available():
    raise SystemExit('CUDA-enabled torch/torchvision must already work in this container (and GPU must be exposed).')
Path(sys.argv[1]).write_text(''.join(f'{name}=={version(name)}\n' for name in ('torch', 'torchvision')))
print('torch', torch.__version__, 'CUDA', torch.version.cuda,
      'GPU', torch.cuda.get_device_name(0),
      'VRAM GiB', round(torch.cuda.get_device_properties(0).total_memory/1024**3, 1), flush=True)
PY

if [[ "$INSTALL_REQUIREMENTS" == "1" ]]; then
  "$PYTHON" -m pip install --disable-pip-version-check \
    -c "$SETUP_DIR/torch-constraints.txt" -r "$PACKAGE/requirements-cuda.txt"
fi
"$PYTHON" - "$SETUP_DIR/torch-constraints.txt" <<'PY'
import sys
from importlib.metadata import version
from pathlib import Path
actual = ''.join(f'{name}=={version(name)}\n' for name in ('torch', 'torchvision'))
if actual != Path(sys.argv[1]).read_text():
    raise SystemExit('Preinstalled torch/torchvision versions changed unexpectedly.')
import torch, torchvision, transformers, sklearn
assert torch.cuda.is_available()
PY
"$PYTHON" -m pip freeze > "$SETUP_DIR/pip-freeze.txt"

if [[ ! -d "$IMAGES" || ! -f "$RATINGS" ]]; then
  if [[ "$IMAGES" != "$DATA_DIR/images" || "$RATINGS" != "$DATA_DIR/ratings.pkl" ]]; then
    echo "Custom IMAGES/RATINGS paths must already exist: $IMAGES ; $RATINGS" >&2
    exit 1
  fi
  mkdir -p "$DATA_DIR"
  DATA_MODE=unpack
  if [[ -z "$DATA_ARCHIVE" ]]; then
    if [[ "$SKIP_DOWNLOAD" == "1" ]]; then
      echo "Dataset missing and SKIP_DOWNLOAD=1. Supply IMAGES/RATINGS or DATA_ARCHIVE." >&2
      exit 1
    fi
    command -v curl >/dev/null || { echo "Install curl in the container." >&2; exit 1; }
    DATA_ARCHIVE="$DATA_DIR/source.tar.gz"
    if [[ -z "$DATA_URL" ]]; then
      OMI_REVISION="53bb3c13113afc3ae67aa6ede4ac9dfe33f306af"
      DATA_URL="https://codeload.github.com/jcpeterson/omi/tar.gz/$OMI_REVISION"
      DATA_MODE=omi
    fi
    if [[ ! -f "$DATA_ARCHIVE" ]]; then
      # Leave a failed transfer as .part; never treat it as a completed archive.
      curl --fail --location --retry 3 --silent --show-error \
        --output "$DATA_ARCHIVE.part" "$DATA_URL"
      mv "$DATA_ARCHIVE.part" "$DATA_ARCHIVE"
    fi
  fi
  "$PYTHON" - "$DATA_ARCHIVE" "$DATA_SHA256" "$SETUP_DIR/data-sha256.txt" <<'PY'
import hashlib, sys
from pathlib import Path
h = hashlib.sha256()
with open(sys.argv[1], 'rb') as f:
    for chunk in iter(lambda: f.read(1024*1024), b''): h.update(chunk)
digest = h.hexdigest()
if sys.argv[2] and digest.lower() != sys.argv[2].lower():
    raise SystemExit('Dataset SHA256 mismatch; archive will not be extracted.')
Path(sys.argv[3]).write_text(digest+'\n')
PY
  "$PYTHON" -m CLIP.fgclip2_multitarget.prepare_data "$DATA_MODE" \
    --archive "$DATA_ARCHIVE" --output "$DATA_DIR"
fi

if [[ "$SKIP_DOWNLOAD" != "1" ]]; then
  "$PYTHON" - "$MODEL_CACHE" "$MODEL" <<'PY'
import sys
from CLIP.fgclip2_core import download_model, DEFAULT_MODEL_ID, BASE_MODEL_ID
model = {'so400m': DEFAULT_MODEL_ID, 'base': BASE_MODEL_ID}[sys.argv[2]]
print('Pinned model snapshot:', download_model(model, cache_dir=sys.argv[1]), flush=True)
PY
fi

COMMON=(--images "$IMAGES" --ratings "$RATINGS" --model "$MODEL"
        --model-cache "$MODEL_CACHE" --cache "$FEATURE_CACHE" --device cuda --precision "$PRECISION")
if [[ "$RUN_SMOKE" == "1" ]]; then
  "$PYTHON" -m CLIP.fgclip2_multitarget "${COMMON[@]}" \
    --variants chain-joint --smoke --patches 128 \
    --output "${RUN_DIR}-smoke-$STAMP" 2>&1 | tee "$SETUP_DIR/smoke.log"
fi

if [[ "$RUN_TRAINING" == "1" ]]; then
  read -r -a VARIANT_ARGS <<< "$VARIANTS"
  EXTRA=()
  [[ "$RESUME" == "1" ]] && EXTRA+=(--resume)
  [[ "$FINAL_FIT" == "1" ]] && EXTRA+=(--final-fit)
  echo "Starting $PROTOCOL research: $VARIANTS; output=$RUN_DIR"
  "$PYTHON" -m CLIP.fgclip2_multitarget "${COMMON[@]}" \
    --variants "${VARIANT_ARGS[@]}" --protocol "$PROTOCOL" \
    --folds "$FOLDS" --inner-folds "$INNER_FOLDS" --seed "$SEED" \
    --patches "$PATCHES" --batch-size "$BATCH_SIZE" --encode-batch "$ENCODE_BATCH" \
    --accumulate "$ACCUMULATE" --text-chunk "$TEXT_CHUNK" --output "$RUN_DIR" \
    "${EXTRA[@]}" "$@" 2>&1 | tee "$SETUP_DIR/training.log"
fi

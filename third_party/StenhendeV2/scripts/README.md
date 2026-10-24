# StenhendeV2/scripts/

Runnable entry points for the digitizer + overlay. Each script declares
its dependencies via a PEP 723 inline metadata block and is run with
`uv`:

```bash
# Single image -> coloured per-lead overlay
uv run src/StenhendeV2/scripts/stenhede_overlay.py \
    --image photo.jpg --out overlay.png

# Folder of images -> canonical CSV + 4-panel PNG per image
uv run src/StenhendeV2/scripts/stenhede_digitize.py \
    --config inference_wrapper_george-moody-2024.yml \
    DATA.images_path=/tmp/in DATA.output_path=/tmp/out
```

Both expect `STENHEDE_WEIGHTS_PATH` to be set in `data.env` at the
project root.

The scripts add the project root to `sys.path` because `uv run` with a
PEP 723 block uses an ephemeral env that does not include it.

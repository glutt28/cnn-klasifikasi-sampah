# UniRepLKNet-S pipeline — ITERA DL Studio 2

This overlay adds the training pipeline and local dataset preparation. Keep the course-provided `evaluate.py` unchanged. The five-class order is fixed:

1. `botol_plastik`
2. `gelas_plastik`
3. `kertas_kardus`
4. `saset_kemasan`
5. `styrofoam`

## 1. One virtual environment

Use `.venv` as the only project environment. For this Gentoo machine with an RX 6700 XT, install the ROCm build:

```bash
uv venv --python 3.12 .venv
./scripts/install_amd.sh
```

The installer uses the official PyTorch ROCm 7.2 wheel index, PyPI for other dependencies, and explicit PyTorch/TorchVision version pins. It intentionally uses `--index-strategy unsafe-best-match` because `uv` otherwise can stop at the first index containing a package name.

To run on a non-AMD setup, use the appropriate PyTorch/TorchVision build for that machine, then install `requirements.txt`.

### Migrating from the old two-environment state

Do not move/rename a virtual environment directory: generated entry-point scripts can contain absolute paths. Instead, create or update `.venv` in place using the installer above. Test it, then remove the old environments only after confirming the new one works:

```bash
rm -rf .venv-rocm-official
# Remove .venv first only if you explicitly want to recreate it from scratch:
# rm -rf .venv
```

If `.venv` already exists, the installer updates its installed PyTorch/TorchVision to the pinned ROCm builds. The new environment should report `torch.version.hip` as non-`None` and `torch.cuda.is_available()` as `True` on a working GPU setup.

> **RX 6700 XT caveat:** this card is not officially supported by the ROCm 7.2 Linux GPU matrix. The tested setup reports `gfx1030` through `HSA_OVERRIDE_GFX_VERSION=10.3.0`, a community compatibility workaround. Basic GPU/CNN forward-backward tests passed, but that does not guarantee every kernel is compatible.

## 2. Download and prepare the dataset

Download the curated dataset ZIP from the course Google Drive. This pipeline does not re-download the `image_url` entries in `datasets/train_sources.csv`: those URLs are not the curated dataset source and may fail or return different images.

Put the downloaded archive at `datasets/course_dataset.zip` and run:

```bash
.venv/bin/python scripts/setup_dataset.py \
    --archive datasets/course_dataset.zip
```

The archive may contain a top-level folder such as `train/`; the script searches for the directory whose direct subfolders are the five class names. It extracts into `datasets/raw/extracted/`, validates readable image files, and creates this split:

```text
datasets/
├── course_dataset.zip              # downloaded by you; ignored by Git
├── raw/extracted/                  # extracted Drive dataset; ignored by Git
└── split/
    ├── train/<class-name>/
    ├── val/<class-name>/
    └── manifest.json
```

If you extracted the Drive dataset yourself, pass the exact directory that directly contains the five class folders:

```bash
.venv/bin/python scripts/setup_dataset.py --source /path/to/dataset/train
```

If automatic detection finds multiple datasets, select the intended Drive dataset with `--source`. To regenerate a previously created split, use `--force`; it removes only `datasets/split/`, not the downloaded or extracted source. If you replace the ZIP and need to extract it again, first remove `datasets/raw/extracted/` manually; an existing valid extraction is reused. The default validation ratio is 20% per class, with random seed 42. The script prefers hard links to avoid duplicating images and falls back to copies when necessary.

## 3. Test the GPU and run training

First test actual GPU compute—not just GPU detection:

```bash
HSA_OVERRIDE_GFX_VERSION=10.3.0 MIOPEN_FIND_MODE=FAST \
    .venv/bin/python scripts/check_gpu.py
```

Run a one-epoch smoke test first:

```bash
./scripts/train_amd.sh --epochs 1 --batch-size 4 --workers 2 --patience 1
```

Then train normally:

```bash
./scripts/train_amd.sh --epochs 20 --batch-size 4 --workers 2 --patience 6
```

The launch script defaults `HSA_OVERRIDE_GFX_VERSION=10.3.0` and `MIOPEN_FIND_MODE=FAST` for the tested RX 6700 XT setup. Both environment variables can be overridden in the command environment. `FAST` aims to avoid costly exhaustive MIOpen algorithm searches; performance may differ from a fully supported GPU.

The first pretrained run fetches the ImageNet-1K UniRepLKNet-S checkpoint from Hugging Face and caches it. Subsequent runs reuse the cache. Training saves the best local validation Macro-F1 checkpoint as `submission/model.pth` and writes `outputs/training_history.csv`.

Optional training arguments:

```bash
./scripts/train_amd.sh --epochs 25 --batch-size 4 --patience 7
./scripts/train_amd.sh --class-weighting none
```

## 4. Validate the submission contract

```bash
.venv/bin/python scripts/check_submission.py
.venv/bin/python evaluate.py --submission ./submission --data ./datasets/split/val
```

`get_model()` returns five-class logits and does not load weights. The evaluator loads `submission/model.pth` with `strict=True`. The architecture implementation is stored in `submission/unireplknet.py`, so inference does not download architecture code or pretrained weights.

Do not commit dataset images, extracted data, virtual environments, or generated training outputs. Commit the source code and required submission files according to the course rules.

# UniRepLKNet-S clean training overlay

Overlay the archive contents onto the root of `glutt28/cnn-klasifikasi-sampah`. The course-provided `evaluate.py` is not included or modified.

The setup is intentionally split by responsibility:

- `requirements.txt`: default PyPI dependencies.
- `requirements-amd.txt` + `scripts/install_amd.sh`: pinned PyTorch ROCm 7.2 environment.
- `scripts/setup_dataset.py`: validates a downloaded course dataset and creates a deterministic train/validation split. It does **not** read `train_sources.csv` or fetch individual images.
- `scripts/check_gpu.py`: reusable GPU matrix/CNN forward-backward smoke test.
- `scripts/train_amd.sh`: launches training with the RX 6700 XT compatibility environment used in this setup.
- `submission/unireplknet.py`: vendored architecture implementation; architecture source is not downloaded at evaluation time.

Start with [`docs/UNIREPLKNET_PIPELINE.md`](docs/UNIREPLKNET_PIPELINE.md).

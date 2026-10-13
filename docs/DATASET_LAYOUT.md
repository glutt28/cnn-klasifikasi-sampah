# Dataset layout

The source dataset downloaded from the course Google Drive must have one directory containing all five class folders directly:

```text
train/
├── botol_plastik/
├── gelas_plastik/
├── kertas_kardus/
├── saset_kemasan/
└── styrofoam/
```

The class folders may contain nested image directories. Supported formats include JPEG, PNG, WebP, BMP, and TIFF. Unreadable images are skipped and reported.

For a downloaded ZIP:

```bash
.venv/bin/python scripts/setup_dataset.py --archive datasets/course_dataset.zip
```

For a directory that has already been extracted:

```bash
.venv/bin/python scripts/setup_dataset.py --source /path/to/train
```

The script creates `datasets/split/train/` and `datasets/split/val/`, using a deterministic 80/20 per-class split by default. `--force` replaces only the generated split. The script does not read the CSV manifest or download individual images.

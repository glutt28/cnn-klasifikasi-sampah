# Local dataset directory

Download the curated dataset ZIP from the course Google Drive and save it here as `course_dataset.zip`, or extract it into a directory containing the five class folders. Dataset images, archives, and generated splits are ignored by Git.

Prepare the split with:

```bash
.venv/bin/python scripts/setup_dataset.py --archive datasets/course_dataset.zip
```

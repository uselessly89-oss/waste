# Dataset

Place YOLO segmentation data here (images and labels are gitignored because real datasets and trained artifacts are large).

```text
dataset/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
```

Each image should have a matching `.txt` label file with YOLO segmentation annotations. Keep train/val/test independent and representative of real conveyor conditions.

Classes in `configs/data.yaml`: 0 plastic, 1 paper, 2 metal, 3 organic.

Before training, run:

```bash
python tools/dataset_audit.py
```

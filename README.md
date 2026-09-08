# Waste Segregation AI

A from-scratch prototype for automated waste sorting on a conveyor.

## Pipeline

Camera → YOLO detector/segmenter → ByteTrack → temporal consensus → uncertainty/safety gate → per-object conveyor timing → actuator interface → target bin

Initial classes:
- plastic
- paper
- metal
- organic

The design is extensible to glass, textile, e-waste, and other.

## Important limitation
An RGB camera cannot reliably infer hidden contents or chemistry. The system therefore has an explicit `REJECT` state for uncertain cases. Material verification can be added later with validated auxiliary sensors such as NIR/hyperspectral, depth, weight, or other industrial sensing.

## Setup

```bash
python -m pip install -r requirements.txt
```

Prepare a YOLO dataset under `dataset/` and edit `configs/data.yaml` if needed.

Audit the dataset:

```bash
python tools/dataset_audit.py
```

Train:

```bash
python src/train.py --data configs/data.yaml --model yolo11n-seg.pt --epochs 100 --imgsz 640
```

Run camera inference in simulation mode:

```bash
python src/detect.py --model runs/waste/waste_detector/weights/best.pt --source 0 --show --actuator simulation
```

Run a video file:

```bash
python src/detect.py --model runs/waste/waste_detector/weights/best.pt --source path/to/video.mp4 --show
```

## Safety defaults

- Hardware output is **simulation by default**.
- Low confidence or unstable temporal consensus becomes `REJECT`.
- Poor frame quality is rejected.
- Each track can trigger at most one actuator action.
- Sorting uses a per-object queue rather than a global `sleep()`.
- Missed timing windows do not trigger late actuation.

## Dataset guidance

Include real conveyor data: lighting changes, blur, dirt, crushed/deformed items, occlusion, overlap, different orientations, empty belt, different camera distances, and representative backgrounds. Split train/validation/test by scene/session where possible to reduce leakage.

## Output schema

```json
{
  "track_id": 17,
  "class": "plastic",
  "confidence": 0.93,
  "bounding_box": [112, 85, 356, 420],
  "center_x": 234,
  "center_y": 252,
  "consensus": 0.86,
  "status": "SORT_PLASTIC"
}
```

## Production hardening

Before connecting real actuators, validate electrical isolation, emergency-stop behavior, PLC interlocks, timing calibration, missed-object behavior, and the actual wrong-bin rate using a representative holdout test set.

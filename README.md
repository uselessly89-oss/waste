# ♻️ Waste Segregation AI — Easy Camera + Automated Sorting

A complete computer-vision prototype for waste segregation. It has two modes:

1. **Easy Camera Mode** — point a laptop webcam or open the app on a phone over the same Wi-Fi and inspect waste immediately.
2. **Conveyor Mode** — detect → segment → track → temporally verify → estimate sorting arrival → safely command the correct actuator.

## What it can recognize

Initial model classes:
- `plastic`
- `paper`
- `metal`
- `organic`

The architecture can later add `glass`, `textile`, `e_waste`, and `other`.

## Easiest way: laptop or phone camera

Install the app dependencies:

```bash
pip install -r requirements-app.txt
```

Start:

```bash
python app.py
```

### Laptop

Open the displayed browser address and select **Webcam**.

### Phone

Keep the laptop and phone on the same Wi-Fi network. Start the app with:

```bash
python app.py
```

Then open `http://<LAPTOP-IP>:7860` on the phone and use the phone camera.

See `docs/EASY_CAMERA_MODE.md` for the full procedure.

## Full automated sorting pipeline

```text
Camera
  ↓
Image quality gate
  ↓
YOLO segmentation + detection
  ↓
ByteTrack object tracking
  ↓
Temporal consensus
  ↓
Overlap / uncertainty safety gate
  ↓
Per-object conveyor ETA
  ↓
Sorting queue
  ↓
Actuator interface
  ↓
Plastic / Paper / Metal / Organic bin
```

Run:

```bash
python -m pip install -r requirements.txt
python tools/dataset_audit.py
python src/train.py --data configs/data.yaml --model yolo11n-seg.pt --epochs 100 --imgsz 640
python src/detect_v2.py --model runs/waste/waste_detector/weights/best.pt --source 0 --show --actuator simulation
```

## Mixed waste clusters

The system is designed to **handle difficult clusters rather than blindly guess**. Segmentation helps separate visible overlapping objects and tracking keeps identities across frames. Temporal voting reduces one-frame mistakes and the overlap gate rejects cases where objects are too visually inseparable.

However, no RGB-only model can guarantee identification of an item completely hidden inside a pile. For those cases the correct behavior is `REJECT` / re-feed / second-pass separation. Production-grade identification of hidden material requires additional validated sensing such as depth, weight, NIR/hyperspectral, or another appropriate sensor.

## Dataset

Put YOLO segmentation data under:

```text
dataset/
├── images/train
├── images/val
├── images/test
├── labels/train
├── labels/val
└── labels/test
```

Use real conveyor examples containing lighting changes, shadows, motion blur, dirty/wet objects, crushed/deformed objects, partial occlusion, severe overlap, different orientations, camera distances, empty frames, and multiple simultaneous objects.

Audit it:

```bash
python tools/dataset_audit.py
```

## Safety

Physical actuator control is disabled unless explicitly enabled. Always validate emergency stops, electrical isolation, PLC interlocks, actuator timing, missed-object behavior, and wrong-bin rate before connecting a real machine.

## Project status

The repository contains the application, inference pipeline, training code, tracking/temporal logic, safety gates, conveyor scheduling, actuator adapters, dataset audit, evaluation, and tests. **A trained `best.pt` cannot be included until a real labeled dataset is supplied.**

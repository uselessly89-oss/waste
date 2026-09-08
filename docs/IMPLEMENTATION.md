# Waste Segregation Implementation

## What is implemented

1. **Capture** — OpenCV reads a camera, video, or image sequence.
2. **Detection + segmentation** — Ultralytics YOLO segmentation models produce class, confidence, boxes and optional instance masks.
3. **Tracking** — ByteTrack maintains an identity for each item so multiple objects can be on the conveyor simultaneously.
4. **Temporal consensus** — Per-track voting reduces one-frame misclassification.
5. **Quality gate** — Very dark, very bright, or strongly blurred frames are rejected.
6. **Overlap gate** — Strongly overlapping boxes are held/rejected rather than blindly actuated.
7. **Material fusion hook** — Optional NIR/depth/other sensor evidence can confirm a vision class; conflicts become `REJECT`.
8. **Conveyor scheduling** — Each track gets its own sorting event instead of a global sleep.
9. **Actuation** — Simulation is the default. Arduino serial output is explicitly opt-in.
10. **Evaluation + CI** — Dataset audit, model evaluation, unit tests and GitHub Actions are included.

## Classes

Initial benchmark:

- `plastic`
- `paper`
- `metal`
- `organic`

Future classes can be added only after collecting and validating labeled data: `glass`, `textile`, `e_waste`, `other`.

## Dataset requirements

Use YOLO segmentation labels for the segmentation model. Keep train/validation/test identities separate. Include clean and difficult cases: different lighting, blur, camera distance, object orientation, crushed/dirty waste, partial occlusion, multiple objects, empty conveyor frames, and realistic backgrounds.

A practical starting point is 500–1000 annotated images per class; production systems should collect substantially more and report per-class metrics instead of relying only on overall mAP.

## Training

```bash
python -m pip install -r requirements.txt
python tools/dataset_audit.py
python src/train.py --data configs/data.yaml --model yolo11n-seg.pt --epochs 100 --imgsz 640
```

The model weights are intentionally not committed to Git because they are large. Training cannot produce meaningful `best.pt` until the dataset is supplied.

## Evaluation

```bash
python tools/evaluate.py --model runs/waste/waste_detector/weights/best.pt --data configs/data.yaml --split test
```

Record box/segmentation mAP, per-class precision/recall, confusion matrix, false-sort rate, reject rate and missed-item rate.

## Real-time simulation

```bash
python src/detect_v2.py --model runs/waste/waste_detector/weights/best.pt --source 0 --show --actuator simulation
```

Do not enable physical actuation until the complete camera-to-diverter geometry, belt speed, actuator latency, emergency stop and mechanical interlocks have been validated.

## Hardware protocol

The Arduino adapter sends one newline-terminated command such as `BIN_PLASTIC`. The microcontroller must independently enforce safe pulse limits, emergency-stop state and actuator interlocks. The computer vision application is not a safety controller.

## Important limitation

RGB images cannot reliably identify hidden contents, chemical composition or material properties when the visible appearance is ambiguous. The correct behavior is `REJECT`/manual review, or use calibrated auxiliary sensing such as NIR/hyperspectral/depth/weight where appropriate.

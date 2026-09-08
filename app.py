"""Universal Waste Segregator web app.

Runs inference on the computer hosting the app while accepting a camera,
photo, or video stream from a laptop/phone browser. This is the easiest UI
for demonstrations: open the displayed LAN URL on the phone and use its
rear camera.
"""
import os
from pathlib import Path
import cv2
import numpy as np
import gradio as gr
from ultralytics import YOLO

MODEL_PATH = os.getenv("WASTE_MODEL", "runs/waste/waste_detector/weights/best.pt")
model = None

COLORS = {}


def get_model():
    global model
    if model is None:
        if not Path(MODEL_PATH).exists():
            raise FileNotFoundError(
                f"Model not found: {MODEL_PATH}. Train a model first or set WASTE_MODEL."
            )
        model = YOLO(MODEL_PATH)
    return model


def annotate(image, conf=0.45, iou=0.50, cluster_mode=True):
    if image is None:
        return None, "No image received."
    m = get_model()
    frame = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    results = m.predict(frame, conf=float(conf), iou=float(iou), imgsz=640, verbose=False)
    r = results[0]
    counts = {}
    total = 0
    if r.boxes is not None:
        for i, box in enumerate(r.boxes.xyxy.cpu().numpy()):
            cls_id = int(r.boxes.cls[i].item())
            score = float(r.boxes.conf[i].item())
            name = str(m.names[cls_id])
            counts[name] = counts.get(name, 0) + 1
            total += 1
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.putText(frame, f"{name} {score:.2f}", (x1, max(20, y1-7)),
                        cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 2)

    if cluster_mode and total == 0:
        message = "NO CONFIDENT OBJECTS — send this item/cluster to REJECT."
    elif cluster_mode:
        parts = ", ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
        message = f"CLUSTER: {total} visible object(s) | {parts} | Unknown/hidden material → REJECT"
    else:
        message = f"Detected {total} object(s)."
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), message


def run():
    return gr.Interface(
        fn=annotate,
        inputs=[
            gr.Image(sources=["webcam", "upload"], type="numpy", label="Waste camera / photo"),
            gr.Slider(0.20, 0.90, value=0.45, step=0.05, label="Confidence threshold"),
            gr.Slider(0.20, 0.90, value=0.50, step=0.05, label="IoU threshold"),
            gr.Checkbox(value=True, label="Cluster mode"),
        ],
        outputs=[gr.Image(type="numpy", label="Segregation view"), gr.Textbox(label="Decision")],
        title="♻️ Universal Waste Segregator",
        description="Use a laptop webcam or phone browser camera. Visible items are segmented/detected; uncertain or hidden material is routed to REJECT rather than guessed.",
    )


if __name__ == "__main__":
    run().launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")), share=False)

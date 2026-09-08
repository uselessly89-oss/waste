import argparse
import json
from pathlib import Path
import cv2
import yaml
from ultralytics import YOLO

from temporal_voting import TemporalVoter
from quality import assess_frame
from conveyor import SortingQueue, eta_to_sorting_point
from sorter import DecisionEngine
from hardware.simulator import Simulator


def source_value(value):
    try:
        return int(value)
    except ValueError:
        return value


def main():
    p = argparse.ArgumentParser(description="Real-time waste detection and sorting")
    p.add_argument("--model", required=True)
    p.add_argument("--source", default="0")
    p.add_argument("--config", default="configs/runtime.yaml")
    p.add_argument("--show", action="store_true")
    p.add_argument("--actuator", choices=["simulation"], default="simulation")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    model = YOLO(args.model)
    names = model.names
    voter = TemporalVoter(cfg["decision"]["temporal_window"])
    engine = DecisionEngine(
        min_confidence=cfg["decision"]["min_confidence"],
        min_consensus=cfg["decision"]["min_consensus_ratio"],
        min_observations=cfg["decision"]["min_observations"],
        cooldown_s=cfg["decision"]["cooldown_s"],
    )
    queue = SortingQueue()
    actuator = Simulator()

    cap = cv2.VideoCapture(source_value(args.source))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera/video: {args.source}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        quality = assess_frame(frame, **cfg["quality"])
        if not quality["ok"]:
            if args.show:
                cv2.putText(frame, "REJECT: poor frame quality", (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("Waste Segregation", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            continue

        results = model.track(frame, persist=True, tracker="bytetrack.yaml",
                              conf=0.20, iou=0.70, verbose=False)
        result = results[0]
        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.cpu().numpy().astype(int)
            classes = result.boxes.cls.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()

            for box, track_id, cls_id, conf in zip(boxes, ids, classes, confs):
                x1, y1, x2, y2 = map(float, box)
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                winner, winner_conf, consensus, count = voter.update(track_id, int(cls_id), float(conf))
                class_name = names[int(winner)]
                eta = eta_to_sorting_point(cfg["belt"]["camera_to_diverter_m"], cfg["belt"]["speed_mps"])
                decision = engine.evaluate(track_id, class_name, winner_conf, consensus, count, eta)

                if decision and decision.accepted:
                    queue.schedule(track_id, decision.action, decision.eta_s)

                output = {
                    "track_id": int(track_id),
                    "class": class_name,
                    "confidence": round(float(winner_conf), 3),
                    "bounding_box": [round(x1), round(y1), round(x2), round(y2)],
                    "center_x": round(cx),
                    "center_y": round(cy),
                    "consensus": round(float(consensus), 3),
                    "status": decision.action if decision else "OBSERVING",
                }
                print(json.dumps(output))

                if args.show:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID {track_id}: {class_name} {winner_conf:.2f}",
                                (int(x1), max(20, int(y1)-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)

        # Execute only commands whose individual ETA has expired.
        for track_id, action in queue.due():
            actuator.command(action, cfg["actuator"]["pulse_s"])

        if args.show:
            cv2.imshow("Waste Segregation", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

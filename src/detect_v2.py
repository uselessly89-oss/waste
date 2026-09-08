"""Real-time segmentation + tracking + safe sorting pipeline.

Run from the repository root. Default actuator mode is simulation and the
physical actuator requires an explicit configuration enable flag.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))

from pipeline import TrackObservation, TrackHistory, temporal_consensus, severe_overlap, eta_from_position
from quality import assess_frame
from conveyor import SortingQueue
from sorter import DecisionEngine
from hardware.simulator import Simulator


def source_value(value):
    try:
        return int(value)
    except ValueError:
        return value


def mask_area(result, index):
    if result.masks is None or result.masks.data is None:
        return 0.0
    mask = result.masks.data[index].cpu().numpy()
    return float(mask.sum())


def build_actuator(cfg, requested_mode):
    mode = requested_mode or cfg['actuator'].get('mode', 'simulation')
    if mode == 'simulation':
        return Simulator()
    if mode == 'arduino':
        from hardware.arduino_safe import ArduinoSafe
        ac = cfg['actuator']
        return ArduinoSafe(ac.get('serial_port', ''), ac.get('baudrate', 115200), enabled=bool(ac.get('enabled', False)))
    raise ValueError(f'Unsupported actuator mode: {mode}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--source', default='0')
    p.add_argument('--config', default='configs/runtime_v2.yaml')
    p.add_argument('--show', action='store_true')
    p.add_argument('--actuator', choices=['simulation', 'arduino'], default=None)
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    model = YOLO(args.model)
    names = model.names
    history = TrackHistory(cfg['decision']['temporal_window'])
    engine = DecisionEngine(**{k: cfg['decision'][k] for k in ('min_confidence', 'min_consensus_ratio', 'min_observations', 'cooldown_s')})
    queue = SortingQueue()
    actuator = build_actuator(cfg, args.actuator)
    cap = cv2.VideoCapture(source_value(args.source))
    if not cap.isOpened():
        raise RuntimeError(f'Unable to open source: {args.source}')

    previous_positions = {}
    previous_times = {}
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            now = time.monotonic()
            quality = assess_frame(frame, **cfg['quality'])
            if not quality['ok']:
                if args.show:
                    cv2.putText(frame, 'REJECT: poor frame quality', (20, 35), 0, .7, (0, 0, 255), 2)
                    cv2.imshow('Waste Segregation', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                continue

            results = model.track(frame, persist=True, tracker='bytetrack.yaml', conf=0.20, iou=0.70, verbose=False)
            result = results[0]
            active_ids = []
            boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else []
            if result.boxes is not None and result.boxes.id is not None:
                ids = result.boxes.id.cpu().numpy().astype(int)
                classes = result.boxes.cls.cpu().numpy().astype(int)
                confs = result.boxes.conf.cpu().numpy()
                active_ids = ids.tolist()
                for i, (box, track_id, cls_id, conf) in enumerate(zip(boxes, ids, classes, confs)):
                    x1, y1, x2, y2 = map(float, box)
                    cx, cy = (x1+x2)/2, (y1+y2)/2
                    others = [b for j, b in enumerate(boxes) if j != i]
                    occluded = severe_overlap(box, others, cfg['decision']['max_overlap_iou'])
                    obs = TrackObservation(int(track_id), int(cls_id), names[int(cls_id)], float(conf), cx, cy,
                                           mask_area(result, i), max(0.0, x2-x1)*max(0.0, y2-y1), now)
                    q = history.add(obs)
                    winner_id, winner_conf, consensus, count = temporal_consensus(q)
                    status = 'OBSERVING'
                    reason = 'temporal_consensus'
                    if occluded:
                        status, reason = 'REJECT', 'severe_overlap'
                    elif winner_id is not None:
                        class_name = names[int(winner_id)]
                        distance = cfg['belt']['camera_to_diverter_m']
                        eta = eta_from_position(distance, cfg['belt']['speed_mps'], cfg['belt'].get('actuator_latency_s', 0.0))
                        decision = engine.evaluate(int(track_id), class_name, winner_conf, consensus, count, eta or 0.0)
                        if decision and decision.accepted:
                            if queue.schedule(int(track_id), decision.action, decision.eta_s):
                                status, reason = decision.action, decision.reason
                            else:
                                engine.release(int(track_id))
                                status, reason = 'REJECT', 'queue_rejected'
                    output = {'track_id': int(track_id), 'class': names[int(winner_id)] if winner_id is not None else names[int(cls_id)],
                              'confidence': round(float(winner_conf), 3), 'bbox': [round(x1), round(y1), round(x2), round(y2)],
                              'center': [round(cx), round(cy)], 'consensus': round(float(consensus), 3),
                              'mask_fill_ratio': round(obs.mask_fill_ratio, 3), 'occluded': occluded, 'status': status, 'reason': reason}
                    print(json.dumps(output))
                    if args.show:
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(frame, f'ID {track_id} {output["class"]} {winner_conf:.2f}', (int(x1), max(20, int(y1)-8)), 0, .55, (0,255,0), 2)

            history.purge(active_ids)
            for track_id, action in queue.due():
                if actuator.command(action, cfg['actuator']['pulse_s']):
                    engine.record_actuation(track_id)
                else:
                    engine.release(track_id)

            if args.show:
                cv2.imshow('Waste Segregation', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        close = getattr(actuator, 'close', None)
        if close:
            close()


if __name__ == '__main__':
    main()

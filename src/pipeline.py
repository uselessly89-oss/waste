"""Robust perception-to-sort decision helpers.

This module keeps perception, temporal consensus, quality/occlusion gates,
and conveyor timing separate from physical actuator control.
"""
from dataclasses import dataclass
from collections import deque
import math
import time


@dataclass
class TrackObservation:
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    cx: float
    cy: float
    mask_area: float = 0.0
    bbox_area: float = 0.0
    frame_time: float = 0.0

    @property
    def mask_fill_ratio(self):
        if self.bbox_area <= 0:
            return 0.0
        return self.mask_area / self.bbox_area


class TrackHistory:
    def __init__(self, window=10, max_tracks=500):
        self.window = int(window)
        self.max_tracks = int(max_tracks)
        self.data = {}

    def add(self, obs: TrackObservation):
        q = self.data.setdefault(obs.track_id, deque(maxlen=self.window))
        q.append(obs)
        if len(self.data) > self.max_tracks:
            oldest = min(self.data, key=lambda k: self.data[k][-1].frame_time)
            if oldest != obs.track_id:
                self.data.pop(oldest, None)
        return q

    def get(self, track_id):
        return self.data.get(track_id, ())

    def remove(self, track_id):
        self.data.pop(track_id, None)

    def purge(self, active_ids):
        active = set(active_ids)
        for track_id in list(self.data):
            if track_id not in active:
                self.data.pop(track_id, None)


def temporal_consensus(history):
    """Return majority class, mean confidence, and vote ratio."""
    if not history:
        return None, 0.0, 0.0, 0
    votes = {}
    for obs in history:
        votes[obs.class_id] = votes.get(obs.class_id, 0) + 1
    winner = max(votes, key=votes.get)
    winning = [o.confidence for o in history if o.class_id == winner]
    return winner, sum(winning) / len(winning), votes[winner] / len(history), len(history)


def overlap_ratio(a, b):
    """IoU for xyxy boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    aa = max(0.0, ax2-ax1) * max(0.0, ay2-ay1)
    ba = max(0.0, bx2-bx1) * max(0.0, by2-by1)
    union = aa + ba - inter
    return inter / union if union else 0.0


def severe_overlap(box, other_boxes, threshold=0.45):
    return any(overlap_ratio(box, other) >= threshold for other in other_boxes)


def estimate_belt_velocity(previous, current, pixels_per_meter, dt):
    """Estimate longitudinal belt velocity from image motion.

    The caller should use a calibrated conveyor axis. If calibration is absent,
    return None instead of guessing physical units.
    """
    if previous is None or dt <= 0 or pixels_per_meter <= 0:
        return None
    return (current - previous) / dt / pixels_per_meter


def eta_from_position(distance_remaining_m, belt_speed_mps, actuator_latency_s=0.0):
    if belt_speed_mps <= 0:
        return None
    return max(0.0, distance_remaining_m / belt_speed_mps - max(0.0, actuator_latency_s))

from dataclasses import dataclass
import time


@dataclass
class TrackState:
    track_id: int
    first_seen: float
    last_seen: float
    center_x: float
    center_y: float
    class_name: str
    confidence: float
    acted: bool = False


def travel_time(distance_m, speed_mps):
    if speed_mps <= 0:
        raise ValueError("Conveyor speed must be > 0")
    return distance_m / speed_mps


def eta_to_sorting_point(camera_to_diverter_m, belt_speed_mps):
    return travel_time(camera_to_diverter_m, belt_speed_mps)


class SortingQueue:
    """Per-track scheduling. Multiple waste items can coexist on the belt."""

    def __init__(self):
        self.items = {}

    def schedule(self, track_id, action, eta_s):
        if track_id in self.items:
            return False
        self.items[track_id] = {
            "action": action,
            "due_at": time.monotonic() + max(0.0, eta_s),
            "created_at": time.monotonic(),
        }
        return True

    def due(self):
        now = time.monotonic()
        ready = []
        for track_id, item in list(self.items.items()):
            if item["due_at"] <= now:
                ready.append((track_id, item["action"]))
                del self.items[track_id]
        return ready

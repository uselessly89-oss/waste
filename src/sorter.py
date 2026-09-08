import time
from dataclasses import dataclass

SORTING_ACTIONS = {
    "plastic": "BIN_PLASTIC",
    "paper": "BIN_PAPER",
    "metal": "BIN_METAL",
    "organic": "BIN_ORGANIC",
}


@dataclass
class SortDecision:
    track_id: int
    class_name: str
    confidence: float
    consensus: float
    action: str
    eta_s: float
    accepted: bool
    reason: str


class DecisionEngine:
    def __init__(self, min_confidence=0.55, min_consensus=0.70,
                 min_observations=5, cooldown_s=0.5):
        self.min_confidence = min_confidence
        self.min_consensus = min_consensus
        self.min_observations = min_observations
        self.cooldown_s = cooldown_s
        self.acted = set()
        self.last_actuation = 0.0

    def evaluate(self, track_id, class_name, confidence, consensus,
                 observations, eta_s):
        if track_id in self.acted:
            return None
        if observations < self.min_observations:
            return SortDecision(track_id, class_name, confidence, consensus,
                                "REJECT", eta_s, False, "insufficient_temporal_evidence")
        if confidence < self.min_confidence:
            return SortDecision(track_id, class_name, confidence, consensus,
                                "REJECT", eta_s, False, "low_confidence")
        if consensus < self.min_consensus:
            return SortDecision(track_id, class_name, confidence, consensus,
                                "REJECT", eta_s, False, "unstable_class")
        action = SORTING_ACTIONS.get(class_name)
        if action is None:
            return SortDecision(track_id, class_name, confidence, consensus,
                                "REJECT", eta_s, False, "unsupported_class")
        if time.monotonic() - self.last_actuation < self.cooldown_s:
            return SortDecision(track_id, class_name, confidence, consensus,
                                "REJECT", eta_s, False, "actuator_cooldown")
        self.acted.add(track_id)
        self.last_actuation = time.monotonic()
        return SortDecision(track_id, class_name, confidence, consensus,
                            action, eta_s, True, "stable_prediction")

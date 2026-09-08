"""Optional material-evidence fusion layer.

RGB vision cannot see the hidden contents of a closed/opaque object. This
module therefore never invents a material label: auxiliary sensors may confirm
or reject a vision result, and ambiguous items remain REJECT.
"""
from dataclasses import dataclass


@dataclass
class SensorEvidence:
    available: bool = False
    material: str | None = None
    confidence: float = 0.0


class MaterialFusionGate:
    def __init__(self, minimum_sensor_confidence=0.70):
        self.minimum_sensor_confidence = float(minimum_sensor_confidence)

    def decide(self, vision_class, vision_confidence, evidence=None):
        if evidence is None or not evidence.available:
            return vision_class, float(vision_confidence), 'VISION_ONLY'
        if evidence.material and evidence.confidence >= self.minimum_sensor_confidence:
            if evidence.material != vision_class:
                return 'REJECT', min(float(vision_confidence), float(evidence.confidence)), 'SENSOR_VISION_CONFLICT'
            return vision_class, min(float(vision_confidence), float(evidence.confidence)), 'FUSED'
        return 'REJECT', float(vision_confidence), 'INSUFFICIENT_SENSOR_EVIDENCE'

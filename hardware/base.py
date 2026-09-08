from abc import ABC, abstractmethod


class Actuator(ABC):
    """Common actuator interface used by simulation and hardware adapters."""

    @abstractmethod
    def command(self, action: str, pulse_s: float = 0.15) -> bool:
        raise NotImplementedError


ALLOWED_ACTIONS = {
    "BIN_PLASTIC",
    "BIN_PAPER",
    "BIN_METAL",
    "BIN_ORGANIC",
    "REJECT",
}


def validate_action(action: str) -> str:
    action = str(action).strip().upper()
    if action not in ALLOWED_ACTIONS:
        raise ValueError(f"Unsupported actuator action: {action}")
    return action

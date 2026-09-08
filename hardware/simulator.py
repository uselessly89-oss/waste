class Simulator:
    """Safe default actuator: logs commands and never drives hardware."""

    def command(self, action: str, pulse_s: float = 0.15):
        print(f"[SIMULATION] {action} pulse={pulse_s:.3f}s")
        return True

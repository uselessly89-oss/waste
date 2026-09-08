import time


class ArduinoSerial:
    """Serial adapter. Keep hardware disconnected until the control protocol is validated."""

    def __init__(self, port, baudrate=115200, timeout=1):
        import serial
        self.serial = serial.Serial(port, baudrate=baudrate, timeout=timeout)

    def command(self, action: str, pulse_s: float = 0.15):
        self.serial.write((action + "\n").encode("utf-8"))
        time.sleep(max(0.0, pulse_s))
        return True

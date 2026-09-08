"""Opt-in Arduino serial adapter.

Hardware is deliberately disabled unless the caller explicitly enables it.
The Arduino should implement its own emergency-stop and interlock logic.
"""
import time
from .base import validate_action


class ArduinoSafe:
    def __init__(self, port, baudrate=115200, timeout=1.0, enabled=False):
        if not enabled:
            raise RuntimeError('Hardware actuation is disabled. Set enabled=True only after physical validation.')
        if not port:
            raise ValueError('serial port is required for Arduino mode')
        import serial
        self.serial = serial.Serial(port, baudrate=baudrate, timeout=timeout)

    def command(self, action, pulse_s=0.15):
        action = validate_action(action)
        self.serial.write((action + '\n').encode('utf-8'))
        self.serial.flush()
        time.sleep(max(0.0, float(pulse_s)))
        return True

    def close(self):
        if getattr(self, 'serial', None):
            self.serial.close()

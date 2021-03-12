import subprocess
from visualswarm.contrib import control

def asebamedulla_init():
    """Establishing initial connection with the Thymio robot on a predefined interface
        Args: None
        Vars: visualswarm.control.THYMIO_DEVICE_PORT: serial port on which the robot is available for the Pi
        Returns: None
    """
    subprocess.run(['asebamedulla', f'"ser:device={control.THYMIO_DEVICE_PORT}"']
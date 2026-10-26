import numpy as np

class FluidCell:
    def __init__(self,
                 velocity: np.ndarray,
                 pressure: float,
                 volume_fraction: float) -> None:

        self.velocity = velocity
        self.pressure = pressure
        self.volume_fraction = volume_fraction
        pass
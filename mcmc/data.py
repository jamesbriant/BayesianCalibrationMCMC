import numpy as np
from dataclasses import dataclass

@dataclass(repr=False)
class Data:
    """
    """
    x_c: np.ndarray
    t: np.ndarray
    y: np.ndarray
    x_f: np.ndarray
    z: np.ndarray

    def __post_init__(self) -> None:
        self.d = np.vstack([self.z, self.y])

        self.n = self.z.shape[0]
        self.m = self.y.shape[0]

        self.p = self.x_f.shape[1]
        self.q = self.t.shape[1]
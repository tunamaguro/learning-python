import re
import numpy as np


def AND(x1: int, x2: int) -> int:
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    out = np.sum(x*w)+b
    if out > 0:
        return 1
    elif out <= 0:
        return 0

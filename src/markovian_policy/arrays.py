"""Shared type aliases."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

type FloatArray = NDArray[np.float64]
type Image = NDArray[np.uint8]  # (H, W, 3)
type Array = NDArray[Any]

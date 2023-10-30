from typing import Literal, Sequence, Tuple, Unpack

from matplotlib.axes._axes import Axes
from numpy import ndarray
from pandas import DataFrame, Series

# Generic Types
ARRAY = ndarray | Series | Sequence[int | float]

# Colormap types
COLOR = str | Tuple[float, float, float]
COLORS_SEQ = Sequence[COLOR]
COLOR_FORMAT = Literal['hex', 'rgb', 'hls']

# Plot types
BW_TYPES = int | float | Literal['silverman', 'scott']
BOUNDS = Sequence[Tuple[float | int | None, float | int | None]]
AXES = Axes
ERROR = Literal['ci', 'std', 'se']
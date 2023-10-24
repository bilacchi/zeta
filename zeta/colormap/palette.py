from colorsys import hls_to_rgb, rgb_to_hls

import numpy as np
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap

from ..typing.generics import *

__all__ = ['blend_palette']


def _color2rgb(color: COLOR, format: COLOR_FORMAT):
    if format == 'hls':
        color = hls_to_rgb(*color)
    return colors.to_rgb(color)


def blend_palette(
    colors: COLORS_SEQ,
    format: COLOR_FORMAT = 'rgb',
) -> LinearSegmentedColormap:
    """A function that generates an interpolated colormap from
    the input sequence.

    Args:
        colors: A sequence of colors
        format: The format of the input, hex, rgb, hls

    Returns:
        A colormap from the inputed colors

    Examples:

    """
    colors = [_color2rgb(color, format) for color in colors]
    name = 'blend'
    pal = LinearSegmentedColormap.from_list(name, colors)
    return pal


def dimmer(color: COLOR, p: float):
    rgb = colors.to_rgb(color)
    hls = list(rgb_to_hls(*rgb))
    hls[1] = np.clip(hls[1] * p, 0, 1)
    return hls_to_rgb(*hls)

import matplotlib as mpl
from cycler import cycler

__all__ = []

context = {
    'lines.marker': 's',
    'lines.linewidth': 1,
    'lines.markersize': 4,
    'figure.dpi': 300,
    'figure.titlesize': 10,
    'figure.labelsize': 8,
    'figure.figsize': (7.09, 3.14),
    'font.size': 8,
    'axes.prop_cycle': cycler('color', mpl.colormaps['pantone'].colors),
    'axes.grid': True,
    'grid.linestyle': 'dashed',
    'grid.alpha': 0.4,
}

mpl.rcParams.update(context)

del context, cycler, mpl

import matplotlib as mpl
from matplotlib.colors import ListedColormap

__all__ = []

PALETTES = dict(
    wong=[
        '#000000',
        '#E69F00',
        '#50B4E9',
        '#009E73',
        '#F0E442',
        '#0072B2',
        '#D55E00',
        '#CC79A7',
    ],
    pantone=[
        '#9BB7D4',
        '#C74375',
        '#BF1932',
        '#7BC4C4',
        '#E2583E',
        '#53B0AE',
        '#DECDBE',
        '#9B1B30',
        '#5A5B9F',
        '#F0C05A',
        '#45B5AA',
        '#D94F70',
        '#DD4124',
        '#009473',
        '#B163A3',
        '#955251',
        '#F7CAC9',
        '#92A8D1',
        '#88B04B',
        '#5F4B8B',
        '#FF6F61',
        '#0F4C81',
        '#939597',
        '#F5DF4D',
        '#6667AB',
        '#BB2649',
    ],
)


def register_colormap(name: str, cmap: ListedColormap) -> None:
    try:
        if name not in mpl.colormaps:
            mpl.colormaps.register(cmap, name=name)
    except AttributeError:
        mpl.cm.register_cmap(name, cmap)


for _name, _lut in PALETTES.items():

    _cmap = ListedColormap(_lut, _name)
    locals()[_name] = _cmap

    _cmap_r = ListedColormap(_lut[::-1], _name + '_r')
    locals()[_name + '_r'] = _cmap_r

    register_colormap(_name, _cmap)
    register_colormap(_name + '_r', _cmap_r)


del PALETTES, register_colormap

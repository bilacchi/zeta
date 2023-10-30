#%%
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from ..colormap.palette import dimmer
from ..typing.generics import *
from .scalers import MinMaxScaler, OneScaler
from .utils import ensure_fit
from scipy.stats import t

from numpy.typing import ArrayLike

#%%
__all__ = ['dispersion_plot']

#%%
class KDE:
    def __init__(
        self,
        *,
        bw_method: BW_TYPES = 'scott',
        scale: bool = True,
    ):

        self.bw_method = bw_method
        self.scaler = MinMaxScaler() if scale else OneScaler()
        self.isfit = False
        self.name = 'KDE'

    def fit(self, X: ArrayLike):
        self._kde = gaussian_kde(
            self.scaler.fit_transform(X), bw_method=self.bw_method
        )
        self.isfit = True
        return self

    @ensure_fit
    def __call__(self, X: ArrayLike) -> np.ndarray:
        t = self.scaler.transform(X)
        return self._kde.pdf(t)

error_func = dict(ci = lambda x, alpha: np.max(t.interval(alpha, loc=0, scale=np.std(x), df=len(x)-1)),
              std = lambda x, alpha: np.std(x),
              se = lambda x, alpha: np.std(x, ddof=1) / np.sqrt(len(x)))

#%%
def raincloud(
    x: ArrayLike,
    y: ArrayLike,
    *,
    jitter: float = 0.3,
    cmap: str = 'jet',
    ax: AXES = None,
    bounds: BOUNDS = [(None, None)],
    errors:ERROR = 'ci',
    alpha: float = 0.95
) -> AXES:
    """Dispersion + Error bar + KDE plot for continous data with multiple groups.

    Params:
        x: A n-shape array indicating the groups
        y: A nxm-shape array with the value for each group
        jitter: The amplitude of dispersion of the data
        cmap: An available matplotlib colormap name
        ax: An axes to be plot
        bounds: A list of bounds to restrict y data whithin

    Returns:
        The axes where the data was plot

    Examples:
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>>
        >>> from zeta.plot import dispersion_plot
        >>>
        >>> N = 5
        >>> X = np.arange(N)
        >>> Y = np.random.rand(N, 250)
        >>>
        >>> dispersion_plot(x=X, y=Y, cmap='pantone')
        >>> plt.show()
    """
    assert (
        len(bounds) == len(x) or len(bounds) == 1
    ), f'ybounds should be 1 or the same size to x. Got {len(bounds)}.'
    
    ybounds_ls = bounds * len(x) if len(bounds) == 1 else bounds

    if ax is None:
        ax = plt.gca()

    for x_, y_, bounds_ in zip(x, y, ybounds_ls):
        bounds_l = np.min(y_) - np.std(y_) if bounds_[0] is None else bounds_[0]
        bounds_u = np.max(y_) + np.std(y_) if bounds_[1] is None else bounds_[1]
        t = np.linspace(bounds_l, bounds_u, 100)
        kde = KDE().fit(y_)
        prob = kde(t)

        x1 = x_ - (prob / prob.max()) * jitter
        x2 = np.full_like(x1, x_)

        color = plt.get_cmap(cmap)(x_ / np.max(x))
        lighter, darker = dimmer(color, 1.05), dimmer(color, 0.7)

        ax.fill_betweenx(t, x1, x2, facecolor=lighter, alpha=0.7)
        ax.scatter(
            x_ + np.random.rand(len(y_)) * jitter,
            y_,
            alpha=0.7,
            edgecolors=lighter,
            facecolor='none',
        )
        ax.errorbar(x_, 
                    np.mean(y_), 
                    yerr=error_func[errors](y_, alpha), 
                    capsize=5, 
                    color=darker
                    )

    return ax

#%%
def paired_raincloud(x: ArrayLike,
    y: ArrayLike,
    *,
    jitter: float = 0.3,
    cmap: str = 'jet',
    ax: AXES = None,
    bounds: BOUNDS = [(None, None)],
    errors: ERROR = 'ci',
    alpha:float=.95):
    """Dispersion + Error bar + KDE plot for continous data with multiple groups.

    Params:
        x: A n-shape array indicating the groups
        y: A nxm-shape array with the value for each group
        jitter: The amplitude of dispersion of the data
        cmap: An available matplotlib colormap name
        ax: An axes to be plot
        bounds: A list of bounds to restrict y data whithin

    Returns:
        The axes where the data was plot

    Examples:
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>>
        >>> from zeta.plot import dispersion_plot
        >>>
        >>> N = 5
        >>> X = np.arange(N)
        >>> Y = np.random.rand(N, 250)
        >>>
        >>> dispersion_plot(x=X, y=Y, cmap='pantone')
        >>> plt.show()
    """
    assert (
        len(bounds) == len(x) or len(bounds) == 1
    ), f'ybounds should be 1 or the same size to x. Got {len(bounds)}.'
    
    ybounds_ls = bounds * len(x) if len(bounds) == 1 else bounds

    if ax is None:
        ax = plt.gca()

    jitter_hist = np.empty_like(y)
    for n, (x_, y_, bounds_) in enumerate(zip(x, y, ybounds_ls)):
        bounds_l = np.min(y_) - np.std(y_) if bounds_[0] is None else bounds_[0]
        bounds_u = np.max(y_) + np.std(y_) if bounds_[1] is None else bounds_[1]
        t = np.linspace(bounds_l, bounds_u, 100)
        kde = KDE().fit(y_)
        prob = kde(t)

        x1 = x_ - (prob / prob.max()) * jitter
        x2 = np.full_like(x1, x_)

        color = plt.get_cmap(cmap)(n / len(x))
        lighter, darker = dimmer(color, 1.05), dimmer(color, 0.7)

        ax.fill_betweenx(t, x1, x2, facecolor=lighter, alpha=0.7, zorder=2)
        jitter_hist[n, :] = np.random.rand(len(y_)) * jitter
        ax.scatter(
            x_ + jitter_hist[n, :],
            y_,
            alpha=0.7,
            edgecolors=lighter,
            facecolor='none',
            zorder=3
        )
        ax.errorbar(x_, 
                    np.mean(y_),
                    yerr=error_func[errors](y_, alpha),
                    capsize=5,
                    color=darker,
                    zorder=2)
    
    ax.plot((x+jitter_hist.T).T, y, linestyle='solid', color='#dfdfdf', marker='none', zorder=1, linewidth=1)
    return ax
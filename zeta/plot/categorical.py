#%%
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from ..colormap.palette import dimmer
from ..typing.generics import *
from .scalers import MinMaxScaler, OneScaler
from .utils import ensure_fit

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

    def fit(self, X: ARRAY):
        self._kde = gaussian_kde(
            self.scaler.fit_transform(X), bw_method=self.bw_method
        )
        self.isfit = True
        return self

    @ensure_fit
    def __call__(self, X: ARRAY) -> np.ndarray:
        t = self.scaler.transform(X)
        return np.exp(self._kde.logpdf(t))


#%%
def dispersion_plot(
    x: ARRAY,
    y: ARRAY,
    *,
    jitter: float = 0.3,
    cmap: str = 'jet',
    ax: AXES = None,
    ybounds: BOUNDS = [(0, 1)],
) -> AXES:
    """Dispersion + Error bar + KDE plot for continous data with multiple groups.

    Params:
        x: A n-shape array indicating the groups
        y: A nxm-shape array with the value for each group
        jitter: The amplitude of dispersion of the data
        cmap: An available matplotlib colormap name
        ax: An axes to be plot
        ybounds: A list of bounds to restrict y data whithin

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

        ![dispersion](dispersion.png){width="300" .center}
    """
    assert (
        len(ybounds) == len(x) or len(ybounds) == 1
    ), f'ybounds should be 1 or the same size to x. Got {len(ybounds)}.'
    if len(ybounds) == 1:
        ybounds *= len(x)

    if ax is None:
        ax = plt.gca()

    for x_, y_, bounds_ in zip(x, y, ybounds):
        t = np.linspace(bounds_[0], bounds_[1], 100)
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
        ax.errorbar(x_, np.mean(y_), yerr=np.std(y), capsize=5, color=darker)

    return ax


#%%
def cocurrence_plot(x, y, *, group, cmap, ax=None):
    ...

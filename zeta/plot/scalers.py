from abc import ABC, abstractmethod

import numpy as np

from ..typing.generics import *
from .utils import ensure_fit


class Scaler(ABC):
    def __init__(self, name: str):
        self.isfit = False
        self.name = name

    @abstractmethod
    def fit(self):
        ...

    @abstractmethod
    def transform(self):
        ...

    def fit_transform(self, X: ARRAY) -> ARRAY:
        return self.fit(X).transform(X)

    @abstractmethod
    def inverse_transform(self):
        ...


class MinMaxScaler(Scaler):
    def __init__(self):
        super().__init__(name='MinMaxScaler')

    def fit(self, X: ARRAY):
        self.min = np.nanmin(X)
        self.max = np.nanmax(X)
        self.scale = 1 / (self.max - self.min)
        self.x_std = X - self.min
        self.scaled = self.x_std * (self.max - self.min) + self.min

        self.isfit = True
        return self

    @ensure_fit
    def transform(self, X: ARRAY):
        return (X - self.min) * self.scale

    def inverse_transform(self, X):
        return X / self.scale + self.min


class OneScaler(Scaler):
    def __init__(self):
        super().__init__(name='OneScaler')

    def fit(self, X: ARRAY):
        self.isfit = True
        return self

    @ensure_fit
    def transform(self, X: ARRAY) -> ARRAY:
        return X

    def inverse_transform(self, X: ARRAY) -> ARRAY:
        return X

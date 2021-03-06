from abc import abstractmethod
from abc import ABC


class Manifold(ABC):
    def __init__(self, *args, **kwargs):
        pass

    @property
    def name(self):
        raise NotImplementedError

    @staticmethod
    def dim(dim):
        return dim

    def normalize(self, u):
        return u

    @abstractmethod
    def distance(self, u, v):
        """
        Distance function
        """
        raise NotImplementedError

    @abstractmethod
    def exp_map(self, p, d_p, lr=None, out=None):
        """
        Exponential map
        """
        raise NotImplementedError

    @abstractmethod
    def log_map(self, x, y):
        """
        Logarithmic map
        """
        raise NotImplementedError

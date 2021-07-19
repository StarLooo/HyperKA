from abc import abstractmethod
from abc import ABC


# modified by lxy
# change some func or property name for readability
class Manifold(ABC):
    def __init__(self, *args, **kwargs):
        pass

    # lxy: I guess this property is the name of the Manifold such as Euclidean,Spherical or Hyperbolic
    @property
    def name(self):
        raise NotImplementedError

    # lxy: I don't know the usage of this func
    @staticmethod
    def dim(dim):
        return dim

    # lxy: according to the paper:
    # @para u is a vector
    def normalize(self, u):
        return u

    # lxy: according to the paper:
    # @para u and v are vectors with the same dim in this manifold space
    @abstractmethod
    def distance(self, u, v):
        """
        Distance function
        """
        raise NotImplementedError

    # TODO: I don't know the param of this func
    @abstractmethod
    def exp_map(self, p, d_p, lr=None, out=None):
        """
        Exponential map
        """
        raise NotImplementedError

    # TODO: I don't know the param of this func
    @abstractmethod
    def log_map(self, x, y):
        """
        Logarithmic map
        """
        raise NotImplementedError

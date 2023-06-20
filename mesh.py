import math
import numpy as np

import numpy.typing as npt
from scipy.special import roots_legendre, roots_jacobi


class Mesh:
    dim: int

    nel: int
    nodes: npt.NDArray[np.float_]
    elem_size: npt.NDArray[np.float_]

    nqn_gauss: int
    param_gauss_nodes: npt.NDArray[np.float_]
    param_gauss_weights: npt.NDArray[np.float_]

    def __init__(self, nel: int = 1, nqn_gauss: int = 10, dim: int = 1):
        self.dim = dim  # for now, it is only implemented for dim=1, TODO: generalize

        self.nel = nel
        self.nodes = np.linspace(0, 1, self.nel)
        self.elem_size = np.diff(self.nodes)

        self.nqn_gauss = nqn_gauss
        self.__compute_gauss_rule()

    def __compute_gauss_rule(self):
        [nodes, weights] = roots_legendre(self.nqn_gauss)
        self.param_gauss_nodes = (nodes + 1) / 2
        self.param_gauss_weights = weights / 2


class MeshY(Mesh):
    alpha: float
    Y: float

    nqn_gauss_jacobi: int
    param_gauss_jacobi_nodes: npt.NDArray[np.float_]
    param_gauss_jacobi_weights: npt.NDArray[np.float_]

    def __init__(self, alpha: float, nel: int = 1, nqn_gauss: int = 10, nqn_gauss_jacobi: int = 10):
        Mesh.__init__(self, nel=nel, nqn_gauss=nqn_gauss, dim=1)
        self.alpha = alpha
        self.Y = math.floor(math.log(self.nel))

        self.nqn_gauss_jacobi = nqn_gauss_jacobi
        self.__compute_gauss_jacobi_rule()

    def __compute_gauss_jacobi_rule(self):
        [nodes, weights] = roots_jacobi(self.nqn_gauss_jacobi, 0, self.alpha)
        self.param_gauss_jacobi_nodes = (nodes + 1) / 2
        self.param_gauss_jacobi_weights = weights / 2

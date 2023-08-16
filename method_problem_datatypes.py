import math
import numpy as np
import numpy.typing as npt

from dataclasses import dataclass
from typing import Callable, List

from basis_functions_type import BasisFunctions1DType


@dataclass
class FEMethodData:
    degree_per_direction: npt.NDArray[np.int_]  # in space and in the extended direction
    basis_functions_type: BasisFunctions1DType
    number_of_gauss_quadrature_nodes: int
    number_of_gauss_jacobi_quadrature_nodes: int
    number_of_elements_per_direction: npt.NDArray[np.int_]  # in space and in the extended direction


@dataclass
class LinearProblemData:
    rhs_function: Callable  # right hand side of the differential problem
    dirichlet_boundaries: List[int]
    neumann_boundaries: List[int]
    dirichlet_function: Callable = lambda x: np.zeros_like(x)
    neumann_function: Callable = lambda x: np.zeros_like(x)


class DiffusionReactionProblemData(LinearProblemData):
    diffusion_coefficient: float
    reaction_coefficient: float

    def __init__(self, diffusion_coefficient: float, reaction_coefficient: float, rhs_function: Callable,
                 dirichlet_boundaries: List[int], neumann_boundaries: List[int],
                 dirichlet_function: Callable = lambda x: np.zeros_like(x),
                 neumann_function: Callable = lambda x: np.zeros_like(x)):
        self.diffusion_coefficient = diffusion_coefficient
        self.reaction_coefficient = reaction_coefficient
        super().__init__(rhs_function=rhs_function, dirichlet_boundaries=dirichlet_boundaries,
                         neumann_boundaries=neumann_boundaries, dirichlet_function=dirichlet_function,
                         neumann_function=neumann_function)


class LaplacianProblemData(LinearProblemData):
    s: float = 0.5  # fractional power, different from 0.5 to consider a spectral fractional laplacian operator
    alpha: float
    ds: float

    def __init__(self, rhs_function: Callable, dirichlet_boundaries: List[int], neumann_boundaries: List[int],
                 dirichlet_function: Callable = lambda x: np.zeros_like(x),
                 neumann_function: Callable = lambda x: np.zeros_like(x), s: float = 0.5):
        self.s = s
        self.alpha = 1 - 2 * self.s
        self.ds = (2 ** self.alpha) * math.gamma(1 - self.s) / math.gamma(self.s)

        super().__init__(rhs_function=rhs_function, dirichlet_boundaries=dirichlet_boundaries,
                         neumann_boundaries=neumann_boundaries, dirichlet_function=dirichlet_function,
                         neumann_function=neumann_function)

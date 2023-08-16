import numpy as np
import numpy.typing as npt

from abc import ABC, abstractmethod
from typing import List
from scipy.special import roots_legendre, roots_jacobi


class QuadratureFormula(ABC):
    number_of_quadrature_nodes: int
    reference_quadrature_nodes: npt.NDArray[np.float_]
    reference_quadrature_weights: npt.NDArray[np.float_]

    def __init__(self, number_of_quadrature_nodes: int = 10):
        self.number_of_quadrature_nodes = number_of_quadrature_nodes
        self.compute_quadrature_rule()

    @abstractmethod
    def compute_quadrature_rule(self):
        raise NotImplementedError("Not implemented")


class QuadratureFormulaTensorProduct(QuadratureFormula, ABC):
    dimension: int

    def __init__(self, number_of_quadrature_nodes: int = 10):
        super().__init__(number_of_quadrature_nodes=number_of_quadrature_nodes)


class QuadratureFormulaND(QuadratureFormulaTensorProduct):
    quadrature_formula_per_direction: List[QuadratureFormula]

    def __init__(self, quadrature_formula_per_direction: List[QuadratureFormula]):
        self.dimension = len(quadrature_formula_per_direction)
        self.quadrature_formula_per_direction = quadrature_formula_per_direction
        number_of_quadrature_nodes = np.array([formula_1d.number_of_quadrature_nodes
                                               for formula_1d in quadrature_formula_per_direction]).prod()
        super().__init__(number_of_quadrature_nodes=number_of_quadrature_nodes)

    def compute_quadrature_rule(self):
        coordinates = [formula_1d.reference_quadrature_nodes for formula_1d in self.quadrature_formula_per_direction]
        grid_coordinates = np.meshgrid(*coordinates)
        self.reference_quadrature_nodes = np.vstack([single_coordinate.ravel()
                                                     for single_coordinate in grid_coordinates])  # shape = dim x npts

        weights_per_direction = [formula_1d.reference_quadrature_weights
                                 for formula_1d in self.quadrature_formula_per_direction]
        grid_weights_per_direction = np.meshgrid(*weights_per_direction)
        self.reference_quadrature_weights = np.prod([weights_1d.ravel()
                                                     for weights_1d in grid_weights_per_direction], axis=0)


class QuadratureFormula1D(QuadratureFormulaTensorProduct, ABC):
    dimension = 1

    def __init__(self, number_of_quadrature_nodes: int = 10):
        self.dimension = 1
        super().__init__(number_of_quadrature_nodes=number_of_quadrature_nodes)


class GaussQuadrature(QuadratureFormula1D):
    def __init__(self, number_of_quadrature_nodes: int = 10):
        super().__init__(number_of_quadrature_nodes=number_of_quadrature_nodes)

    def compute_quadrature_rule(self):
        [nodes, weights] = roots_legendre(self.number_of_quadrature_nodes)
        nodes = nodes[np.newaxis, :]
        [self.reference_quadrature_nodes, self.reference_quadrature_weights] = rescale_to_unit_interval(nodes=nodes,
                                                                                                        weights=weights)


class GaussJacobiQuadrature(QuadratureFormula1D):
    alpha: float
    beta: float

    def __init__(self, alpha: float, beta: float, number_of_quadrature_nodes: int = 10):
        self.alpha = alpha
        self.beta = beta
        super().__init__(number_of_quadrature_nodes=number_of_quadrature_nodes)

        self.dimension = 1
        self.quadrature_formula_per_direction = [self]

    def compute_quadrature_rule(self):
        [nodes, weights] = roots_jacobi(self.number_of_quadrature_nodes, self.alpha, self.beta)
        nodes = nodes[np.newaxis, :]
        [self.reference_quadrature_nodes, self.reference_quadrature_weights] = \
            rescale_to_unit_interval(nodes=nodes, weights=weights, alpha=self.alpha, beta=self.beta)


def rescale_to_unit_interval(nodes, weights, alpha: float = 0, beta: float = 0,
                             initial_interval: npt.NDArray[np.float_] = np.array([-1., 1.])):
    interval_length = initial_interval[1] - initial_interval[0]
    rescaled_nodes = (nodes - initial_interval[0]) / interval_length
    rescaled_weights = weights / (interval_length ** (1 + alpha + beta))
    return [rescaled_nodes, rescaled_weights]

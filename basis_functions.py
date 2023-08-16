import math
import numpy as np
import numpy.typing as npt

from abc import ABC, abstractmethod
from typing import List

from function_evaluation import FunctionEvaluation


class BasisFunctions(ABC):
    @abstractmethod
    def evaluate(self, evaluation_points: npt.NDArray[np.float_], derivatives_up_to: int = 0) -> FunctionEvaluation:
        return NotImplemented


class TensorProductBasisFunctions(BasisFunctions, ABC):
    dimension: int


class BasisFunctions1D(TensorProductBasisFunctions, ABC):
    degree: int

    def __init__(self, degree: int = 1):
        self.dimension = 1
        self.degree = degree


class TensorProductBasisFunctionsND(TensorProductBasisFunctions, ABC):
    degree_per_direction: npt.NDArray[np.int_] = None
    basis_functions_1d: List[BasisFunctions1D]

    def __init__(self, basis_functions_1d: List[BasisFunctions1D]):
        self.dimension = len(basis_functions_1d)
        self.basis_functions_1d = basis_functions_1d
        if self.degree_per_direction is None:
            self.set_degree_per_direction()

    def set_degree_per_direction(self):
        self.degree_per_direction = np.array([function_1d.degree for function_1d in self.basis_functions_1d])

    def evaluate(self, evaluation_points: npt.NDArray[np.float_], derivatives_up_to: int = 0) -> FunctionEvaluation:
        evaluations_1d = [function_1d.evaluate(derivatives_up_to=derivatives_up_to, evaluation_points=evaluation_points)
                          for function_1d in self.basis_functions_1d]
        component_wise_evaluation = FunctionEvaluation.from_evaluations_1d(evaluations_1d)

        tensor_product_evaluation = FunctionEvaluation(value=np.prod(component_wise_evaluation.value, axis=0))
        if derivatives_up_to > 0:
            tensor_product_evaluation.gradient = self.evaluate_gradient(component_wise_evaluation)
            if derivatives_up_to > 1:
                tensor_product_evaluation.hessian = self.evaluate_hessian(component_wise_evaluation)

        return tensor_product_evaluation

    def evaluate_gradient(self, component_wise_evaluation: FunctionEvaluation) -> npt.NDArray[np.float_]:
        # return shape = dimension x number_of_points x (max(degree) + 1)
        gradient = component_wise_evaluation.gradient

        for direction in range(self.dimension):
            function_values_in_other_directions = np.delete(component_wise_evaluation.value, direction, axis=0)
            gradient[direction, :, :] *= np.prod(function_values_in_other_directions, axis=0)

        return gradient

    def evaluate_hessian(self, component_wise_evaluation: FunctionEvaluation) -> npt.NDArray[np.float_]:
        # return shape = dimension x dimension x number_of_points x (max(degree) + 1)
        hessian = np.empty((self.dimension, *component_wise_evaluation.value.shape))
        for direction in range(self.dimension):
            function_values_in_other_directions = np.delete(component_wise_evaluation.value, direction, axis=0)
            hessian[direction, direction, :, :] = component_wise_evaluation.hessian[direction, :, :]
            hessian[direction, direction, :, :] *= np.prod(function_values_in_other_directions, axis=0)

            other_directions = np.linspace(direction + 1, self.dimension)
            for direction2 in other_directions:
                hessian[direction, direction2, :, :] = component_wise_evaluation.gradient[direction, :, :] \
                                                       * component_wise_evaluation.gradient[direction2, :, :]
                if self.dimension > 2:
                    function_evaluations_in_remaining_direction = np.delete(function_values_in_other_directions,
                                                                            direction2, axis=0)
                    hessian[direction, direction2, :, :] *= function_evaluations_in_remaining_direction

                hessian[direction2, direction, :, :] = hessian[direction, direction2, :, :]

        return hessian


class IntegratedLegendrePolynomials(BasisFunctions1D):
    def evaluate(self, evaluation_points: npt.NDArray[np.float_], derivatives_up_to: int = 0) -> FunctionEvaluation:
        if len(evaluation_points.shape) > 1:
            evaluation_points = evaluation_points[0]
        legendre_polynomials_evaluation = self.evaluate_legendre_polynomials(evaluation_points)

        evaluations = self.evaluate_function(evaluation_points, legendre_polynomials_evaluation)
        if derivatives_up_to > 0:
            evaluations.gradient = self.evaluate_derivative(evaluation_points, legendre_polynomials_evaluation)
            if derivatives_up_to > 1:
                evaluations.hessian = self.evaluate_derivative2(evaluation_points, legendre_polynomials_evaluation)

        return evaluations

    def evaluate_legendre_polynomials(self, evaluation_points: npt.NDArray[np.float_]) -> FunctionEvaluation:
        number_of_points = len(evaluation_points)
        maximal_degree = self.degree + 1
        legendre_polynomials_values = np.empty((number_of_points, maximal_degree))

        legendre_polynomials_values[:, 0] = np.ones(number_of_points)
        legendre_polynomials_values[:, 1] = 2 * evaluation_points - 1
        for degree in np.arange(2, maximal_degree):
            legendre_polynomials_values[:, degree] = ((2 * degree - 1) / degree) * (2 * evaluation_points - 1) \
                                                     * legendre_polynomials_values[:, degree - 1] \
                                                     - ((degree - 1) / degree) \
                                                     * legendre_polynomials_values[:, degree - 2]
        return FunctionEvaluation(value=legendre_polynomials_values)

    def evaluate_function(self, evaluation_points: npt.NDArray[np.float_],
                          legendre_polynomials_evaluation: FunctionEvaluation) -> FunctionEvaluation:
        number_of_points = len(evaluation_points)
        maximal_degree = self.degree + 1
        integrated_legendre_polynomials_values = np.empty((number_of_points, maximal_degree))

        integrated_legendre_polynomials_values[:, 0] = 1 - evaluation_points
        integrated_legendre_polynomials_values[:, 1] = evaluation_points
        for degree in np.arange(2, maximal_degree):
            integrated_legendre_polynomials_values[:, degree] = (1 / math.sqrt(4 * degree - 2)) \
                                                                * (legendre_polynomials_evaluation[:, degree]
                                                                   - legendre_polynomials_evaluation[:, degree - 2])
        return FunctionEvaluation(value=integrated_legendre_polynomials_values)

    def evaluate_derivative(self, evaluation_points: npt.NDArray[np.float_],
                            legendre_polynomials_evaluation: FunctionEvaluation) -> npt.NDArray[np.float_]:
        number_of_points = len(evaluation_points)
        maximal_degree = self.degree + 1
        integrated_legendre_derivative = np.empty((number_of_points, maximal_degree))

        integrated_legendre_derivative[:, 0] = -np.ones_like(evaluation_points)
        integrated_legendre_derivative[:, 1] = np.ones_like(evaluation_points)
        for degree in np.arange(2, maximal_degree):
            integrated_legendre_derivative[:, degree] = 2 * math.sqrt(degree - 1 / 2) \
                                                         * legendre_polynomials_evaluation[:, degree - 1]
        return integrated_legendre_derivative

    def evaluate_derivative2(self, evaluation_points: npt.NDArray[np.float_],
                             legendre_polynomials_evaluation: FunctionEvaluation) -> npt.NDArray[np.float_]:
        number_of_points = len(evaluation_points)
        maximal_degree = self.degree + 1
        integrated_legendre_derivative2 = np.empty((number_of_points, maximal_degree))

        integrated_legendre_derivative2[:, 0] = np.zeros_like(evaluation_points)
        integrated_legendre_derivative2[:, 1] = np.zeros_like(evaluation_points)
        for degree in np.arange(2, maximal_degree):
            integrated_legendre_derivative2[:, degree] = 2 * math.sqrt(degree - 1 / 2) * (degree - 1) \
                                                         / (2 * evaluation_points ** 2 - 2 * evaluation_points) \
                                                         * ((2 * evaluation_points - 1)
                                                            * legendre_polynomials_evaluation[:, degree - 1]
                                                            - legendre_polynomials_evaluation[:, degree - 2])
        return integrated_legendre_derivative2


class TensorProductIntegratedLegendrePolynomials(TensorProductBasisFunctionsND):
    basis_functions_1d: List[IntegratedLegendrePolynomials]

    def __init__(self, degree_per_direction: npt.NDArray[np.int_]):
        basis_functions_1d = [IntegratedLegendrePolynomials(degree) for degree in degree_per_direction]
        self.degree_per_direction = degree_per_direction
        super().__init__(basis_functions_1d=basis_functions_1d)

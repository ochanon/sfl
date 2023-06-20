import math
import numpy as np

import numpy.typing as npt
from abc import ABC, abstractmethod
# from scipy.special import eval_legendre --> TODO: can I use this function wisely?


class BasisFunctions(ABC):
    deg: int
    derivatives: tuple[bool, bool, bool]

    def __init__(self, deg: int = 1, derivatives: tuple[bool, bool, bool] = (True, False, False)):
        self.deg = deg
        self.derivatives = derivatives

    def evaluate(self, points: npt.NDArray[np.float_]) -> dict:
        preliminary_eval = self.preliminary_eval(points)
        basis_functions = {"fun": None, "der": None, "hess": None}

        if self.derivatives[0]:
            basis_functions["fun"] = self.eval_function(points, preliminary_eval)
        if self.derivatives[1]:
            basis_functions["der"] = self.eval_derivative(points, preliminary_eval)
        if self.derivatives[2]:
            basis_functions["hess"] = self.eval_hessian(points, preliminary_eval)

        return basis_functions

    @abstractmethod
    def preliminary_eval(self, points: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        return NotImplemented

    @abstractmethod
    def eval_function(self, points: npt.NDArray[np.float_],
                      preliminary_eval: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        return NotImplemented

    @abstractmethod
    def eval_derivative(self, points: npt.NDArray[np.float_],
                        preliminary_eval: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        return NotImplemented

    @abstractmethod
    def eval_hessian(self, points: npt.NDArray[np.float_],
                     preliminary_eval: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        return NotImplemented


class IntegratedLegendrePolynomials(BasisFunctions):
    def preliminary_eval(self, points: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        npts = len(points)
        legendre_poly = np.empty((npts, self.deg + 1))

        legendre_poly[:, 0] = np.ones(npts)
        legendre_poly[:, 1] = 2 * points - 1
        for ideg in np.arange(2, self.deg + 1):
            legendre_poly[:, ideg] = ((2 * ideg - 1) / ideg) * (2 * points - 1) * legendre_poly[:, ideg - 1]\
                                - ((ideg - 1) / ideg) * legendre_poly[:, ideg - 2]

        return legendre_poly

    def eval_function(self, points: npt.NDArray[np.float_],
                      preliminary_eval: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        legendre_poly = preliminary_eval
        integrated_legendre_fun = np.empty((len(points), self.deg + 1))

        integrated_legendre_fun[:, 0] = 1 - points
        integrated_legendre_fun[:, 1] = points
        for ideg in np.arange(2, self.deg + 1):
            integrated_legendre_fun[:, ideg] = (1 / math.sqrt(4 * ideg - 2)) \
                                               * (legendre_poly[:, ideg] - legendre_poly[:, ideg - 2])

        return integrated_legendre_fun

    def eval_derivative(self, points: npt.NDArray[np.float_],
                        preliminary_eval: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        legendre_poly = preliminary_eval
        integrated_legendre_der = np.empty((len(points), self.deg + 1))

        integrated_legendre_der[:, 0] = -np.ones_like(points)
        integrated_legendre_der[:, 1] = np.ones_like(points)
        for ideg in np.arange(2, self.deg + 1):
            integrated_legendre_der[:, ideg] = 2 * math.sqrt(ideg - 1 / 2) * legendre_poly[:, ideg - 1]

        return integrated_legendre_der

    def eval_hessian(self, points: npt.NDArray[np.float_],
                     preliminary_eval: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        legendre_poly = preliminary_eval
        integrated_legendre_hess = np.empty((len(points), self.deg + 1))

        integrated_legendre_hess[:, 0] = np.zeros_like(points)
        integrated_legendre_hess[:, 1] = np.zeros_like(points)
        for ideg in np.arange(2, self.deg + 1):
            integrated_legendre_hess[:, ideg] = 2 * math.sqrt(ideg - 1 / 2) \
                                                * (ideg - 1) / (2 * points ** 2 - 2 * points) \
                                                * ((2 * points - 1) * legendre_poly[:, ideg - 1]
                                                   - legendre_poly[:, ideg - 2])
        return integrated_legendre_hess

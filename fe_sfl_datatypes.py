import math

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class SFLProblemData:
    s: float  # fractional power
    rhs_f: Callable  # right hand side f of the differential problem
    alpha: float = field(init=False)
    ds: float = field(init=False)

    def __post_init__(self):
        self.alpha = 1 - 2 * self.s
        self.ds = (2 ** self.alpha) * math.gamma(1 - self.s) / math.gamma(self.s)


@dataclass
class FEMethodData:
    deg: int  # polynomial degree
    shape_fun_type: str  # type of basis functions
    nqn_gauss: int  # number of Gauss quadrature nodes
    nqn_gauss_jacobi: int  # number of Gauss-Jacobi quadrature nodes
    nel_x: int  # number of elements per space direction -- TODO: generalize to npt.NDArray[np.int_]
    nel_y: int  # number of elements in the extended direction

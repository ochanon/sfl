import numpy as np
import numpy.typing as npt

from dataclasses import dataclass

from helpers import to_array_with_zero_padding


@dataclass
class FunctionEvaluation:
    value: npt.NDArray[np.float_]
    gradient: npt.NDArray[np.float_] = None
    hessian: npt.NDArray[np.float_] = None

    @classmethod
    def from_evaluations_1d(cls, evaluations_1d: list):
        return cls(
            value=to_array_with_zero_padding([eval_1d.value for eval_1d in evaluations_1d]),
            gradient=to_array_with_zero_padding([eval_1d.gradient if eval_1d.gradient is not None else np.array([])
                                                 for eval_1d in evaluations_1d]),
            hessian=to_array_with_zero_padding([eval_1d.hessian if eval_1d.hessian is not None else np.array([])
                                                for eval_1d in evaluations_1d]),
        )

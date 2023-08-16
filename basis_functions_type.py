from enum import Enum


class BasisFunctionsType(Enum):
    pass


class BasisFunctions1DType(BasisFunctionsType):
    INTEGRATED_LEGENDRE_POLYNOMIALS = "IntegratedLegendrePolynomials"


class BasisFunctionsNDType(BasisFunctionsType):
    TP_INTEGRATED_LEGENDRE_POLYNOMIALS = "TensorProductIntegratedLegendrePolynomials"


TENSOR_PRODUCT_MAPPING = {
    BasisFunctions1DType.INTEGRATED_LEGENDRE_POLYNOMIALS: BasisFunctionsNDType.TP_INTEGRATED_LEGENDRE_POLYNOMIALS
}


def to_tensor_product_type(basis_functions_1d_type: BasisFunctions1DType) -> BasisFunctionsNDType:
    return TENSOR_PRODUCT_MAPPING.get(basis_functions_1d_type, None)

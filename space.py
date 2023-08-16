import numpy as np
import numpy.typing as npt

from typing import List

from basis_functions import IntegratedLegendrePolynomials, TensorProductIntegratedLegendrePolynomials
from basis_functions_type import BasisFunctionsType, BasisFunctions1DType, BasisFunctionsNDType, to_tensor_product_type
from function_evaluation import FunctionEvaluation
from helpers import are_all_equal
from mesh import Mesh, TensorProductMesh, Mesh1D, TensorProductMeshND, MeshY, ExtendedTensorProductMesh


class Space:
    mesh: Mesh
    number_of_dofs: int
    number_of_dofs_in_element: int

    basis_functions_type: BasisFunctionsType
    reference_basis_functions: FunctionEvaluation
    local2global_dofs: npt.NDArray[np.int_]

    # nsh_max (max per element), nsh_dir (max uni-variate per element),
    # boundary (space of traces), ncomp (scalar, vectorial functions, etc)
    # deg_per_el: npt.NDArray[np.float_]


class TensorProductSpace(Space):
    mesh: TensorProductMesh


class Space1D(TensorProductSpace):
    mesh: Mesh1D
    degree: int = 0

    basis_functions_type: BasisFunctions1DType

    def __init__(self, mesh: Mesh1D, degree: int, derivatives_up_to: int = 0,
                 basis_functions_type: BasisFunctions1DType = BasisFunctions1DType.INTEGRATED_LEGENDRE_POLYNOMIALS):
        self.mesh = mesh
        self.degree = degree
        self.number_of_dofs = mesh.number_of_elements * degree + 1
        self.number_of_dofs_in_element = degree + 1

        if basis_functions_type not in BasisFunctions1DType:
            raise ValueError(
                f"The chosen type of basis function '{basis_functions_type}' has not been implemented. "
                f"Available types: {', '.join([possible_type.value for possible_type in BasisFunctions1DType])}.")
        self.basis_functions_type = basis_functions_type
        self.set_reference_basis_functions(derivatives_up_to=derivatives_up_to)
        self.set_local2global_dofs()

    def set_reference_basis_functions(self, derivatives_up_to: int = 0):
        basis_functions_class = globals()[self.basis_functions_type.value]
        basis_functions = basis_functions_class(degree=self.degree)

        self.reference_basis_functions = basis_functions.evaluate(
            evaluation_points=self.mesh.quadrature_formula.reference_quadrature_nodes,
            derivatives_up_to=derivatives_up_to
        )

    def set_local2global_dofs(self):  # TODO this depends on the basis_functions_type !!
        local2global_dofs = np.empty((self.mesh.number_of_elements, self.number_of_dofs_in_element), dtype=int)
        global_dof_index = 0

        for element_id in range(self.mesh.number_of_elements):
            local2global_dofs[element_id, :] = np.arange(global_dof_index, global_dof_index
                                                         + self.number_of_dofs_in_element)
            global_dof_index = global_dof_index + self.degree

        self.local2global_dofs = local2global_dofs


class TensorProductSpaceND(TensorProductSpace):
    mesh: TensorProductMeshND
    basis_functions_type: BasisFunctionsNDType

    space_per_direction: List[Space1D]
    degree_per_direction: npt.NDArray[np.int_]
    number_of_dofs_per_direction: npt.NDArray[np.int_] = np.array([])
    number_of_dofs_per_direction_in_element: npt.NDArray[np.int_]

    def __init__(self, space_per_direction: List[Space1D], tensor_product_mesh: TensorProductMeshND = None,
                 derivatives_up_to: int = 0):
        self.space_per_direction = space_per_direction
        self.set_degree_per_direction()
        if len(self.number_of_dofs_per_direction) == 0:
            self.set_number_of_dofs_per_direction()
        self.set_number_of_dofs_per_direction_per_element()
        self.number_of_dofs_in_element = self.number_of_dofs_per_direction_in_element.prod()

        if tensor_product_mesh is None:
            self.set_tensor_product_mesh()
        else:
            self.mesh = tensor_product_mesh
        self.number_of_dofs = self.number_of_dofs_per_direction.prod()

        if not are_all_equal([space_1d.basis_functions_type for space_1d in space_per_direction]):
            raise ValueError("The 'basis_fun_type' of all spaces in a TensorProductSpace must be the same.")
        self.basis_functions_type = to_tensor_product_type(space_per_direction[0].basis_functions_type)
        self.set_reference_basis_functions(derivatives_up_to=derivatives_up_to)
        self.set_local2global_dofs()

    def set_degree_per_direction(self):
        self.degree_per_direction = np.array([space_1d.degree for space_1d in self.space_per_direction])

    def set_number_of_dofs_per_direction(self):
        self.number_of_dofs_per_direction = np.array([space_1d.number_of_dofs
                                                      for space_1d in self.space_per_direction])

    def set_number_of_dofs_per_direction_per_element(self):
        self.number_of_dofs_per_direction_in_element = np.array([space_1d.number_of_dofs_in_element
                                                                 for space_1d in self.space_per_direction])

    def set_tensor_product_mesh(self):
        mesh_per_direction = [space_1d.mesh for space_1d in self.space_per_direction]
        self.mesh = TensorProductMeshND(mesh_per_direction=mesh_per_direction)

    def set_reference_basis_functions(self, derivatives_up_to: int = 0):
        basis_functions_class = globals()[self.basis_functions_type.value]
        basis_functions = basis_functions_class(degree_per_direction=self.degree_per_direction)

        self.reference_basis_functions = basis_functions.evaluate(
            evaluation_points=self.mesh.quadrature_formula.reference_quadrature_nodes,
            derivatives_up_to=derivatives_up_to
        )

    """def set_local2global_dofs(self):
        local2global_dofs = np.empty((self.mesh.nel, self.number_of_dofs_in_element), dtype=int)

        elements_ids_per_direction = np.array(np.unravel_index(np.arange(self.mesh.nel), self.mesh.nel_dir)).T
        number_of_dofs_per_direction_cumprod = np.cumprod(np.append(1, self.number_of_dofs_per_direction))
        for element_global_id in range(self.mesh.nel):
            element_ids = elements_ids_per_direction[element_global_id]
            dofs_dir_of_el = [self.space_per_direction[direction].local2global_dofs[element_ids[direction], :]
                              for direction in range(self.mesh.dim)]
            global_dofs_per_dir = [dofs_dir_of_el[direction] * number_of_dofs_per_direction_cumprod[direction]
                                   for direction in range(self.mesh.dim)]
            l2g = global_dofs_per_dir[0]
            if self.mesh.dim > 1:
                l2g = l2g[:, np.newaxis] + global_dofs_per_dir[1].T
                l2g = np.ravel(l2g)
                if self.mesh.dim > 2:
                    l2g = l2g[:, np.newaxis] + global_dofs_per_dir[2].T
                    l2g = np.ravel(l2g)
            local2global_dofs[element_global_id, :] = np.sort(l2g)

        self.local2global = local2global_dofs"""

    def set_local2global_dofs(self):
        local2global_dofs = np.empty((self.mesh.number_of_elements, self.number_of_dofs_in_element), dtype=int)

        grid_ids_of_all_elements = np.array(np.unravel_index(np.arange(self.mesh.number_of_elements),
                                                             self.mesh.number_of_elements_per_direction)).T
        number_of_dofs_per_direction_cumprod = np.cumprod(np.append(1, self.number_of_dofs_per_direction))

        for global_element_id in range(self.mesh.number_of_elements):
            element_grid_ids = grid_ids_of_all_elements[global_element_id]
            global_grid_dofs_in_element = \
                [self.space_per_direction[direction].local2global_dofs[element_grid_ids[direction], :]
                 for direction in range(self.mesh.dimension)]

            translated_global_grid_dofs_in_element = [global_grid_dofs_in_element[direction]
                                                      * number_of_dofs_per_direction_cumprod[direction]
                                                      for direction in range(self.mesh.dimension)]

            local2global_in_element = translated_global_grid_dofs_in_element[0]
            for direction in range(1, self.mesh.dimension):
                local2global_in_element = np.add.outer(local2global_in_element,
                                                       translated_global_grid_dofs_in_element[direction]).ravel()
            local2global_dofs[global_element_id, :] = np.sort(local2global_in_element)

        self.local2global_dofs = local2global_dofs


class SpaceY(Space1D):
    mesh: MeshY
    reference_basis_functions_for_element0: FunctionEvaluation

    def __init__(self, mesh: MeshY, degree: int, derivatives_up_to: int = 0,
                 basis_functions_type: BasisFunctions1DType = BasisFunctions1DType.INTEGRATED_LEGENDRE_POLYNOMIALS):
        super().__init__(mesh=mesh, degree=degree, derivatives_up_to=derivatives_up_to,
                         basis_functions_type=basis_functions_type)
        self.set_reference_basis_functions_for_element0(derivatives_up_to=derivatives_up_to)

    def set_reference_basis_functions_for_element0(self, derivatives_up_to: int = 0):
        basis_functions_class = globals()[self.basis_functions_type.value]
        basis_functions = basis_functions_class(degree=self.degree)

        self.reference_basis_functions_for_element0 = basis_functions.evaluate(
            evaluation_points=self.mesh.quadrature_formula_element0.reference_quadrature_nodes,
            derivatives_up_to=derivatives_up_to
        )


class ExtendedTensorProductSpace(TensorProductSpaceND):
    space_x: TensorProductSpace
    space_y: Space1D

    mesh: ExtendedTensorProductMesh

    def __init__(self, space_x: TensorProductSpace, space_y: Space1D, derivatives_up_to: int = 0):
        space_per_direction = []
        if isinstance(space_x, Space1D):
            space_per_direction = [space_x, space_y]
        elif isinstance(space_x, TensorProductSpaceND):
            space_per_direction = space_x.space_per_direction + [space_y]
        super().__init__(space_per_direction=space_per_direction, derivatives_up_to=derivatives_up_to)

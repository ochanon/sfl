import numpy as np
import numpy.typing as npt

from typing import Callable, List

from space import Space1D, SpaceY
from mesh import Mesh1D, MeshY


def f_v_vector(function_f: Callable, space_v: Space1D, mesh: Mesh1D, elements_to_assemble: range = None,
               boundaries: List[int] = None) -> npt.NDArray[np.float_]:
    vector = np.zeros((space_v.number_of_dofs,))
    if elements_to_assemble is None:
        elements_to_assemble = range(mesh.number_of_elements)

    elements_diameters = mesh.diameters_of_elements[elements_to_assemble, np.newaxis]
    f_on_quadrature_nodes = function_f(mesh.nodes[:-1, np.newaxis] + elements_diameters
                                       * mesh.quadrature_formula.reference_quadrature_nodes)
    f_on_quadrature_nodes = f_on_quadrature_nodes[:, :, np.newaxis]

    reference_v = space_v.reference_basis_functions.value[np.newaxis, :, :]
    reference_quadrature_weights = mesh.quadrature_formula.reference_quadrature_weights[np.newaxis, :, np.newaxis]
    reference_integral = np.sum(reference_quadrature_weights * f_on_quadrature_nodes * reference_v, axis=1)
    jacobian_determinant = elements_diameters

    for element_id in elements_to_assemble:
        vector[np.ix_(space_v.local2global_dofs[element_id, :])] \
            += reference_integral[element_id, :] * jacobian_determinant[element_id]

    return vector


def du_dv_matrix(space_u: Space1D, space_v: Space1D, mesh: Mesh1D, elements_to_assemble: range = None) \
        -> npt.NDArray[np.float_]:
    matrix = np.zeros((space_v.number_of_dofs, space_u.number_of_dofs))
    if elements_to_assemble is None:
        elements_to_assemble = range(mesh.number_of_elements)

    elements_diameters = mesh.diameters_of_elements[elements_to_assemble, np.newaxis, np.newaxis, np.newaxis]
    reference_du_dv = space_v.reference_basis_functions.gradient[np.newaxis, :, :, np.newaxis] / elements_diameters \
                      * space_u.reference_basis_functions.gradient[np.newaxis, :, np.newaxis, :] / elements_diameters

    assemble_matrix(reference_operation=reference_du_dv, elements_to_assemble=elements_to_assemble,
                    space_u=space_u, space_v=space_v, mesh=mesh, matrix=matrix)

    return matrix


def weighted_du_dv_matrix(space_u: SpaceY, space_v: SpaceY, mesh: MeshY, elements_to_assemble: range = None) \
        -> npt.NDArray[np.float_]:
    matrix = np.zeros((space_v.number_of_dofs, space_u.number_of_dofs))
    if elements_to_assemble is None:
        elements_to_assemble = range(mesh.number_of_elements)

    if 0 in elements_to_assemble:
        assemble_weighted_du_dv_element0(space_u=space_u, space_v=space_v, mesh=mesh, matrix=matrix)
        elements_to_assemble = elements_to_assemble[1:]

    alpha = mesh.quadrature_formula_element0.beta
    elements_diameters = mesh.diameters_of_elements[elements_to_assemble, np.newaxis]

    quadrature_nodes_in_each_element = elements_diameters * mesh.quadrature_formula.reference_quadrature_nodes \
                                       + mesh.nodes[elements_to_assemble, np.newaxis]
    y_alpha = quadrature_nodes_in_each_element ** alpha
    y_alpha = y_alpha[:, :, np.newaxis, np.newaxis]

    elements_diameters = elements_diameters[:, :, np.newaxis, np.newaxis]
    reference_du_dv = space_v.reference_basis_functions.gradient[np.newaxis, :, :, np.newaxis] / elements_diameters \
                      * space_u.reference_basis_functions.gradient[np.newaxis, :, np.newaxis, :] / elements_diameters
    reference_y_alpha_du_dv = reference_du_dv * y_alpha

    assemble_matrix(reference_operation=reference_y_alpha_du_dv, elements_to_assemble=elements_to_assemble,
                    space_u=space_u, space_v=space_v, mesh=mesh, matrix=matrix)

    return matrix


def assemble_weighted_du_dv_element0(space_u: SpaceY, space_v: SpaceY, mesh: MeshY, matrix: npt.NDArray[np.float_]):
    diameter_element0 = mesh.diameters_of_elements[0]
    reference_quadrature_weights_element0 = mesh.quadrature_formula_element0.reference_quadrature_weights[:, np.newaxis,
                                            np.newaxis]
    rescaled_weighted_measure = diameter_element0 ** mesh.quadrature_formula_element0.beta
    mapped_reference_gradients_u = space_u.reference_basis_functions_for_element0.gradient / diameter_element0
    mapped_reference_gradients_v = space_v.reference_basis_functions_for_element0.gradient / diameter_element0
    reference_du_dv_element0 = mapped_reference_gradients_v[:, :, np.newaxis] \
                               * mapped_reference_gradients_u[:, np.newaxis, :] * rescaled_weighted_measure
    reference_integral_du_dv_element0 = np.sum(reference_quadrature_weights_element0 * reference_du_dv_element0, axis=0)

    jacobian_determinant = diameter_element0
    matrix[np.ix_(space_v.local2global_dofs[0, :], space_u.local2global_dofs[0, :])] \
        += reference_integral_du_dv_element0 * jacobian_determinant


def u_v_matrix(space_u: Space1D, space_v: Space1D, mesh: Mesh1D, elements_to_assemble: range = None) \
        -> npt.NDArray[np.float_]:
    matrix = np.zeros((space_v.number_of_dofs, space_u.number_of_dofs))
    if elements_to_assemble is None:
        elements_to_assemble = range(mesh.number_of_elements)

    reference_u_v = space_v.reference_basis_functions.value[np.newaxis, :, :, np.newaxis] \
                    * space_u.reference_basis_functions.value[np.newaxis, :, np.newaxis, :] \
                    * np.ones((len(elements_to_assemble), 1,
                               space_v.number_of_dofs_in_element, space_u.number_of_dofs_in_element))

    assemble_matrix(reference_operation=reference_u_v, elements_to_assemble=elements_to_assemble,
                    space_u=space_u, space_v=space_v, mesh=mesh, matrix=matrix)

    return matrix


def weighted_u_v_matrix(space_u: SpaceY, space_v: SpaceY, mesh: MeshY, elements_to_assemble: range = None) \
        -> npt.NDArray[np.float_]:
    matrix = np.zeros((space_v.number_of_dofs, space_u.number_of_dofs))
    if elements_to_assemble is None:
        elements_to_assemble = range(mesh.number_of_elements)

    if 0 in elements_to_assemble:
        assemble_weighted_u_v_element0(space_u=space_u, space_v=space_v, mesh=mesh, matrix=matrix)
        elements_to_assemble = elements_to_assemble[1:]

    alpha = mesh.quadrature_formula_element0.beta
    elements_diameters = mesh.diameters_of_elements[elements_to_assemble, np.newaxis]

    quadrature_nodes_in_each_element = elements_diameters * mesh.quadrature_formula.reference_quadrature_nodes \
                                       + mesh.nodes[elements_to_assemble, np.newaxis]
    y_alpha = quadrature_nodes_in_each_element ** alpha
    y_alpha = y_alpha[:, :, np.newaxis, np.newaxis]

    reference_u_v = space_v.reference_basis_functions.value[np.newaxis, :, :, np.newaxis] \
                    * space_u.reference_basis_functions.value[np.newaxis, :, np.newaxis, :]
    reference_y_alpha_u_v = reference_u_v * y_alpha

    assemble_matrix(reference_operation=reference_y_alpha_u_v, elements_to_assemble=elements_to_assemble,
                    space_u=space_u, space_v=space_v, mesh=mesh, matrix=matrix)

    return matrix


def assemble_weighted_u_v_element0(space_u: SpaceY, space_v: SpaceY, mesh: MeshY, matrix: npt.NDArray[np.float_]):
    diameter_element0 = mesh.diameters_of_elements[0]
    quadrature_weights_element0 = mesh.quadrature_formula_element0.reference_quadrature_weights[:, np.newaxis,
                                  np.newaxis]

    rescaled_weighted_measure = diameter_element0 ** mesh.quadrature_formula_element0.beta  # TODO check
    mapped_reference_values_u = space_u.reference_basis_functions_for_element0.value
    mapped_reference_values_v = space_v.reference_basis_functions_for_element0.value
    reference_u_v_element0 = mapped_reference_values_u[:, :, np.newaxis] \
                             * mapped_reference_values_v[:, np.newaxis, :] * rescaled_weighted_measure
    reference_integral_u_v_element0 = np.sum(quadrature_weights_element0 * reference_u_v_element0, axis=0)

    matrix[np.ix_(space_v.local2global_dofs[0, :], space_u.local2global_dofs[0, :])] \
        += reference_integral_u_v_element0 * mesh.diameters_of_elements[0]


def assemble_matrix(reference_operation: npt.NDArray[np.float_], elements_to_assemble: range,
                    space_u: Space1D, space_v: Space1D, mesh: Mesh1D, matrix: npt.NDArray[np.float_]):
    reference_quadrature_weights = mesh.quadrature_formula.reference_quadrature_weights[np.newaxis, :, np.newaxis,
                                   np.newaxis]
    reference_integral = np.sum(reference_quadrature_weights * reference_operation, axis=1)
    jacobian_determinant = mesh.diameters_of_elements

    for assembled_element_id, element_id in enumerate(elements_to_assemble):
        matrix[np.ix_(space_v.local2global_dofs[element_id, :], space_u.local2global_dofs[element_id, :])] \
            += reference_integral[assembled_element_id, :, :] * jacobian_determinant[element_id]


def impose_boundary_conditions(matrix: npt.NDArray[np.float_], zero_dirichlet_boundaries: list):
    matrix_with_bc = matrix
    if 0 in zero_dirichlet_boundaries:
        matrix_with_bc = matrix[1:, 1:]
        matrix = matrix_with_bc
    if 1 in zero_dirichlet_boundaries:
        matrix_with_bc = matrix[:-1, :-1]
    return matrix_with_bc

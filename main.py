import math
import numpy as np
import numpy.typing as npt

from scipy import linalg

from basis_functions_type import BasisFunctions1DType
from method_problem_datatypes import LaplacianProblemData, DiffusionReactionProblemData, FEMethodData
from quadrature_formula import GaussQuadrature, GaussJacobiQuadrature
from mesh import LinearMesh1D, MeshY
from operators import f_v_vector, u_v_matrix, du_dv_matrix, weighted_u_v_matrix, weighted_du_dv_matrix, \
    impose_boundary_conditions
from space import Space1D, SpaceY


# TODO:
#  --> space.boundaries and mesh.boundaries (see solve_diffusion_diffusion)
#  --> generalize f_v_vector to boundaries (in operators.py)
#  --> automatize Dirichlet boundary conditions (see solve_sfl and solve_diffusion_diffusion)
#  --> compute error


def solve_sfl(problem_data: LaplacianProblemData, method_data: FEMethodData) -> float:
    gauss_quadrature = GaussQuadrature(number_of_quadrature_nodes=method_data.number_of_gauss_quadrature_nodes)
    gauss_jacobi_quadrature = GaussJacobiQuadrature(alpha=0, beta=problem_data.alpha,
                                                    number_of_quadrature_nodes=
                                                    method_data.number_of_gauss_jacobi_quadrature_nodes)

    mesh_y = MeshY(number_of_elements=method_data.number_of_elements_per_direction[1],
                   quadrature_formula=gauss_quadrature,
                   quadrature_formula_element0=gauss_jacobi_quadrature)

    space_y = SpaceY(mesh=mesh_y,
                     degree=method_data.degree_per_direction[1],
                     derivatives_up_to=2,
                     basis_functions_type=method_data.basis_functions_type)

    stiffness_y = weighted_du_dv_matrix(space_y, space_y, mesh_y)
    mass_y = weighted_u_v_matrix(space_y, space_y, mesh_y)

    # impose Dirichlet BC TODO better
    dirichlet_boundaries_y = [1]
    stiffness_y = impose_boundary_conditions(stiffness_y, dirichlet_boundaries_y)
    mass_y = impose_boundary_conditions(mass_y, dirichlet_boundaries_y)

    # TODO better until the end of this function
    eigenvalues, eigenvectors = linalg.eig(a=mass_y, b=stiffness_y)
    eigenvectors_weighted_norm = np.sqrt(np.diag(eigenvectors.T @ stiffness_y @ eigenvectors))
    eigenvectors = eigenvectors / eigenvectors_weighted_norm

    method_data.degree_per_direction = np.array([method_data.degree_per_direction[0]])
    method_data.number_of_elements_per_direction = np.array([method_data.number_of_elements_per_direction[0]])

    number_of_dofs_x = method_data.number_of_elements_per_direction[0] * method_data.degree_per_direction[0] + 1
    extended_solution = np.zeros((number_of_dofs_x - 2, space_y.number_of_dofs - 1))
    for eigenvalue_id in range(len(eigenvalues)):
        eigenvalue = np.real(eigenvalues[eigenvalue_id])
        diffusion_reaction_rhs = lambda x: eigenvectors[0, eigenvalue_id] \
                                            * problem_data.ds * problem_data.rhs_function(x)
        diffusion_reaction_problem_data = DiffusionReactionProblemData(diffusion_coefficient=eigenvalue,
                                                                       reaction_coefficient=1,
                                                                       rhs_function=diffusion_reaction_rhs,
                                                                       dirichlet_boundaries=[0, 1],
                                                                       neumann_boundaries=[],
                                                                       dirichlet_function=lambda x: np.zeros_like(x),
                                                                       neumann_function=lambda x: np.zeros_like(x))

        solution_diffusion_reaction = solve_diffusion_reaction(problem_data=diffusion_reaction_problem_data,
                                                               method_data=method_data)
        extended_solution += solution_diffusion_reaction[:, np.newaxis] @ eigenvectors[np.newaxis, :, eigenvalue_id]

    solution = extended_solution[:, 0]
    return solution


def solve_diffusion_reaction(problem_data: DiffusionReactionProblemData, method_data: FEMethodData) -> float:
    gauss_quadrature = GaussQuadrature(number_of_quadrature_nodes=method_data.number_of_gauss_quadrature_nodes)

    mesh_x = LinearMesh1D(number_of_elements=method_data.number_of_elements_per_direction[0],
                          quadrature_formula=gauss_quadrature,
                          node_min=0, node_max=1)

    space_x = Space1D(mesh=mesh_x,
                      degree=method_data.degree_per_direction[0],
                      derivatives_up_to=2,
                      basis_functions_type=method_data.basis_functions_type)

    stiffness_x = du_dv_matrix(space_x, space_x, mesh_x)
    mass_x = u_v_matrix(space_x, space_x, mesh_x)
    rhs_vector = f_v_vector(function_f=problem_data.rhs_function, space_v=space_x, mesh=mesh_x)

    # TODO
    # neumann_vector = problem_data.diffusion_coefficient * f_v_vector(function_f=problem_data.neumann_function,
    #                                                                  space_v=space_x, mesh=mesh_x,
    #                                                                  boundaries=problem_data.neumann_boundaries)

    # impose Dirichlet BC TODO better
    stiffness_x = impose_boundary_conditions(stiffness_x, problem_data.dirichlet_boundaries)
    mass_x = impose_boundary_conditions(mass_x, problem_data.dirichlet_boundaries)
    rhs_vector = rhs_vector[1:-1]

    diffusion_reaction_matrix = problem_data.diffusion_coefficient * stiffness_x \
                                 + problem_data.reaction_coefficient * mass_x
    solution = np.linalg.inv(diffusion_reaction_matrix) @ (rhs_vector)  # TODO + neumann_vector)
    return solution


if __name__ == '__main__':
    dimension_in_x = 1
    s = 0.25


    def rhs_f(x: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]: return (math.pi ** (2 * s)) * np.sin(math.pi * x)


    number_of_elements_in_each_direction = 4
    degree = 1
    number_of_quadrature_nodes = 2 * degree + 5

    dimension = dimension_in_x + 1
    number_of_elements_per_direction = number_of_elements_in_each_direction * np.ones(dimension, dtype=int)
    degree_per_direction = degree * np.ones(dimension, dtype=int)

    sfl_problem_data = LaplacianProblemData(s=s, rhs_function=rhs_f, dirichlet_boundaries=[0, 1], neumann_boundaries=[])
    fe_method_data = FEMethodData(degree_per_direction=degree_per_direction,
                                  basis_functions_type=BasisFunctions1DType.INTEGRATED_LEGENDRE_POLYNOMIALS,
                                  number_of_gauss_quadrature_nodes=number_of_quadrature_nodes,
                                  number_of_gauss_jacobi_quadrature_nodes=number_of_quadrature_nodes,
                                  number_of_elements_per_direction=number_of_elements_per_direction)
    u_approx = solve_sfl(problem_data=sfl_problem_data, method_data=fe_method_data)

    # Exact solution of the differential problem
    u_exact = lambda x: math.sin(math.pi * x)
    du_exact = lambda x: math.pi * math.cos(math.pi * x)
    integral_f_u_exact = math.pi ** (2 * s) / 2

    err = 0  # TODO: compute the error

    print("Approximation error: ", err)

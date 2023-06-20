import math

from mesh import Mesh, MeshY
from space import Space, SpaceY

from fe_sfl_datatypes import SFLProblemData, FEMethodData


def solve_sfl(problem_data: SFLProblemData, method_data: FEMethodData) -> float:
    mesh_x = Mesh(nel=method_data.nel_x, nqn_gauss=method_data.nqn_gauss)
    mesh_y = MeshY(alpha=problem_data.alpha, nel=method_data.nel_y, nqn_gauss=method_data.nqn_gauss,
                   nqn_gauss_jacobi=method_data.nqn_gauss_jacobi)

    space_x = Space(mesh_x, deg=method_data.deg, shape_fun_type=method_data.shape_fun_type)
    space_y = SpaceY(mesh_y, deg=method_data.deg, shape_fun_type=method_data.shape_fun_type)

    Kx = space_x.stiffness_matrix()
    Mx = space_x.mass_matrix()

    Ky = space_y.stiffness_matrix()
    My = space_y.mass_matrix()

    uh = 0  # TODO: write the rest of the function
    return uh


if __name__ == '__main__':
    s = 0.25
    def rhs_f(x: float) -> float: return (math.pi ** (2 * s)) * math.sin(math.pi * x)
    nel = 4
    degree = 1
    nqn = 2 * degree + 1

    sfl_problem_data = SFLProblemData(s=s, rhs_f=rhs_f)
    fe_method_data = FEMethodData(deg=degree, shape_fun_type="IntegratedLegendrePolynomials",
                                  nqn_gauss=nqn, nqn_gauss_jacobi=nqn, nel_x=nel, nel_y=nel)
    u_approx = solve_sfl(sfl_problem_data, fe_method_data)

    # Exact solution of the differential problem
    u_ex = lambda x: math.sin(math.pi * x)
    du_ex = lambda x: math.pi * math.cos(math.pi * x)
    integral_f_u_ex = math.pi ** (2 * s) / 2

    err = 0  # TODO: compute the error

    print("Approximation error: ", err)

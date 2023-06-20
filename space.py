import numpy as np

import numpy.typing as npt

from basis_functions import IntegratedLegendrePolynomials
from mesh import Mesh, MeshY


class Space:
    mesh: Mesh
    deg: int

    ndof: int
    ndof_per_el: int
    local2global: npt.NDArray[np.int_]

    shape_fun_type: str
    shape_fun: npt.NDArray[np.float_]
    shape_der: npt.NDArray[np.float_]
    shape_hess: npt.NDArray[np.float_]

    def __init__(self, mesh: Mesh, deg: int = 1, derivatives: tuple[bool, bool, bool] = (True, False, False),
                 shape_fun_type: str = "IntegratedLegendrePolynomials"):
        self.mesh = mesh
        self.deg = deg
        self.shape_fun_type = shape_fun_type

        self.ndof = self.mesh.nel * self.deg + 1
        self.ndof_per_el = self.deg + 1
        self.local2global = self.__compute_local2global()

        self.__compute_shape_fun(derivatives)

    def __compute_local2global(self) -> npt.NDArray[np.int_]:
        local2global = np.empty((self.mesh.nel, self.ndof_per_el))
        global_dof_index = 0

        for iel in range(self.mesh.nel):
            local2global[iel, :] = np.arange(global_dof_index, global_dof_index + self.ndof_per_el)
            global_dof_index = global_dof_index + self.deg

        return local2global

    def __compute_shape_fun(self, derivatives: tuple[bool, bool, bool] = (True, False, False)):
        if self.shape_fun_type != "IntegratedLegendrePolynomials":
            print("""Warning: the chosen type of basis function has not been implemented yet, so it has
                  been changed to 'IntegratedLegendrePolynomials'.""")
        basis_fun = IntegratedLegendrePolynomials(deg=self.deg, derivatives=derivatives)

        reference_basis_fun = basis_fun.evaluate(self.mesh.param_gauss_nodes)
        self.shape_fun = reference_basis_fun["fun"]
        self.shape_der = reference_basis_fun["der"]
        self.shape_hess = reference_basis_fun["hess"]

    def stiffness_matrix(self):
        return NotImplemented  # TODO

    def mass_matrix(self):
        return NotImplemented  # TODO


class SpaceY(Space):
    mesh: MeshY

    shape_fun_jacobi: npt.NDArray[np.float_]
    shape_der_jacobi: npt.NDArray[np.float_]
    shape_hess_jacobi: npt.NDArray[np.float_]

    def __init__(self, mesh: MeshY, deg: int = 1, derivatives: tuple[bool, bool, bool] = (True, False, False),
                 shape_fun_type: str = "IntegratedLegendrePolynomials"):
        Space.__init__(self, mesh=mesh, deg=deg, derivatives=derivatives, shape_fun_type=shape_fun_type)
        self.__compute_shape_fun_jacobi(derivatives)

    def __compute_shape_fun_jacobi(self, derivatives: tuple[bool, bool, bool] = (True, False, False)):
        basis_fun = IntegratedLegendrePolynomials(deg=self.deg, derivatives=derivatives)
        reference_basis_fun_jacobi = basis_fun.evaluate(self.mesh.param_gauss_jacobi_nodes)
        self.shape_fun_jacobi = reference_basis_fun_jacobi["fun"]
        self.shape_der_jacobi = reference_basis_fun_jacobi["der"]
        self.shape_hess_jacobi = reference_basis_fun_jacobi["hess"]

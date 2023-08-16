import math
import numpy as np
import numpy.typing as npt

from typing import List

from quadrature_formula import QuadratureFormulaND, GaussJacobiQuadrature, QuadratureFormula


class Mesh:
    dimension: int
    number_of_elements: int = 0
    nodes: npt.NDArray[np.float_]
    diameters_of_elements: npt.NDArray[np.float_]
    quadrature_formula: QuadratureFormula = None


class Mesh0D(Mesh):
    def __init__(self, node: float):
        self.dimension = 0
        self.number_of_elements = 1
        self.nodes = np.array([node])
        self.diameters_of_elements = np.array([1])


class TensorProductMesh(Mesh):
    pass


class Mesh1D(TensorProductMesh):
    boundaries: List[Mesh0D]

    def __init__(self, quadrature_formula: QuadratureFormula, nodes: npt.NDArray[np.float_],
                 compute_boundaries: bool = True):
        self.dimension = 1
        if self.number_of_elements == 0:
            self.number_of_elements = len(nodes) - 1
        self.nodes = nodes
        self.diameters_of_elements = np.diff(nodes)
        self.quadrature_formula = quadrature_formula
        if compute_boundaries:
            self.boundaries = [Mesh0D(node=nodes[0]), Mesh0D(node=nodes[-1])]
        else:
            self.boundaries = []


class TensorProductMeshND(TensorProductMesh):
    mesh_per_direction: List[Mesh1D]
    number_of_elements_per_direction: npt.NDArray[np.int_] = None

    def __init__(self, mesh_per_direction: List[Mesh1D]):
        self.dimension = len(mesh_per_direction)
        self.mesh_per_direction = mesh_per_direction
        if self.number_of_elements_per_direction is None:
            self.set_number_of_elements_per_direction()

        self.number_of_elements = self.number_of_elements_per_direction.prod()
        self.set_nodes()
        self.set_diameters_of_elements()
        self.set_quadrature_formula_tensor_product()

    def set_number_of_elements_per_direction(self):
        self.number_of_elements_per_direction = np.array([mesh_1d.number_of_elements
                                                          for mesh_1d in self.mesh_per_direction])

    def set_nodes(self):
        coordinates_of_nodes = [mesh_1d.nodes.T for mesh_1d in self.mesh_per_direction]
        grid_coordinates_of_nodes = np.meshgrid(*coordinates_of_nodes)
        self.nodes = np.vstack([coordinate.ravel() for coordinate in grid_coordinates_of_nodes]).T

    def set_diameters_of_elements(self):
        length_of_sides = [mesh_1d.diameters_of_elements for mesh_1d in self.mesh_per_direction]
        grid_length_of_sides = np.meshgrid(*length_of_sides)
        self.diameters_of_elements = np.sqrt(np.sum([side_length ** 2 for side_length in grid_length_of_sides]))

    def set_quadrature_formula_tensor_product(self):
        quadrature_formula_per_direction = [mesh_1d.quadrature_formula for mesh_1d in self.mesh_per_direction]
        quad_formula_tensor_product = QuadratureFormulaND(
            quadrature_formula_per_direction=quadrature_formula_per_direction)
        self.quadrature_formula = quad_formula_tensor_product


class LinearMesh1D(Mesh1D):
    def __init__(self, quadrature_formula: QuadratureFormula, number_of_elements: int = 1,
                 node_min: float = 0., node_max: float = 1., compute_boundaries: bool = True):
        self.number_of_elements = number_of_elements
        number_of_nodes = number_of_elements + 1
        nodes = np.linspace(node_min, node_max, num=number_of_nodes)
        super().__init__(quadrature_formula=quadrature_formula, nodes=nodes, compute_boundaries=compute_boundaries)


class MeshY(LinearMesh1D):
    Y: float
    quadrature_formula_element0: GaussJacobiQuadrature

    def __init__(self, quadrature_formula: QuadratureFormula, quadrature_formula_element0: GaussJacobiQuadrature,
                 number_of_elements: int = 1, compute_boundaries: bool = True):
        self.Y = math.floor(math.log(number_of_elements))
        self.quadrature_formula_element0 = quadrature_formula_element0
        super().__init__(quadrature_formula=quadrature_formula, number_of_elements=number_of_elements,
                         node_min=0, node_max=self.Y, compute_boundaries=compute_boundaries)


class LinearTensorProductMesh(TensorProductMeshND):
    mesh_per_direction: List[LinearMesh1D]

    def __init__(self, quadrature_formulas: List[QuadratureFormula],
                 number_of_elements_per_direction: npt.NDArray[np.int_]):
        self.number_of_elements_per_direction = number_of_elements_per_direction
        dimension = len(number_of_elements_per_direction)
        mesh_per_direction = [LinearMesh1D(quadrature_formulas[direction],
                                           number_of_elements=number_of_elements_per_direction[direction])
                              for direction in range(dimension)]
        super().__init__(mesh_per_direction=mesh_per_direction)


class ExtendedTensorProductMesh(TensorProductMeshND):
    mesh_x: TensorProductMesh
    mesh_y: MeshY

    def __init__(self, mesh_x: TensorProductMesh, mesh_y: MeshY):
        mesh_per_direction = []
        if isinstance(mesh_x, Mesh1D):
            mesh_per_direction = [mesh_x, mesh_y]
        elif isinstance(mesh_x, TensorProductMeshND):
            mesh_per_direction = mesh_x.mesh_per_direction + [mesh_y]
        super().__init__(mesh_per_direction=mesh_per_direction)

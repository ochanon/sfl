import numpy as np
import numpy.typing as npt

from typing import List


def are_all_equal(given_list: list) -> bool:
    if len(given_list) == 0:
        return True

    first_element = given_list[0]
    return all(element == first_element for element in given_list)


def to_array_with_zero_padding(given_list: List[npt.NDArray[np.float_]]) -> npt.NDArray[np.float_]:
    shapes_of_elements = np.array([element.shape for element in given_list])
    maximum_dimensions = np.max(shapes_of_elements, axis=0)
    number_of_dimensions = len(maximum_dimensions)
    resulting_array = np.zeros((len(given_list), *maximum_dimensions))

    for dimension, element in enumerate(given_list):
        slices = (dimension, *[slice(element.shape[dimension2]) for dimension2 in range(number_of_dimensions)])
        resulting_array[slices] = element

    return resulting_array

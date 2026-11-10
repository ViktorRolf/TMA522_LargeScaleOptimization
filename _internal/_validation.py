from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _verify_input(
    width: int,
    height: int,
    dual_values: NDArray[np.float64],
    contact_pairs: NDArray[np.int64],
) -> None:
    """Raise ValueError unless the grid size, the dual values and the contact
    pairs are valid. The arguments must already be converted to ints and
    arrays."""
    if width < 1 or height < 1:
        raise ValueError(f"width and height must be positive, got {width} and {height}")
    _verify_dual_values(dual_values, width, height)
    _verify_contact_pairs(contact_pairs, width, height)


def _verify_dual_values_shape(dual_values: NDArray[np.float64], width: int, height: int) -> None:
    if dual_values.shape != (width, height, 2):
        raise ValueError(
            f"dual_values must have shape (width, height, 2) = {(width, height, 2)}, "
            f"got {dual_values.shape}"
        )


def _verify_dual_values(dual_values: NDArray[np.float64], width: int, height: int) -> None:
    _verify_dual_values_shape(dual_values, width, height)
    bad = np.argwhere(~(dual_values >= 0))  # also catches NaN
    if len(bad):
        node = tuple(bad[0].tolist())
        raise ValueError(
            "dual_values must be non-negative (inf is allowed), but "
            f"dual_values[{', '.join(map(str, node))}] = {dual_values[node]}"
        )


def _verify_contact_pairs(contact_pairs: NDArray[np.int64], width: int, height: int) -> None:
    upper = np.array([width, height, 2])
    outside = np.flatnonzero(np.any((contact_pairs < 0) | (contact_pairs >= upper), axis=(1, 2)))
    if outside.size:
        l = int(outside[0])
        raise ValueError(
            f"contact_pairs[{l}] = {contact_pairs[l].tolist()} contains a node outside "
            f"the grid; nodes are (x, y, z) with 0 <= x < {width}, 0 <= y < {height} "
            "and z = 0 or 1"
        )

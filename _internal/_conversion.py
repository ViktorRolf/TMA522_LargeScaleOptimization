from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _as_contiguous_integer_array(values: ArrayLike, argument_name: str) -> NDArray[np.int64]:
    array = np.asarray(values)
    if array.dtype.kind in "iu" or (array.dtype.kind == "f" and np.all(np.mod(array, 1) == 0)):
        return np.ascontiguousarray(array, dtype=np.int64)
    raise TypeError(f"{argument_name} must contain integer node coordinates")


def _as_cost_array(values: ArrayLike) -> NDArray[np.float64]:
    return np.ascontiguousarray(values, dtype=np.float64)


def _as_contact_pair_array(contact_pairs: ArrayLike) -> NDArray[np.int64]:
    contact_pairs = _as_contiguous_integer_array(contact_pairs, "contact_pairs")
    if contact_pairs.size == 0 or contact_pairs.shape == (2, 3):
        contact_pairs = contact_pairs.reshape(-1, 2, 3)
    if contact_pairs.shape[1:] != (2, 3):
        raise ValueError(
            "contact_pairs must have one row [start, end] of (x, y, z) nodes per "
            f"contact pair, i.e. shape (k, 2, 3), got shape {contact_pairs.shape}"
        )
    return contact_pairs


def _as_route_array(route: ArrayLike) -> NDArray[np.int64]:
    route = _as_contiguous_integer_array(route, "route")
    if route.ndim != 2 or route.shape[1] != 3:
        raise ValueError(f"a route must be a sequence of (x, y, z) nodes, got shape {route.shape}")
    return route

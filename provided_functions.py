"""Helper routines for Project 1 in TMA522/MMA511 Large-Scale Optimization:
VLSI routing and Lagrangean duality.

cheapest_grid_routes        cheapest routes for all contact pairs on the wiring grid
plot_grid_routes            draw the wiring grid and a set of routes
load_instance               read a test instance from a JSON file

Nodes are indexed by tuples (x, y, z), with x = 0, ..., width-1, y = 0, ..., height-1,
and z = 0, 1, with z = 0 denoting the front of the PCB (with horizontal traces), and
z = 1 the back (with vertical traces).
"""

from __future__ import annotations

import json
import operator
import os
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np
from matplotlib.axes import Axes
from numpy.typing import ArrayLike, NDArray

if __package__:
    from ._internal import _conversion, _grid, _plotting, _validation
else:
    from _internal import _conversion, _grid, _plotting, _validation

__all__ = [
    "RoutingInstance",
    "cheapest_grid_routes",
    "load_instance",
    "plot_grid_routes",
]


class RoutingInstance(NamedTuple):
    width: int
    height: int
    contact_pairs: NDArray[np.int64]


def load_instance(path: str | os.PathLike[str]) -> RoutingInstance:
    """Create an object representing the instance in the JSON file ``path``.

    The file has the keys ``width``, ``height`` and ``contact_pairs``.
    ``contact_pairs`` has one row ``[start, end]`` per contact pair, where
    each node is ``[x, y, z]``, so its shape is ``(k, 2, 3)`` for ``k =
    len(contact_pairs)`` contact pairs. Example::

        width, height, contact_pairs = load_instance("instances/VLSI_8_6_7.json")
        k = len(contact_pairs)
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    contact_pairs = np.array(data["contact_pairs"], dtype=np.int64).reshape(-1, 2, 3)
    return RoutingInstance(data["width"], data["height"], contact_pairs)


def cheapest_grid_routes(
    width: int, height: int, dual_values: ArrayLike, contact_pairs: ArrayLike
) -> list[list[_grid.Node] | None]:
    """Find a cheapest route for each contact pair on the wiring grid.

    Parameters
    ----------
    width, height : int
        Number of vias in the x and y directions. The grid has the
        ``2 * width * height`` nodes.
    dual_values : array_like, shape (width, height, 2)
        Lagrangean dual values. ``dual_values[x, y, z]`` is the cost of
        passing node ``(x, y, z)`` in the subproblem.
    contact_pairs : array_like of int, shape (_, 2, 3)
        Start and end node ``[s, t]`` of each contact pair, each node as
        ``[x, y, z]``.

    Returns
    -------
    routes : list of (list of (x, y, z) tuples or None)
        One entry per contact pair, in the order of ``contact_pairs``.
        ``routes[l]`` is a cheapest route of pair ``l``: it starts with
        node ``contact_pairs[l, 0]`` and ends with ``contact_pairs[l, 1]``.
        Its cost ``sum(dual_values[v] for v in routes[l])`` is the sum of
        ``dual_values`` over all its nodes. If every route of pair ``l``
        costs more than 1, ``routes[l]`` is ``None``.

    Notes
    -----
    Ties between equally cheap routes are broken at random: each node cost is
    perturbed by a tiny value (in [0, 1e-8)) for each contact pair. The random sequence is
    deterministic, i.e., restarting Python and re-running gives the same
    result.
    """
    width, height = operator.index(width), operator.index(height)
    node_costs = _conversion._as_cost_array(dual_values)
    contact_pairs = _conversion._as_contact_pair_array(contact_pairs)
    _validation._verify_input(width, height, node_costs, contact_pairs)
    return _grid.cheapest_routes(width, height, node_costs, contact_pairs)


def plot_grid_routes(
    width: int,
    height: int,
    routes: Sequence[ArrayLike],
    *,
    dual_values: ArrayLike | None = None,
    route_labels: ArrayLike | None = None,
    axes: Axes | None = None,
    label_nodes: bool = False,
    layer_offset: float = 25,
) -> Axes:
    """Draw the wiring grid and ``routes``.

    Parameters
    ----------
    width, height : int
        Number of vias in the x and y directions.
    routes : sequence of (sequence of (x, y, z) nodes, or None)
        Routes as returned by :func:`cheapest_grid_routes`, one per contact
        pair. ``None`` entries are not drawn.
    dual_values : array_like, shape (width, height, 2), optional
        Node costs; nodes with a positive cost are drawn as rings.
    route_labels : array_like, optional
        Label written at both ends of each route; by default the row number
        in ``routes``.
    axes : matplotlib Axes, optional
        Where to draw. By default the current figure is cleared and used.
    label_nodes : bool, optional
        Write the coordinates ``x,y,z`` of every node next to it.
    layer_offset : float, optional
        Displacement of the lower layer on the screen.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on.
    """

    dual_values = _plotting._checked_dual_values(dual_values, width, height)
    routes = [None if route is None else _conversion._as_route_array(route) for route in routes]
    route_labels = _plotting._checked_route_labels(route_labels, len(routes))
    xy = _plotting._node_plot_coordinates(width, height, layer_offset)

    if axes is None:
        axes = _plotting._fresh_axes()
    _plotting._plot_nodes(axes, xy, dual_values)
    _plotting._plot_routes(axes, xy, routes, route_labels)
    if label_nodes:
        _plotting._label_nodes(axes, xy)
    _plotting._format_axes(axes, width, height)
    return axes

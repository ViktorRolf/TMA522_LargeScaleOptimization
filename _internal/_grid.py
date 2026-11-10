"""The two-layer Manhattan wiring grid of Feo & Hochbaum and the cheapest
routes on it (0-based).

Node ``(x, y, z)`` is the via at column ``x`` and row ``y`` on the upper layer
(``z = 0``, horizontal wires) or on the lower layer (``z = 1``, vertical
wires). Node costs are an array of shape ``(width, height, 2)``, indexed by
the nodes.

The functions here expect valid arguments, already converted to plain ints
and NumPy arrays; ``_conversion`` and ``_validation`` check and convert them for the public functions.
"""

from __future__ import annotations

import functools

import numpy as np
from numpy.typing import NDArray

from ._shortest_path import Graph, dijkstra_shortest_path_to

Node = tuple[int, int, int]
"""A node ``(x, y, z)`` of the wiring grid."""


# Graph will be identical in many calls in sequence. Cache the result
# and re-use
@functools.lru_cache(maxsize=16)
def _grid_graph(width: int, height: int) -> Graph:
    return Graph(width, height)

NOISE_MAX = 1e-8

def cheapest_routes(
    width: int,
    height: int,
    node_costs: NDArray[np.float64],
    contact_pairs: NDArray[np.int64],
) -> list[list[Node] | None]:
    """Cheapest route for each contact pair ``[s, t]`` in ``contact_pairs``
    (shape ``(k, 2, 3)``), where a route costs the sum of ``node_costs`` over
    all its nodes. Returns one entry per contact pair: the route listed from
    ``s`` to ``t`` as ``(x, y, z)`` tuples, or ``None`` if every route of that
    pair costs more than 1.

    Every node cost gets a tiny random perturbation, uniform in
    ``[0, NOISE_MAX)`` and drawn from ``tie_breaking_rng`` afresh for every node and contact
    pair, so that ties between equally cheap routes are broken at random.
    """
    graph = _grid_graph(width, height)
    shape = (width, height, 2)
    flat_costs = node_costs.ravel()
    flat_pairs = np.ravel_multi_index(tuple(contact_pairs.reshape(-1, 3).T), shape).reshape(-1, 2)
    routes: list[list[Node] | None] = []
    for s, t in flat_pairs.tolist():
        # Perturb costs to break ties
        perturbed_costs = (flat_costs + tie_breaking_rng.uniform(0.0, NOISE_MAX, graph.total_node_count)).tolist()
        path = dijkstra_shortest_path_to(graph, perturbed_costs, s, t)
        routes.append(None if path is None else list(zip(*(c.tolist() for c in np.unravel_index(path, shape)))))
    return routes


# The random generator of cheapest_routes. It continues from
# call to call and starts from the same seed in every new Python session.
tie_breaking_rng = np.random.default_rng(0)

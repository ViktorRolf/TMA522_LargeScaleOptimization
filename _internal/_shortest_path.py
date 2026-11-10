"""Dijkstra's shortest-path search on the wiring grid with non-negative node
costs. Node (x, y, z) is the flat index of ``[x, y, z]`` in a C-ordered array
of shape ``(width, height, 2)``.
"""

from __future__ import annotations

import numpy as np
from heapq import heappop, heappush
from typing import NamedTuple

NO_NODE = -1
"""Predecessor node of the root and of nodes the search did not yet reach."""


class Graph:
    """The wiring grid as a directed graph: ``neighbors[v]`` lists the flat
    indices of the nodes reachable in one step from node ``v``.
    """

    def __init__(self, width: int, height: int):
        def node(x, y, z):
            # The position of node (x, y, z) in a C-ordered (width, height, 2) array
            return (x * height + y) * 2 + z

        neighbors: list[tuple[int, ...]] = [()] * (2 * width * height)
        for x in range(width):
            for y in range(height):
                # Upper layer: left, right, and the via down to the lower layer.
                neighbors[node(x, y, 0)] = (
                    *([node(x - 1, y, 0)] if x > 0 else []),
                    *([node(x + 1, y, 0)] if x + 1 < width else []),
                    node(x, y, 1),
                )
                # Lower layer: the via up to the upper layer, down, and up.
                neighbors[node(x, y, 1)] = (
                    node(x, y, 0),
                    *([node(x, y - 1, 1)] if y > 0 else []),
                    *([node(x, y + 1, 1)] if y + 1 < height else []),
                )

        self.neighbors = neighbors

    @property
    def total_node_count(self):
        return len(self.neighbors)


def dijkstra_shortest_path_to(
    graph: Graph, node_costs: list[float], root_node_index: int, target_node_index: int
) -> NDArray[np.int64] | None:
    """Shortest path from ``root_node_index`` to ``target_node_index`` (Dijkstra). ``arc_lengths``
    is indexed by arc and must be non-negative (and not NaN); ``inf`` marks
    an unusable arc. Only the path to ``target_node_index`` is complete in the returned
    tree.
    """
    if node_costs[root_node_index] >= 1 + 1e-6:
        # Every path, even the one-node path to the root itself, is too costly
        return None

    shortest_known_distance_from_root = [np.inf] * graph.total_node_count
    shortest_known_distance_from_root[root_node_index] = node_costs[root_node_index]

    predecessor = [NO_NODE] * graph.total_node_count
    heap = [(node_costs[root_node_index], root_node_index)]

    while heap:
        distance, node_index = heappop(heap)

        if node_index == target_node_index:
            # The first time the target is popped, its distance is final
            path = [target_node_index]
            while path[-1] != root_node_index:
                path.append(predecessor[path[-1]])
            return np.array(path[::-1], dtype=np.int64)

        if distance > shortest_known_distance_from_root[node_index]:
            # Entry is outdated
            continue

        for extension_target_node_index in graph.neighbors[node_index]:
            distance_to_target = distance + node_costs[extension_target_node_index]

            if distance_to_target < min(shortest_known_distance_from_root[extension_target_node_index], 1 + 1e-6):
                shortest_known_distance_from_root[extension_target_node_index] = distance_to_target
                predecessor[extension_target_node_index] = node_index
                heappush(heap, (distance_to_target, extension_target_node_index))

    return None

"""Private helpers of :func:`plot_grid_routes` (port of ``visagrid.m``)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from ._conversion import _as_cost_array
from ._validation import _verify_dual_values_shape

_ROUTE_COLORS = [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#6250d6", "#e34948",
]
_UPPER_LAYER_COLOR = "#b33a3a"
_LOWER_LAYER_COLOR = "#3a5fb3"


def _node_plot_coordinates(
    width: int, height: int, layer_offset: float = 25
) -> NDArray[np.float64]:
    """Plot coordinates of all nodes, as an array of shape ``(width, height, 2, 2)``:
    ``xy[x, y, z]`` is where node ``(x, y, z)`` is drawn.
    """
    x, y, z = np.meshgrid(np.arange(width), np.arange(height), np.arange(2), indexing="ij")
    return np.stack([100.0 * (x + 1) + layer_offset * z, 100.0 * (y + 1) + layer_offset * z], axis=-1)


def _checked_dual_values(
    dual_values: ArrayLike | None, width: int, height: int
) -> NDArray[np.float64] | None:
    """Convert ``dual_values`` to a cost array of shape ``(width, height, 2)``, if given."""
    if dual_values is None:
        return None
    dual_values = _as_cost_array(dual_values)
    _verify_dual_values_shape(dual_values, width, height)
    return dual_values


def _checked_route_labels(route_labels: ArrayLike | None, n_routes: int) -> np.ndarray:
    """Labels for ``n_routes`` routes; by default the route numbers."""
    route_labels = (
        np.arange(n_routes) if route_labels is None else np.asarray(route_labels).reshape(-1)
    )
    if len(route_labels) != n_routes:
        raise ValueError(f"got {len(route_labels)} labels for {n_routes} routes")
    return route_labels


def _fresh_axes() -> Axes:
    fig = plt.gcf()
    fig.clf()
    return fig.add_subplot()


def _plot_nodes(
    axes: Axes, xy: NDArray[np.float64], dual_values: NDArray[np.float64] | None
) -> None:
    """Draw the nodes of both layers, those with a positive dual value as rings."""
    for z, color, layer in ((0, _UPPER_LAYER_COLOR, "upper layer"), (1, _LOWER_LAYER_COLOR, "lower layer")):
        c = xy[:, :, z].reshape(-1, 2)
        if dual_values is None:
            positive = np.zeros(len(c), dtype=bool)
        else:
            positive = dual_values[:, :, z].reshape(-1) > 1e-12
        if (~positive).any():
            axes.plot(*c[~positive].T, ".", color=color, markersize=3, label=layer)
        if positive.any():
            axes.plot(
                *c[positive].T, "o", markerfacecolor="none", markeredgecolor=color,
                markersize=6, label=f"{layer}, dual value > 0",
            )


def _plot_routes(
    axes: Axes,
    xy: NDArray[np.float64],
    routes: Sequence[NDArray[np.int64] | None],
    route_labels: NDArray[np.generic],
) -> None:
    """Draw each route as a coloured line with its label at both ends,
    skipping ``None`` routes."""
    for l, (route, label) in enumerate(zip(routes, route_labels)):
        if route is None:
            continue
        # Integer labels pick the colour, so a pair keeps it in subsets.
        key = int(label) if route_labels.dtype.kind in "iu" else l
        color = _ROUTE_COLORS[key % len(_ROUTE_COLORS)]
        # Offset each route by l units so that overlapping routes stay visible.
        c = xy[tuple(route.T)] + l
        axes.plot(*c.T, "-", color=color, linewidth=2, solid_capstyle="round")
        for end, va in ((c[0], "top"), (c[-1], "bottom")):
            axes.annotate(
                str(label), end, xytext=(0, -6 if va == "top" else 6),
                textcoords="offset points", ha="center", va=va, fontsize=8,
                color="0.2",
            )


def _label_nodes(axes: Axes, xy: NDArray[np.float64]) -> None:
    """Write the coordinates ``x,y,z`` of every node next to it."""
    for node in np.ndindex(*xy.shape[:3]):
        axes.annotate(
            ",".join(map(str, node)), xy[node], xytext=(2, 2), textcoords="offset points",
            fontsize=6, color=(_UPPER_LAYER_COLOR, _LOWER_LAYER_COLOR)[node[2]],
        )


def _format_axes(axes: Axes, width: int, height: int) -> None:
    """Fit the limits to the grid, hide the ticks and put the legend below."""
    axes.set_xlim(0, (width + 1) * 100)
    axes.set_ylim(0, (height + 1) * 100)
    axes.set_aspect("equal")

    axes.set_xticks([])
    axes.set_yticks([])
    axes.legend(
        loc="upper left", bbox_to_anchor=(0, 0), ncol=4, fontsize=8,
        frameon=False, handletextpad=0.2, columnspacing=1.0,
    )

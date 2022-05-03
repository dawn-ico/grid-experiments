from numpy import ndarray
from regex import F
from grid_types import Grid, DEVICE_MISSING_VALUE, GridSet
from location_type import LocationType
from schemas import *
import pymorton.pymorton as pm

import numpy as np
import netCDF4
from functools import cmp_to_key

from matplotlib import (
    pyplot as plt,
    patches,
    collections,
)

NaN = float("nan")


def apply_permutation(
    ncf, perm: np.ndarray, schema: GridScheme, location_type: LocationType
) -> None:
    rev_perm = revert_permutation(perm)

    for field_name, descr in schema.items():

        field = ncf.variables[field_name]
        array = np.copy(field[:])

        if (
            descr.location_type is location_type
            and not descr.do_not_reorder_primary_loc
        ):
            if 1 < len(array.shape):
                assert descr.primary_axis is not None
                array = np.take(array, perm, axis=descr.primary_axis)
            else:
                array = array[perm]

        if descr.indexes_into is location_type and not descr.do_not_reorder_indexes:
            # go from fortran's 1-based indexing to python's 0-based indexing
            array = array - 1

            # remap indices
            missing_values = array == DEVICE_MISSING_VALUE
            array = rev_perm[array]
            array[missing_values] = DEVICE_MISSING_VALUE

            array = array + 1

        field[:] = array


def sort_nbh_list(ncf, schema: GridScheme) -> None:
    nbh_lists = {
        # "edges_of_vertex",       DO NOT REORDER THIS (fails probtest)
        "cells_of_vertex",
        # "edge_vertices",         DO NOT REORDER THIS (fails probtest)
        # "adjacent_cell_of_edge", DO NOT REORDER THIS (crashes icon)
        "vertex_of_cell",
        # "edge_of_cell",          DO NOT REORDER THIS (fails probtest)
    }

    for field_name, descr in schema.items():
        if field_name in nbh_lists:
            field = ncf.variables[field_name]
            array = np.copy(field[:])
            array.sort(axis=0)
            field[:] = array


def fix_hole(ncf, schema: GridScheme):
    for field_name, descr in schema.items():

        field = ncf.variables[field_name]
        array = np.copy(field[:])

        nc = ncf.dimensions["cell"].size
        ne = ncf.dimensions["edge"].size
        nv = ncf.dimensions["vertex"].size

        # NOTE: this seems extremely brittle, but not sure how to improve
        if field_name == "end_idx_c":
            array[0, 8] = nc
            field[:] = array

        if field_name == "end_idx_v":
            array[0, 7] = nv
            field[:] = array

        if field_name == "end_idx_e":
            array[0, 13] = ne
            field[:] = array


def get_grf_ranges(grid: Grid, location_type: LocationType = LocationType.Cell):

    # returns the index ranges of the grid refinement valid_regions
    # region 0 is the compute domain.
    # all other regions are the lateral boundary layers starting from most outer
    # and going to most inner.

    if location_type is LocationType.Vertex:
        n = grid.nv
        start, end = grid.v_grf[:, 0], grid.v_grf[:, 1]
    elif location_type is LocationType.Edge:
        n = grid.ne
        start, end = grid.e_grf[:, 0], grid.e_grf[:, 1]
    elif location_type is LocationType.Cell:
        n = grid.nc
        start, end = grid.c_grf[:, 0], grid.c_grf[:, 1]
    else:
        raise ValueError

    valid_regions = start <= end
    start = start[valid_regions]
    end = end[valid_regions]
    end = end + 1  # end is exclusive

    assert np.min(start) == 0
    assert np.max(end) <= n

    # There's something very weird going on:
    # Some few vertices/edges/cells (at the end) aren't in any valid region,
    # but without them, there will be a hole in the compute domain.
    # We fix this by assigning them to region `0` by default.

    end[0] = n

    return list(zip(start, end))


def range_to_slice(range: typing.Tuple[typing.Optional[int], typing.Optional[int]]):
    return slice(range[0], range[1])


def normalize_angle(angle):
    return np.fmod(angle, 2 * np.pi)


def get_angle(p):
    return np.arctan2(p[:, 1], p[:, 0])


def rotate(points, angle, origin=np.array([[0, 0]])):
    points = points - origin
    rotation_matrix = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    points = (rotation_matrix @ points.T).T
    return points + origin


class UnsupportedPentagonException(Exception):
    pass


def neighbor_array_to_set(array):
    nbhs = np.unique(array)
    return nbhs[nbhs != DEVICE_MISSING_VALUE]


###############################################################################
# Each vertex is the crossing point of 6 rays. Two of twos rays are defined as
# the cartesian x- and y-axis (marked as double lines below).
# With this, we can give each vertex a unique x/y coordinate, as shown below.
#
#                 2                                       1
#                   \                                  //
#                     \                              //
#                       \                          //
#                 [-1, 1] *-----------------------* [0, 1]
#                       /   \                  //   \
#                     /       \              //       \
#                   /           \          //           \
#                 /               \      //               \
#       [-1, 0] /                   \  // [0, 0]            \ [1, 0]
# 3 ===========*======================*======================*=============== 0
#               \                   //  \                   /
#                 \               //      \               /
#                   \           //          \           /
#                     \       //              \       /
#                       \   //                  \   /
#                 [0, -1] *-----------------------* [1, -1]
#                       //                          \
#                     //                              \
#                   //                                  \
#                 4                                       5
#
###############################################################################
structured_v2v_offsets = np.array(
    [
        # neighbor id/ray 0
        [+1, +0],
        # neighbor id/ray 1
        [+0, +1],
        # neighbor id/ray 2
        [-1, +1],
        # neighbor id/ray 3
        [-1, +0],
        # neighbor id/ray 4
        [+0, -1],
        # neighbor id/ray 5
        [+1, -1],
    ],
    dtype=int,
)

###############################################################################
# Once each vertex has a unique x/y coordinate, we use those to assign
# to each edge & cell a x/y coordinate and a color. For each edge & cell
# we look for the closest vertex in the bottom left direction. This vertex
# determines the x/y coordinate of each edge & cell. Then the coloring is done
# from left to right in a counter clock-wise direction.
# (This is similar to dawn's `ICOChainSize`, but uses a slightly different ordering)
#
#        /                          \       /                                  /
# \    /                             \    /                                  /
#  \ /                                \ /                                  /
#  -*------------------------ [x, y+1] *==================================* [x+1, y+1]
#  / \                               // \                               // \
#     \                            //    \                            //    \
#      \                         //       \        [x, y, 1]        //       \
#       \                      //          \                      //          \
#        \                   //             \                   //             \
#         \             [x, y, 0]        [x, y, 1]            //                \
#          \             //                   \             //                   \
#           \          //                      \          //                      \
#            \       //       [x, y, 0]         \       //
#             \    //                            \    //
#              \ //                               \ //
#   ---- [x, y] *============[x, y, 2]=============* [x+1, y] ---------------------
#              / \                                / \
#            /    \                             /    \
#          /       \                          /       \
#        /          \                       /          \
#
###############################################################################
structured_v2e_offsets = np.array(
    [
        # neighbor id/ray 0
        [+0, +0, +2],
        # neighbor id/ray 1
        [+0, +0, +0],
        # neighbor id/ray 2
        [-1, +0, +1],
        # neighbor id/ray 3
        [-1, +0, +2],
        # neighbor id/ray 4
        [+0, -1, +0],
        # neighbor id/ray 5
        [+0, -1, +1],
    ],
    dtype=int,
)
# (for the cells, we shift the rays 15 degrees counter clock-wise)
structured_v2c_offsets = np.array(
    [
        # neighbor id/ray 0
        [+0, +0, +0],
        # neighbor id/ray 1
        [-1, +0, +1],
        # neighbor id/ray 2
        [-1, +0, +0],
        # neighbor id/ray 3
        [-1, -1, +1],
        # neighbor id/ray 4
        [+0, -1, +0],
        # neighbor id/ray 5
        [+0, -1, +1],
    ],
    dtype=int,
)


@dataclasses.dataclass
class GridMapping:
    vertex_mapping: np.ndarray
    edge_mapping: np.ndarray
    cell_mapping: np.ndarray


def create_structured_grid_mapping(
    grid: Grid, right_direction_angle, start_vertex=None, angle_threshold=np.deg2rad(30)
) -> GridMapping:
    # doesn't support pentagons!

    if start_vertex is None:
        start_vertex = 0

    if isinstance(right_direction_angle, np.ndarray):
        right_direction_angle = float(right_direction_angle)

    vertex_mapping = np.full((grid.nv, 2), NaN)
    edge_mapping = np.full((grid.ne, 3), NaN)
    cell_mapping = np.full((grid.nc, 3), NaN)

    vertex_mapping[start_vertex] = [0, 0]

    # This algorithms works as follows:
    #
    # * Carry out a breadth-first search starting from `start_vertex`.
    # * For each vertex:
    #   * Determine for each neighbor edge, cell, vertex what is their relative id.
    #     (see `structured_<...>_offsets`)
    #   * For each neighbor edge, cell, vertex check which ones have no coordinates assigned yet:
    #     * Assign new coordinates to neighbors if they don't have coordinates yet.
    #     * Check if coordinates are consistent if they already have coordinates.
    #   * Update the right direction angle based on the neighboring vertices of the vertex
    #     (this way the algorithm can handle a small local curvature)
    #   * Continue the bsf with the vertices that have newly assigned coordinates.

    def bfs(vertex_id, right_direction_angle):

        # neighbor cells, edges, vertices
        cell_ids = neighbor_array_to_set(grid.v2c[vertex_id])
        edge_ids = neighbor_array_to_set(grid.v2e[vertex_id])
        vertex_ids = neighbor_array_to_set(grid.e2v[edge_ids])
        vertex_ids = vertex_ids[vertex_ids != vertex_id]

        # some sanity checks
        if len(edge_ids) == 5 and len(cell_ids) == 5:
            raise UnsupportedPentagonException

        assert len(edge_ids) == len(cell_ids) == 6 or len(cell_ids) + 1 == len(edge_ids)
        assert len(vertex_ids) == len(edge_ids)
        assert 0 < len(cell_ids) <= len(edge_ids) <= 6

        # get the coordinates of this vertex
        x, y = vertex_mapping[vertex_id]

        assert not np.isnan(x) and not np.isnan(y)

        self_lon_lat = grid.v_lon_lat[vertex_id]

        # compute angles of neighbor vertices
        vertices_angle = normalize_angle(
            get_angle(grid.v_lon_lat[vertex_ids] - self_lon_lat) - right_direction_angle
        )
        vertices_nbh_ids = np.around(vertices_angle / (np.pi / 3)).astype(int)
        assert np.all(
            np.fabs(vertices_angle - vertices_nbh_ids * np.pi / 3) <= angle_threshold
        )

        # compute angles of neighbor edges
        edges_angle = normalize_angle(
            get_angle(grid.e_lon_lat[edge_ids] - self_lon_lat) - right_direction_angle
        )
        edges_nbh_ids = np.around(edges_angle / (np.pi / 3)).astype(int)
        assert np.all(
            np.fabs(edges_angle - edges_nbh_ids * np.pi / 3) <= angle_threshold
        )

        # compute angles of neighbor cells
        # (we rotate the cells by 30 degrees clock-wise (`-np.pi/6`) to get the angle id)
        cells_angle = normalize_angle(
            get_angle(grid.c_lon_lat[cell_ids] - self_lon_lat)
            - right_direction_angle
            - np.pi / 6
        )
        cells_nbh_ids = np.around(cells_angle / (np.pi / 3)).astype(int)
        assert np.all(
            np.fabs(cells_angle - cells_nbh_ids * np.pi / 3) <= angle_threshold
        )

        # update right direction angle
        self_right_direction_angle = (
            np.average(vertices_angle - vertices_nbh_ids * np.pi / 3)
            + right_direction_angle
        )

        # assign coordinates to vertex neighbors that don't have a coordinate yet
        vertices_nbh_structured_coords = structured_v2v_offsets[
            vertices_nbh_ids
        ] + np.array([[x, y]], dtype=int)
        new_vertex_ids = np.all(np.isnan(vertex_mapping[vertex_ids, :]), axis=-1)
        vertex_mapping[vertex_ids[new_vertex_ids], :] = vertices_nbh_structured_coords[
            new_vertex_ids
        ]
        # check vertex neighbors that already had a coordinate, that they are consistent with the ones we computed here
        assert np.all(vertex_mapping[vertex_ids, :] == vertices_nbh_structured_coords)

        # assign coordinates to edge neighbors that don't have a coordinate yet
        edges_nbh_structured_coords = structured_v2e_offsets[edges_nbh_ids] + np.array(
            [[x, y, 0]], dtype=int
        )
        new_edge_ids = np.all(np.isnan(edge_mapping[edge_ids, :]), axis=-1)
        edge_mapping[edge_ids[new_edge_ids], :] = edges_nbh_structured_coords[
            new_edge_ids
        ]
        # check edge neighbors that already had a coordinate, that they are consistent with the ones we computed here
        assert np.all(edge_mapping[edge_ids, :] == edges_nbh_structured_coords)

        # assign coordinates to cell neighbors that don't have a coordinate yet
        cells_nbh_structured_coords = structured_v2c_offsets[cells_nbh_ids] + np.array(
            [[x, y, 0]], dtype=int
        )
        new_cell_ids = np.all(np.isnan(cell_mapping[cell_ids, :]), axis=-1)
        cell_mapping[cell_ids[new_cell_ids], :] = cells_nbh_structured_coords[
            new_cell_ids
        ]
        # check cell neighbors that already had a coordinate, that they are consistent with the ones we computed here
        assert np.all(cell_mapping[cell_ids, :] == cells_nbh_structured_coords)

        # continue bfs with vertices that have newly assigned coordinates
        # (use the updated right direction angle for them)
        return {
            (int(next_vertex_id), self_right_direction_angle)
            for next_vertex_id in vertex_ids[new_vertex_ids]
        }

    current = set()
    next = {(start_vertex, right_direction_angle)}

    while 0 < len(next):
        # swap
        current, next = next, current
        next.clear()

        for vertex_args in current:
            next.update(bfs(*vertex_args))

    assert not np.any(np.isnan(vertex_mapping))
    assert not np.any(np.isnan(edge_mapping))
    assert not np.any(np.isnan(cell_mapping))

    return GridMapping(
        vertex_mapping=vertex_mapping,
        edge_mapping=edge_mapping,
        cell_mapping=cell_mapping,
    )


def argsort_simple(
    mapping: np.ndarray,
    cmp: typing.Callable[[typing.Any, typing.Any], int],
    idx_range: typing.Tuple[typing.Optional[int], typing.Optional[int]] = (None, None),
) -> np.ndarray:
    # Sorts the first axis based on a `cmp` function within the range [start_idx:end_idx].
    # Returns the permutation array for the whole array.
    #
    # A permutation is an array `a` such that: `a[old_index] == new_index`

    total_end_idx = mapping.shape[0]
    start_idx, end_idx = idx_range

    if start_idx is None:
        start_idx = 0

    if end_idx is None:
        end_idx = total_end_idx

    ids = list(range(start_idx, end_idx))
    ids.sort(key=cmp_to_key(lambda a, b: cmp(mapping[a, :], mapping[b, :])))
    return np.concatenate(
        (np.arange(start_idx), np.array(ids), np.arange(end_idx, total_end_idx))
    )


def revert_permutation(perm: np.ndarray) -> np.ndarray:
    perm_rev = np.arange(perm.shape[0])
    perm_rev[perm] = np.copy(perm_rev)
    return perm_rev


class SimpleRowMajorSorting:
    # Provides comparison functions for mappings from `create_structured_grid_mapping`.

    @staticmethod
    def vertex_compare(a, b) -> int:
        return a[0] - b[0] if b[1] == a[1] else b[1] - a[1]

    @staticmethod
    def edge_compare(a, b) -> int:
        if a[2] == 2 and b[2] != 2:
            return b[1] - a[1] + 1 / 2
        if a[2] != 2 and b[2] == 2:
            return b[1] - a[1] - 1 / 2
        return (
            (a[2] - b[2] if a[0] == b[0] else a[0] - b[0])
            if b[1] == a[1]
            else b[1] - a[1]
        )

    @staticmethod
    def cell_compare(a, b) -> int:
        return (
            (a[2] - b[2] if a[0] == b[0] else a[0] - b[0])
            if b[1] == a[1]
            else b[1] - a[1]
        )


def assign_ij(
    positions: ndarray,
    idx_range: typing.Tuple[typing.Optional[int], typing.Optional[int]] = (None, None),
    stagger: bool = False,
):
    start_idx, end_idx = idx_range
    curx = positions[start_idx, 0]
    ij = -np.ones_like(positions)

    tresh = 0.01

    i = 0
    j = 0
    for piter in range(start_idx, end_idx - 1):
        ij[piter, :] = [i, j]

        # peek and reset if next is on next element is on next line
        nextx = positions[piter + 1, 0]
        if abs(nextx - curx) > tresh:
            j = j + 1
            i = 0

        # stagger edge lines (every other edge line has twice the amount of edges)
        i = i + 2 if stagger and j % 2 == 1 else i + 1
        curx = nextx

    ij[-1, :] = [i, j]
    return ij


def morton(
    cartesian: ndarray,
    idx_range: typing.Tuple[typing.Optional[int], typing.Optional[int]] = (None, None),
):
    start_idx, end_idx = idx_range
    n = cartesian.shape[0]
    perm = np.arange(0, n)
    for iter in range(start_idx, end_idx):
        zorder = pm.interleave(int(cartesian[iter, 0]), int(cartesian[iter, 1]))
        perm[iter] = start_idx + zorder
    return perm


def reorder_pool_folder(
    grid_set: GridSet, fix_hole_in_grid: bool, apply_morton: bool, sort_tables: bool
):
    grid_file = netCDF4.Dataset(grid_set.grid.fname + ".nc")
    grid = Grid.from_netCDF4(grid_file)

    suffix = "row-major" if not apply_morton else "morton"
    grid_set.make_data_sets(suffix)

    # the line of the right direction angle for vertex #0:
    p1 = np.array([[0.18511014, 0.79054856]])
    p2 = np.array([[0.18593181, 0.79048109]])
    right_direction_angle = np.squeeze(get_angle(p2 - p1))

    mapping = create_structured_grid_mapping(
        grid, right_direction_angle, angle_threshold=np.deg2rad(15)
    )

    v_grf = get_grf_ranges(grid, LocationType.Vertex)
    e_grf = get_grf_ranges(grid, LocationType.Edge)
    c_grf = get_grf_ranges(grid, LocationType.Cell)

    v_perm = argsort_simple(
        mapping.vertex_mapping, SimpleRowMajorSorting.vertex_compare, v_grf[0]
    )
    e_perm = argsort_simple(
        mapping.edge_mapping, SimpleRowMajorSorting.edge_compare, e_grf[0]
    )
    c_perm = argsort_simple(
        mapping.cell_mapping, SimpleRowMajorSorting.cell_compare, c_grf[0]
    )

    for cur_grid in grid_set:
        apply_permutation(cur_grid.data_set, c_perm, cur_grid.schema, LocationType.Cell)
        apply_permutation(cur_grid.data_set, e_perm, cur_grid.schema, LocationType.Edge)
        apply_permutation(
            cur_grid.data_set, v_perm, cur_grid.schema, LocationType.Vertex
        )

    if apply_morton:
        c_lonlat_rm = np.take(grid.c_lon_lat, c_perm, axis=0)
        e_lonlat_rm = np.take(grid.e_lon_lat, e_perm, axis=0)
        v_lonlat_rm = np.take(grid.v_lon_lat, v_perm, axis=0)

        # assign i,j values
        c_ij = assign_ij(c_lonlat_rm, c_grf[0])
        e_ij = assign_ij(e_lonlat_rm, e_grf[0], stagger=True)
        v_ij = assign_ij(v_lonlat_rm, v_grf[0])

        # cartesian i,j --> morton
        c_zorder = morton(c_ij, c_grf[0])
        e_zorder = morton(e_ij, e_grf[0])
        v_zorder = morton(v_ij, v_grf[0])

        c_perm = np.argsort(c_zorder)
        e_perm = np.argsort(e_zorder)
        v_perm = np.argsort(v_zorder)

        for cur_grid in grid_set:
            apply_permutation(
                cur_grid.data_set, c_perm, cur_grid.schema, LocationType.Cell
            )
            apply_permutation(
                cur_grid.data_set, e_perm, cur_grid.schema, LocationType.Edge
            )
            apply_permutation(
                cur_grid.data_set, v_perm, cur_grid.schema, LocationType.Vertex
            )

    if fix_hole_in_grid:
        fix_hole(grid_set.grid.data_set, grid_set.grid.schema)

    if sort_tables:
        sort_nbh_list(grid_set.grid.data_set, grid_set.grid.schema)

    grid_set.sync_data_sets()

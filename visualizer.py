#!/usr/bin/env python
from functools import cmp_to_key
import dataclasses
import numpy as np
from matplotlib import (
    pyplot as plt,
    patches,
    collections,
)
import netCDF4 as nc


NaN = float("nan")
DEVICE_MISSING_VALUE = -2


def chain(*chains):
    indices = chains[0]
    for chain in chains[1:]:
        indices = chain[indices]
    return indices


@dataclasses.dataclass
class Grid:
    nv: int
    ne: int
    nc: int
    c_lon_lat: np.ndarray
    v_lon_lat: np.ndarray
    e_lon_lat: np.ndarray
    v2e: np.ndarray
    v2c: np.ndarray
    e2c: np.ndarray
    e2v: np.ndarray
    c2e: np.ndarray
    c2v: np.ndarray
    v_grf: np.ndarray
    e_grf: np.ndarray
    c_grf: np.ndarray
    ncf: nc.Dataset


def load(fpath, mode="r") -> Grid:
    ncf = nc.Dataset(fpath, mode)

    get_var = lambda name: np.array(ncf.variables[name][:].T)
    #def get_var(name):
    #    if name not in ncf.variables:
    #        print(f"Gridfile '{fpath}' is missing variable '{name}' (skipping)!")
    #        return np.zeros((1, 1))
    #    return np.array(ncf.variables[name][:].T)
    get_dim = lambda name: ncf.dimensions[name].size

    return Grid(
        nv=get_dim("vertex"),
        ne=get_dim("edge"),
        nc=get_dim("cell"),
        c_lon_lat=np.stack((get_var("clon"), get_var("clat")), axis=1),
        v_lon_lat=np.stack((get_var("vlon"), get_var("vlat")), axis=1),
        e_lon_lat=np.stack((get_var("elon"), get_var("elat")), axis=1),
        v2e=get_var("edges_of_vertex") - 1,
        v2c=get_var("cells_of_vertex") - 1,
        e2c=get_var("adjacent_cell_of_edge") - 1,
        e2v=get_var("edge_vertices") - 1,
        c2e=get_var("edge_of_cell") - 1,
        c2v=get_var("vertex_of_cell") - 1,
        v_grf=np.concatenate((get_var("start_idx_v"), get_var("end_idx_v")), axis=1) - 1,
        e_grf=np.concatenate((get_var("start_idx_e"), get_var("end_idx_e")), axis=1) - 1,
        c_grf=np.concatenate((get_var("start_idx_c"), get_var("end_idx_c")), axis=1) - 1,
        ncf=ncf,
    )


def store(grid: Grid) -> None:
    ncf = grid.ncf

    ncf.variables["vlon"][:] = grid.v_lon_lat[:, 0]
    ncf.variables["vlat"][:] = grid.v_lon_lat[:, 1]
    ncf.variables["elon"][:] = grid.e_lon_lat[:, 0]
    ncf.variables["elat"][:] = grid.e_lon_lat[:, 1]
    ncf.variables["clon"][:] = grid.c_lon_lat[:, 0]
    ncf.variables["clat"][:] = grid.c_lon_lat[:, 1]
    ncf.variables["edges_of_vertex"][:] = grid.v2e.T + 1
    ncf.variables["cells_of_vertex"][:] = grid.v2c.T + 1
    ncf.variables["adjacent_cell_of_edge"][:] = grid.e2c.T + 1
    ncf.variables["edge_vertices"][:] = grid.e2v.T + 1
    ncf.variables["edge_of_cell"][:] = grid.c2e.T + 1
    ncf.variables["vertex_of_cell"][:] = grid.c2v.T + 1

    ncf.sync()


def order_around(center, points):
    centered_points = points - center
    return points[np.argsort(np.arctan2(centered_points[:, 0], centered_points[:, 1]))]


def plot_cells(ax, grid: Grid, field=None):

    if field is None:
        field = np.zeros(grid.nc)

    triangles = []
    for ci in range(grid.nc):

        vs = grid.c2v[ci]
        assert np.all((0 <= vs) & (vs < grid.nv))

        vs = grid.v_lon_lat[vs]
        triangles.append(patches.Polygon(vs))

    collection = collections.PatchCollection(triangles, edgecolors="black")
    collection.set_array(field)
    ax.add_collection(collection)

    return collection


def plot_vertices(ax, grid: Grid, field=None):

    if field is None:
        field = np.zeros(grid.nv)

    hexagons = []
    for vi in range(grid.nv):

        cs = grid.v2c[vi]
        if np.all((0 <= cs) & (cs < grid.nc)):
            points = grid.c_lon_lat[cs]
            points = order_around(grid.v_lon_lat[vi].reshape(1, 2), points)
        else:
            # some of our neighbors are outside the field
            # we have to draw a partial hexagon
            cs = cs[cs != DEVICE_MISSING_VALUE]
            es = grid.v2e[vi]
            es = es[es != DEVICE_MISSING_VALUE]

            vertex = grid.v_lon_lat[vi].reshape(1, 2)
            points = np.concatenate((grid.c_lon_lat[cs], grid.e_lon_lat[es], vertex), axis=0)
            center = np.average(points, axis=0)
            points = order_around(center, points)

        hexagons.append(patches.Polygon(points))

    collection = collections.PatchCollection(hexagons, edgecolors="black")
    collection.set_array(field)
    ax.add_collection(collection)

    return collection


def plot_edges(ax, grid: Grid, field=None):

    if field is None:
        field = np.zeros(grid.nv)

    diamonds = []
    for ei in range(grid.ne):

        vs = grid.e2v[ei]
        cs = grid.e2c[ei]

        if cs[1] == DEVICE_MISSING_VALUE:
            # one of our neighboring cells is outside the grid
            # we just draw half a diamond
            cs = cs[:1]

        assert np.all((0 <= vs) & (vs < grid.nv)) and np.all((0 <= cs) & (cs < grid.nc))

        points = np.concatenate((grid.v_lon_lat[vs], grid.c_lon_lat[cs]), axis=0)
        points = order_around(grid.e_lon_lat[ei].reshape(1, 2), points)
        diamonds.append(patches.Polygon(points))

    collection = collections.PatchCollection(diamonds, edgecolors="black")
    collection.set_array(field)
    ax.add_collection(collection)

    return collection


def get_grf_regions(grid: Grid, location_type: str = 'c'):

    if location_type == 'v':
        n = grid.nv
        start, end = grid.v_grf[:, 0], grid.v_grf[:, 1]
    elif location_type == 'e':
        n = grid.ne
        start, end = grid.e_grf[:, 0], grid.e_grf[:, 1]
    elif location_type == 'c':
        n = grid.nc
        start, end = grid.c_grf[:, 0], grid.c_grf[:, 1]
    else:
        raise ValueError

    valid_regions = start <= end
    start = start[valid_regions]
    end = end[valid_regions]

    assert np.min(start) == 0
    assert np.max(end) < n

    # There's something very weird going on:
    # Some few vertices/edges/cells (at the end) aren't in any valid region,
    # but without them, there will be a hole in the compute domain.
    # We fix this by assigning them to region `0` by default.

    field = np.zeros(n)
    region = 0
    for s, e in zip(start, end):
        field[s:e + 1] = region
        region += 1

    return field


def normalize_angle(angle):
    return np.fmod(angle, 2*np.pi)


def get_angle(p):
    return np.arctan2(p[:, 1], p[:, 0])


def rotate(points, angle, origin=np.array([[0, 0]])):
    points = points - origin
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle) ]
    ])
    points = (rotation_matrix @ points.T).T
    return points + origin


def apply_perms(fields, perms):

    for perm in perms:
        for field in fields:
            field[:] = field[perm, :]


class UnsupportedPentagonException(Exception):
    pass


def neighbor_array_to_set(array):
    nbhs = np.unique(array)
    return nbhs[nbhs != DEVICE_MISSING_VALUE]
    #neighbors = set(array.flatten())
    #neighbors.discard(DEVICE_MISSING_VALUE)  # ignore invalid neighbors
    #return neighbors

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
structured_v2v_offsets = np.array([
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
], dtype=int)

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
structured_v2e_offsets = np.array([
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
], dtype=int)
# (for the cells, we shift the rays 15 degrees counter clock-wise)
structured_v2c_offsets = np.array([
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
], dtype=int)


def order_structured(
    grid: Grid,
    right_direction_angle,
    start_vertex=None,
    angle_threshold=np.deg2rad(30)
):
    # doesn't support pentagons!

    if start_vertex is None:
        start_vertex = 0

    if isinstance(right_direction_angle, np.ndarray):
        right_direction_angle = float(right_direction_angle)

    vertex_mapping = np.full((grid.nv, 2), NaN)
    edge_mapping = np.full((grid.ne, 3), NaN)
    cell_mapping = np.full((grid.nc, 3), NaN)

    vertex_mapping[start_vertex] = [0, 0]

    def bfs(vertex_id, right_direction_angle):

        cell_ids = neighbor_array_to_set(grid.v2c[vertex_id])
        edge_ids = neighbor_array_to_set(grid.v2e[vertex_id])
        vertex_ids = neighbor_array_to_set(grid.e2v[edge_ids])
        #vertex_ids.discard(vertex_id)
        vertex_ids = vertex_ids[vertex_ids != vertex_id]
        #vertex_ids = list(vertex_ids)

        if len(edge_ids) == 5 and len(cell_ids) == 5:
            raise UnsupportedPentagonException

        assert len(edge_ids) == len(cell_ids) == 6 or len(cell_ids) + 1 == len(edge_ids)
        assert len(vertex_ids) == len(edge_ids)
        assert 0 < len(cell_ids) <= len(edge_ids) <= 6

        x, y = vertex_mapping[vertex_id]

        assert not np.isnan(x) and not np.isnan(y)

        self_lon_lat = grid.v_lon_lat[vertex_id]

        vertices_angle = np.fmod(get_angle(grid.v_lon_lat[vertex_ids] - self_lon_lat) - right_direction_angle, np.pi*2)
        vertices_nbh_ids = np.around(vertices_angle/(np.pi/3)).astype(int)
        assert np.all(np.fabs(vertices_angle - vertices_nbh_ids*np.pi/3) <= angle_threshold)

        edges_angle = np.fmod(get_angle(grid.e_lon_lat[edge_ids] - self_lon_lat) - right_direction_angle, np.pi*2)
        edges_nbh_ids = np.around(edges_angle/(np.pi/3)).astype(int)
        assert np.all(np.fabs(edges_angle - edges_nbh_ids*np.pi/3) <= angle_threshold)

        cells_angle = np.fmod(get_angle(grid.c_lon_lat[cell_ids] - self_lon_lat) - right_direction_angle - np.pi/6, np.pi*2)
        cells_nbh_ids = np.around(cells_angle/(np.pi/3)).astype(int)
        assert np.all(np.fabs(cells_angle - cells_nbh_ids*np.pi/3) <= angle_threshold)

        self_right_direction_angle = np.average(vertices_angle - vertices_nbh_ids*np.pi/3) + right_direction_angle

        vertices_nbh_structured_coords = structured_v2v_offsets[vertices_nbh_ids] + np.array([[x, y]], dtype=int)
        new_vertex_ids = np.all(np.isnan(vertex_mapping[vertex_ids, :]), axis=-1)
        vertex_mapping[vertex_ids[new_vertex_ids], :] = vertices_nbh_structured_coords[new_vertex_ids]
        assert np.all(vertex_mapping[vertex_ids, :] == vertices_nbh_structured_coords)

        # FIXME: updated mappings also for edges & cells
        edges_nbh_structured_coords = structured_v2e_offsets[edges_nbh_ids] + np.array([[x, y, 0]], dtype=int)
        new_edge_ids = np.all(np.isnan(edge_mapping[edge_ids, :]), axis=-1)
        edge_mapping[edge_ids[new_edge_ids], :] = edges_nbh_structured_coords[new_edge_ids]
        assert np.all(edge_mapping[edge_ids, :] == edges_nbh_structured_coords)

        cells_nbh_structured_coords = structured_v2c_offsets[cells_nbh_ids] + np.array([[x, y, 0]], dtype=int)
        new_cell_ids = np.all(np.isnan(cell_mapping[cell_ids, :]), axis=-1)
        cell_mapping[cell_ids[new_cell_ids], :] = cells_nbh_structured_coords[new_cell_ids]
        assert np.all(cell_mapping[cell_ids, :] == cells_nbh_structured_coords)

        return {(int(next_vertex_id), self_right_direction_angle) for next_vertex_id in vertex_ids[new_vertex_ids]}

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

    return vertex_mapping, edge_mapping, cell_mapping


def argsort(mapping, cmp):
    ids = list(range(mapping.shape[0]))
    ids.sort(key=cmp_to_key(lambda a, b: cmp(mapping[a, :], mapping[b, :])))
    return np.array(ids)



def main():
    grid = load("grid.nc")
    parent_grid = load("grid.parent.nc")

    grid_modified = None
    #grid = grid_modified = load("grid.benchmark.row-major.nc", "r+")
    #grid = load("grid.benchmark.row-major.nc", "r+")

    # the line we want to align horizontally for this particular grid
    # the line for one of the upper rows
    p1 = np.array([[0.098574, 0.8372010]])
    p2 = np.array([[0.187451, 0.8316751]])
    angle = np.squeeze(get_angle(p2 - p1))

    origin = np.min(grid.v_lon_lat, axis=0)

    grid.v_lon_lat = rotate(grid.v_lon_lat, -angle, origin = origin)
    grid.e_lon_lat = rotate(grid.e_lon_lat, -angle, origin = origin)
    grid.c_lon_lat = rotate(grid.c_lon_lat, -angle, origin = origin)

    #v_perm1 = np.argsort(grid.v_lon_lat[:, 0])
    #v_perm2 = np.argsort(grid.v_lon_lat[v_perm1, 1], kind='stable')

    #e_perm1 = np.argsort(grid.e_lon_lat[:, 0])
    #e_perm2 = np.argsort(grid.e_lon_lat[e_perm1, 1], kind='stable')

    #c_perm1 = np.argsort(grid.c_lon_lat[:, 0])
    #c_perm2 = np.argsort(grid.c_lon_lat[c_perm1, 1], kind='stable')

    #apply_perms((grid.v_lon_lat, grid.v2e, grid.v2c), (v_perm1, v_perm2))
    #apply_perms((grid.e_lon_lat, grid.e2v, grid.e2c), (e_perm1, e_perm2))
    #apply_perms((grid.c_lon_lat, grid.c2v, grid.c2e), (c_perm1, c_perm2))

    # the line of the right direction angle for vertex #0:
    p1 = np.array([[0.18511014, 0.79054856]])
    p2 = np.array([[0.18593181, 0.79048109]])
    angle = np.squeeze(get_angle(p2 - p1))

    vertex_mapping, edge_mapping, cell_mapping = order_structured(grid, angle, angle_threshold=np.deg2rad(15))

    v_grf = get_grf_regions(grid, "v")
    v_compute_domain = np.nonzero(v_grf == 0)
    v_compute_domain_start = np.min(v_compute_domain)
    v_compute_domain_end = np.max(v_compute_domain) + 1
    assert v_compute_domain_end == grid.nv

    e_grf = get_grf_regions(grid, "e")
    e_compute_domain = np.nonzero(e_grf == 0)
    e_compute_domain_start = np.min(e_compute_domain)
    e_compute_domain_end = np.max(e_compute_domain) + 1
    assert e_compute_domain_end == grid.ne

    c_grf = get_grf_regions(grid, "c")
    c_compute_domain = np.nonzero(c_grf == 0)
    c_compute_domain_start = np.min(c_compute_domain)
    c_compute_domain_end = np.max(c_compute_domain) + 1
    assert c_compute_domain_end == grid.nc

    v_perm = argsort(vertex_mapping[v_compute_domain_start:v_compute_domain_end], lambda a, b: a[0] - b[0] if b[1] == a[1] else b[1] - a[1])
    v_perm = np.concatenate((np.arange(v_compute_domain_start), v_perm + v_compute_domain_start))
    v_perm_rev = np.arange(grid.nv)
    v_perm_rev[v_perm] = np.copy(v_perm_rev)

    #e_perm = argsort(edge_mapping, lambda a, b: (a[2] - b[2] if a[0] == b[0] else a[0] - b[0]) if b[1] == a[1] else b[1] - a[1])
    def e_cmp(a, b):
        if a[2] == 2 and b[2] != 2:
            return b[1] - a[1] + 1/2
        if a[2] != 2 and b[2] == 2:
            return b[1] - a[1] - 1/2
        return (a[2] - b[2] if a[0] == b[0] else a[0] - b[0]) if b[1] == a[1] else b[1] - a[1]
    e_perm = argsort(edge_mapping[e_compute_domain_start:e_compute_domain_end], e_cmp)
    e_perm = np.concatenate((np.arange(e_compute_domain_start), e_perm + e_compute_domain_start))
    e_perm_rev = np.arange(grid.ne)
    e_perm_rev[e_perm] = np.copy(e_perm_rev)

    c_perm = argsort(cell_mapping[c_compute_domain_start:c_compute_domain_end], lambda a, b: (a[2] - b[2] if a[0] == b[0] else a[0] - b[0]) if b[1] == a[1] else b[1] - a[1])
    c_perm = np.concatenate((np.arange(c_compute_domain_start), c_perm + c_compute_domain_start))
    c_perm_rev = np.arange(grid.nc)
    c_perm_rev[c_perm] = np.copy(c_perm_rev)

    # reorder vertex fields
    apply_perms((grid.v_lon_lat, grid.v2e, grid.v2c), (v_perm,))
    # fix the indices of the neighbor tables
    v2e_missing_values = grid.v2e == DEVICE_MISSING_VALUE
    grid.v2e = e_perm_rev[grid.v2e]
    grid.v2e[v2e_missing_values] = DEVICE_MISSING_VALUE
    # fix the indices of the neighbor tables
    v2c_missing_values = grid.v2c == DEVICE_MISSING_VALUE
    grid.v2c = c_perm_rev[grid.v2c]
    grid.v2c[v2c_missing_values] = DEVICE_MISSING_VALUE
    # sort the neighbors to improve coalescing in the neighbor tables
    grid.v2e[grid.v2e == DEVICE_MISSING_VALUE] = grid.ne + 1
    grid.v2e = np.sort(grid.v2e)
    grid.v2e[grid.v2e == grid.ne + 1] = DEVICE_MISSING_VALUE
    grid.v2c[grid.v2c == DEVICE_MISSING_VALUE] = grid.nc + 1
    grid.v2c = np.sort(grid.v2c)
    grid.v2c[grid.v2c == grid.nc + 1] = DEVICE_MISSING_VALUE

    # reorder edge fields
    apply_perms((grid.e_lon_lat, grid.e2v, grid.e2c), (e_perm,))
    # fix the indices of the neighbor tables (edges never have missing cells)
    grid.e2v = v_perm_rev[grid.e2v]
    # fix the indices of the neighbor tables
    e2c_missing_values = grid.e2c == DEVICE_MISSING_VALUE
    grid.e2c = c_perm_rev[grid.e2c]
    grid.e2c[e2c_missing_values] = DEVICE_MISSING_VALUE
    # sort the neighbors to improve coalescing in the neighbor tables
    grid.e2v = np.sort(grid.e2v)
    grid.e2c[grid.e2c == DEVICE_MISSING_VALUE] = grid.nc + 1
    grid.e2c = np.sort(grid.e2c)
    grid.e2c[grid.e2c == grid.nc + 1] = DEVICE_MISSING_VALUE

    # reorder cell fields
    apply_perms((grid.c_lon_lat, grid.c2v, grid.c2e), (c_perm,))
    # fix the indices of the neighbor tables (cells never have missing values)
    grid.c2v = v_perm_rev[grid.c2v]
    grid.c2e = e_perm_rev[grid.c2e]
    # sort the neighbors to improve coalescing in the neighbor tables
    grid.c2v = np.sort(grid.c2v)
    grid.c2e = np.sort(grid.c2e)

    #grid = load("./lateral_boundary.grid.nc")

    fig, ax = plt.subplots()

    take_branch = 'c'

    if 'v' == take_branch:
        #vertices = plot_vertices(ax, grid, field=grid.ncf.variables["parent_vertex_index"][:][v_perm])
        vertices = plot_vertices(ax, grid, field=np.arange(grid.nv))
        #vertices = plot_vertices(ax, grid, field=v_grf)

        # transparent edges
        vertices.set_edgecolor((0, 0, 0, 0))
        fig.colorbar(vertices, ax=ax)

    elif 'e' == take_branch:
        #edges = plot_edges(ax, grid, field=grid.ncf.variables["parent_edge_index"][:][e_perm])
        edges = plot_edges(ax, grid, field=np.arange(grid.ne))
        #edges = plot_edges(ax, grid, field=e_grf)

        # transparent edges
        edges.set_edgecolor((0, 0, 0, 0))
        fig.colorbar(edges, ax=ax)

    elif 'c' == take_branch:
        #cells = plot_cells(ax, grid, field=grid.ncf.variables["parent_cell_index"][:][c_perm])
        cells = plot_cells(ax, grid, field=np.arange(grid.nc))
        #cells = plot_cells(ax, grid, field=c_grf)

        # transparent edges
        cells.set_edgecolor((0, 0, 0, 0))
        fig.colorbar(cells, ax=ax)

    elif 'n' == take_branch:
        pass # this is the plot none of the branch...
    else:
        assert False

    #ax.plot(grid.v_lon_lat[v_compute_domain_start:, 0], grid.v_lon_lat[v_compute_domain_start:, 1], 'o-')
    #ax.plot(grid.e_lon_lat[e_compute_domain_start:, 0], grid.e_lon_lat[e_compute_domain_start:, 1], 'o-')
    #ax.plot(grid.c_lon_lat[c_compute_domain_start:, 0], grid.c_lon_lat[c_compute_domain_start:, 1], 'o-')

    if grid_modified is not None:
        store(grid_modified)

    ax.autoscale()
    plt.show()


if __name__ == "__main__":
    main()

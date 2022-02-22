from typing import Grid, LocationType, DEVICE_MISSING_VALUE
import numpy as np
from matplotlib import (
    pyplot as plt,
    patches,
    collections,
)

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

    collection = collections.PatchCollection(triangles)
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

    collection = collections.PatchCollection(hexagons)
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

    collection = collections.PatchCollection(diamonds)
    collection.set_array(field)
    ax.add_collection(collection)

    return collection

def plot_grid(grid: Grid, location: LocationType):
      # reload the grid after we've modified it
    # parent_grid_modified_file.sync()
    # grid = Grid.from_netCDF4(parent_grid_modified_file)
    
    # fig, ax = plt.subplots()

    # take_branch: typing.Optional[LocationType] = LocationType.Vertex

    # if take_branch is LocationType.Vertex:
    #     vertices = plot_vertices(ax, grid, field=np.arange(grid.nv))
    #     # transparent edges
    #     vertices.set_edgecolor((0, 0, 0, 0))
    #     fig.colorbar(vertices, ax=ax)

    # elif take_branch is LocationType.Edge:
    #     edges = plot_edges(ax, grid, field=np.arange(grid.ne))
    #     # transparent edges
    #     edges.set_edgecolor((0, 0, 0, 0))
    #     fig.colorbar(edges, ax=ax)

    # if take_branch is LocationType.Cell:
    #     cells = plot_cells(ax, grid, field=np.arange(grid.nc))
    #     # transparent edges
    #     cells.set_edgecolor((0, 0, 0, 0))
    #     fig.colorbar(cells, ax=ax)

    # elif take_branch is None:
    #     pass # this is the plot none branch
    # else:
    #     assert False

    #ax.plot(grid.v_lon_lat[range_to_slice(v_grf[0]), 0], grid.v_lon_lat[range_to_slice(v_grf[0]), 1], 'o-')
    #ax.plot(grid.e_lon_lat[range_to_slice(e_grf[0]), 0], grid.e_lon_lat[range_to_slice(e_grf[0]), 1], 'o-')
    #ax.plot(grid.c_lon_lat[range_to_slice(c_grf[0]), 0], grid.c_lon_lat[range_to_slice(c_grf[0]), 1], 'o-')

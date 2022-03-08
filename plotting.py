from grid_types import Grid, DEVICE_MISSING_VALUE
from location_type import LocationType
import numpy as np
from matplotlib import (
    pyplot as plt,
    patches,
    collections,
)
import netCDF4


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
            points = np.concatenate(
                (grid.c_lon_lat[cs], grid.e_lon_lat[es], vertex), axis=0
            )
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


def plot_grid(fname: str, location: LocationType):
    grid_file = netCDF4.Dataset(fname)
    grid = Grid.from_netCDF4(grid_file)

    fig, ax = plt.subplots()

    if location is LocationType.Vertex:
        vertices = plot_vertices(ax, grid, field=np.arange(grid.nv))
        # transparent edges
        vertices.set_edgecolor((0, 0, 0, 0))
        fig.colorbar(vertices, ax=ax)

    elif location is LocationType.Edge:
        edges = plot_edges(ax, grid, field=np.arange(grid.ne))
        # transparent edges
        edges.set_edgecolor((0, 0, 0, 0))
        fig.colorbar(edges, ax=ax)

    elif location is LocationType.Cell:
        cells = plot_cells(ax, grid, field=np.arange(grid.nc))
        # transparent edges
        cells.set_edgecolor((0, 0, 0, 0))
        fig.colorbar(cells, ax=ax)

    # plot "trace" of elements
    # ax.plot(grid.v_lon_lat[range_to_slice(v_grf[0]), 0], grid.v_lon_lat[range_to_slice(v_grf[0]), 1], 'o-')
    # ax.plot(grid.e_lon_lat[range_to_slice(e_grf[0]), 0], grid.e_lon_lat[range_to_slice(e_grf[0]), 1], 'o-')
    # ax.plot(grid.c_lon_lat[range_to_slice(c_grf[0]), 0], grid.c_lon_lat[range_to_slice(c_grf[0]), 1], 'o-')

    # plot some onions
    # ax.plot(grid.c_lon_lat[0:814, 0], grid.c_lon_lat[0:814, 1], "o")
    # ax.plot(grid.c_lon_lat[815:1613, 0], grid.c_lon_lat[815:1613, 1], "o")
    # ax.plot(grid.c_lon_lat[1613:2397, 0], grid.c_lon_lat[1613:2397, 1], "o")
    # ax.plot(grid.c_lon_lat[2398:3160, 0], grid.c_lon_lat[2398:3160, 1], "o")
    # ax.plot(grid.c_lon_lat[3160:3908, 0], grid.c_lon_lat[3160:3908, 1], "o")
    # ax.plot(grid.c_lon_lat[3908:4709, 0], grid.c_lon_lat[3908:4709, 1], "o") #not an onion anymore

    # plot lateral boundary condition
    # latbc_file = netCDF4.Dataset("igfff00000000_lbc.nc")
    # grid_latbc = GridLBC.from_netCDF4_lbc(latbc_file)
    # ax.plot(grid_latbc.c_lon_lat[:, 0], grid_latbc.c_lon_lat[:, 1], "o")

    # plot the location of the cell hole
    # ax.plot(grid.c_lon_lat[-8:, 0], grid.c_lon_lat[-8:, 1], "ro")

    ax.autoscale()
    plt.show()


if __name__ == "__main__":
    plot_grid("grid_row-major.nc", LocationType.Cell)

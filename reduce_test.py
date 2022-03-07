import netCDF4
import numpy as np
import math

from grid_types import Grid, DEVICE_MISSING_VALUE

from matplotlib import (
    pyplot as plt,
    patches,
    collections,
)


class SpatialHasher:
    def _insert(self, px, py, idx):
        i = math.floor((px - self.lox) / self.dx)
        j = math.floor((py - self.loy) / self.dy)
        self.map[i][j].append(idx)

    def __init__(self, points: np.ndarray):

        self.lox = np.min(points[:, 0])
        self.loy = np.min(points[:, 1])
        self.hix = np.max(points[:, 0])
        self.hiy = np.max(points[:, 1])

        self.n = math.floor(math.sqrt(len(points[:, 1])))

        self.dx = (self.hix - self.lox) / self.n
        self.dy = (self.hix - self.lox) / self.n

        self.map = [[[] for i in range(self.n + 1)] for j in range(self.n + 1)]

        for i, point in enumerate(points):
            self._insert(point[0], point[1], i)

        self.points = points

    def find(self, px, py):
        i = max(math.floor((px - self.lox) / self.dx), 0)
        j = max(math.floor((py - self.loy) / self.dy), 0)

        for ii in range(max(i - 1, 0), min(i + 1, self.n + 1)):
            for jj in range(max(j - 1, 0), min(j + 1, self.n + 1)):

                for idx in self.map[ii][jj]:
                    if np.allclose([px, py], self.points[idx]):
                        return idx
        return None


def get_map(grid_a_lon_lat: np.ndarray, grid_b_lon_lat: np.ndarray):
    v_hasher = SpatialHasher(grid_b_lon_lat)
    a_to_b = []
    for i in range(0, len(grid_a_lon_lat)):
        vi = grid_a_lon_lat[i, :]
        idx = v_hasher.find(vi[0], vi[1])
        assert idx is not None
        a_to_b.append(idx)
    return a_to_b


def reduce_test(grid_a_x2y, grid_b_x2y, map, grid_a_field, grid_b_field):
    errors = 0
    correct = 0

    assert grid_a_x2y.shape == grid_b_x2y.shape

    num_el = grid_a_x2y.shape[0]
    num_nbh = grid_a_x2y.shape[1]

    for i in range(0, num_el):

        sum_a = 0
        for j in range(0, num_nbh):
            nbh = grid_a_x2y[i, j]
            if nbh == DEVICE_MISSING_VALUE:
                continue
            sum_a += grid_a_field[nbh][0]

        sum_b = 0
        for j in range(0, num_nbh):
            nbh = grid_b_x2y[map[i], j]
            if nbh == DEVICE_MISSING_VALUE:
                continue
            sum_b += grid_b_field[nbh][0]

        if not np.isclose(sum_a, sum_b):
            errors += 1
        else:
            correct += 1

    return (correct, errors)


def test_permutation(fname_a, fname_b):

    grid_file_a = netCDF4.Dataset(fname_a)
    grid_a = Grid.from_netCDF4(grid_file_a)

    grid_file_b = netCDF4.Dataset(fname_b)
    grid_b = Grid.from_netCDF4(grid_file_b)

    assert grid_a.nv == grid_b.nv
    assert grid_a.nc == grid_b.nc
    assert grid_a.ne == grid_b.ne

    map_c = get_map(grid_a.c_lon_lat, grid_b.c_lon_lat)
    map_v = get_map(grid_a.v_lon_lat, grid_b.v_lon_lat)
    map_e = get_map(grid_a.e_lon_lat, grid_b.e_lon_lat)

    correct, errors = reduce_test(
        grid_a.v2e, grid_b.v2e, map_v, grid_a.e_lon_lat, grid_b.e_lon_lat
    )
    print(f"v2e: corr: {correct} err: {errors}")
    correct, errors = reduce_test(
        grid_a.v2c, grid_b.v2c, map_v, grid_a.c_lon_lat, grid_b.c_lon_lat
    )
    print(f"v2c: corr: {correct} err: {errors}")
    correct, errors = reduce_test(
        grid_a.e2c, grid_b.e2c, map_e, grid_a.c_lon_lat, grid_b.c_lon_lat
    )
    print(f"e2c: corr: {correct} err: {errors}")
    correct, errors = reduce_test(
        grid_a.e2v, grid_b.e2v, map_e, grid_a.v_lon_lat, grid_b.v_lon_lat
    )
    print(f"e2v: corr: {correct} err: {errors}")
    correct, errors = reduce_test(
        grid_a.c2e, grid_b.c2e, map_c, grid_a.e_lon_lat, grid_b.e_lon_lat
    )
    print(f"c2e: corr: {correct} err: {errors}")
    correct, errors = reduce_test(
        grid_a.c2v, grid_b.c2v, map_c, grid_a.v_lon_lat, grid_b.v_lon_lat
    )
    print(f"c2v: corr: {correct} err: {errors}")


def test_init_cond(fname_a, fname_b, ifname_a, ifname_b):
    grid_file_a = netCDF4.Dataset(fname_a)
    grid_a = Grid.from_netCDF4(grid_file_a)

    grid_file_b = netCDF4.Dataset(fname_b)
    grid_b = Grid.from_netCDF4(grid_file_b)

    init_cond_a = netCDF4.Dataset(ifname_a)
    init_cond_b = netCDF4.Dataset(ifname_b)

    assert grid_a.nc == grid_b.nc

    map_c = get_map(grid_a.c_lon_lat, grid_b.c_lon_lat)

    clon_a = init_cond_a.variables["clon"]
    clat_a = init_cond_a.variables["clat"]

    clon_b = init_cond_b.variables["clon"]
    clat_b = init_cond_b.variables["clat"]

    T_a = init_cond_a.variables["T"][0, 89, :]
    T_b = init_cond_b.variables["T"][0, 89, :]

    result_grid = netCDF4.Dataset(
        "/scratch/mroeth/icon-nwp-new/build/experiments/mch_ch_r04b09/mch_ch_r04b09_atm_3d_ml_20180921T000000Z.nc"
    )

    clon_res = result_grid.variables["clon"]
    clat_res = result_grid.variables["clat"]
    T_res = result_grid.variables["ta"][0, 64, :]

    fig, (ax1, ax2) = plt.subplots(1, 2)
    test = np.copy(T_b[:])
    tx = np.copy(clon_res[:])
    ty = np.copy(clat_res[:])
    # ax.plot(clon_res, clat_res)
    ax1.scatter(tx, ty, c=T_res)
    ax1.autoscale()
    ax2.scatter(tx, ty, c=T_a)
    ax2.autoscale()
    plt.show()

    for i in range(0, grid_a.nc):
        assert np.isclose(clon_a[i], clon_b[map_c[i]])
        assert np.isclose(clat_a[i], clat_b[map_c[i]])

        assert np.isclose(clon_a[i], clon_res[map_c[i]])
        assert np.isclose(clat_a[i], clat_res[map_c[i]])

        assert np.isclose(T_a[i], T_b[map_c[i]])


if __name__ == "__main__":
    test_permutation("grid.nc", "grid_row-major.nc")
    # test_init_cond(
    #     "grid.nc", "grid_row-major.nc", "igfff00000000.nc", "igfff00000000_row-major.nc"
    # )

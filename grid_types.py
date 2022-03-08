from __future__ import annotations

from schemas import (
    GridScheme,
    ICON_grid_schema,
    ICON_grid_schema_extpar,
    ICON_grid_schema_ic,
    ICON_grid_schema_lat_grid,
)

import numpy as np
import dataclasses
import shutil
import netCDF4


DEVICE_MISSING_VALUE = -2


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

    @staticmethod
    def from_netCDF4(ncf) -> Grid:
        def get_dim(name) -> int:
            return ncf.dimensions[name].size

        def get_var(name) -> np.ndarray:
            return np.array(ncf.variables[name][:].T)

        return Grid(
            nv=get_dim("vertex"),
            ne=get_dim("edge"),
            nc=get_dim("cell"),
            c_lon_lat=np.stack((get_var("clon"), get_var("clat")), axis=1),
            v_lon_lat=np.stack((get_var("vlon"), get_var("vlat")), axis=1),
            e_lon_lat=np.stack((get_var("elon"), get_var("elat")), axis=1),
            v2e=get_var("edges_of_vertex") - 1,
            v2c=get_var("cells_of_vertex") - 1,
            e2v=get_var("edge_vertices") - 1,
            e2c=get_var("adjacent_cell_of_edge") - 1,
            c2v=get_var("vertex_of_cell") - 1,
            c2e=get_var("edge_of_cell") - 1,
            v_grf=np.concatenate((get_var("start_idx_v"), get_var("end_idx_v")), axis=1)
            - 1,
            e_grf=np.concatenate((get_var("start_idx_e"), get_var("end_idx_e")), axis=1)
            - 1,
            c_grf=np.concatenate((get_var("start_idx_c"), get_var("end_idx_c")), axis=1)
            - 1,
        )


@dataclasses.dataclass
class GridFile:
    fname: str
    folder: str
    schema: GridScheme
    data_set = None


class GridSet:
    def __init__(self, pool_folder: str):
        self.pool_folder = pool_folder

        self.grid = GridFile(
            fname="grid",
            folder="grids/ch_r04b09",
            schema=ICON_grid_schema
        )
        self.lateral_boundary_grid = GridFile(
            fname="lateral_boundary.grid",
            folder="input/ch_r04b09",
            schema=ICON_grid_schema_lat_grid
        )
        self.initial_conditions = GridFile(
            fname="igfff00000000",
            folder="input/ch_r04b09",
            schema=ICON_grid_schema_ic
        )
        self.extpar = GridFile(
            fname="extpar",
            folder="grids/ch_r04b09",
            schema=ICON_grid_schema_extpar
        )

        self._grids = [
            self.grid,
            self.lateral_boundary_grid,
            self.initial_conditions,
            self.extpar,
        ]

    def __iter__(self):
        for each in self._grids:
            yield each

    def copy_to_staging(self):
        for grid_file in self:
            shutil.copy(
                f"{self.pool_folder}/{grid_file.folder}/{grid_file.fname}_icon.nc",
                f"{grid_file.fname}.nc",
            )

    def copy_to_pool(self, suffix: str = ""):
        for grid_file in self:
            shutil.copy(
                f"{grid_file.fname}_{suffix}.nc",
                f"{self.pool_folder}/{grid_file.folder}/{grid_file.fname}.nc",
            )

    def make_data_sets(self, suffix: str = ""):
        for grid_file in self:
            shutil.copy(f"{grid_file.fname}.nc", f"{grid_file.fname}_{suffix}.nc")
            grid_file.data_set = netCDF4.Dataset(f"{grid_file.fname}_{suffix}.nc", "r+")

    def sync_data_sets(self):
        for grid_file in self:
            grid_file.data_set.sync()

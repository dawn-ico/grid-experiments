#!/usr/bin/env python
from __future__ import annotations

import shutil

from reordering import reorder_parent_grid, reorder_pool_folder
from plotting import plot_grid
from grid_types import LocationType
from reduce_test import test_permutation


def row_major_permutation_pool(reorder_parent_grid: bool):
    shutil.copy("../my_pool/data/ICON/mch/grids/ch_r04b09/grid_icon.nc", "grid.nc")
    shutil.copy(
        "../my_pool/data/ICON/mch/grids/ch_r04b09/grid.parent_icon.nc", "grid.parent.nc"
    )
    shutil.copy("../my_pool/data/ICON/mch/grids/ch_r04b09/extpar_icon.nc", "extpar.nc")
    shutil.copy(
        "../my_pool/data/ICON/mch/input/ch_r04b09/lateral_boundary.grid_icon.nc",
        "lateral_boundary.grid.nc",
    )
    shutil.copy(
        "../my_pool/data/ICON/mch/input/ch_r04b09/igfff00000000_icon.nc",
        "igfff00000000.nc",
    )

    reorder_pool_folder(reorder_parent=reorder_parent_grid)

    shutil.copy("grid_row-major.nc", "../my_pool/data/ICON/mch/grids/ch_r04b09/grid.nc")
    shutil.copy(
        "grid.parent_row-major.nc",
        "../my_pool/data/ICON/mch/grids/ch_r04b09/grid.parent.nc",
    )
    shutil.copy(
        "lateral_boundary.grid_row-major.nc",
        "../my_pool/data/ICON/mch/input/ch_r04b09/lateral_boundary.grid.nc",
    )
    shutil.copy(
        "igfff00000000_row-major.nc",
        "../my_pool/data/ICON/mch/input/ch_r04b09/igfff00000000.nc",
    )
    shutil.copy(
        "extpar_row-major.nc", "../my_pool/data/ICON/mch/grids/ch_r04b09/extpar_icon.nc"
    )


if __name__ == "__main__":
    # row_major_permutation_pool(reorder_parent_grid=False)
    plot_grid(
        "grid.nc",
        LocationType.Cell,
    )
    # test_permutation("grid.nc", "grid_row-major.nc")

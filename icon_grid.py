#!/usr/bin/env python
from __future__ import annotations

import shutil

from reordering import reorder_pool_folder, fix_hole_reference
from plotting import plot_grid
from grid_types import LocationType
from reduce_test import test_permutation


def row_major_permutation_pool(reorder_parent_grid: bool, fix_hole: bool):
    shutil.copy("../my_pool/data/ICON/mch/grids/ch_r04b09/grid_icon.nc", "grid.nc")
    if reorder_parent_grid:
        shutil.copy(
            "../my_pool/data/ICON/mch/grids/ch_r04b09/grid.parent_icon.nc",
            "grid.parent.nc",
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

    reorder_pool_folder(reorder_parent=reorder_parent_grid, fix_hole=fix_hole)

    shutil.copy("grid_row-major.nc", "../my_pool/data/ICON/mch/grids/ch_r04b09/grid.nc")
    if reorder_parent_grid:
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
        "extpar_row-major.nc", "../my_pool/data/ICON/mch/grids/ch_r04b09/extpar.nc"
    )


def fix_grid_in_reference():
    shutil.copy("../my_pool/data/ICON/mch/grids/ch_r04b09/grid_icon.nc", "grid.nc")

    fix_hole_reference()

    shutil.copy("grid_fixed.nc", "../my_pool/data/ICON/mch/grids/ch_r04b09/grid.nc")


if __name__ == "__main__":
    # fix_grid_in_reference()
    row_major_permutation_pool(reorder_parent_grid=False, fix_hole=True)
    # plot_grid(
    #     "grid.nc",
    #     LocationType.Cell,
    # )
    # test_permutation("grid.nc", "grid_row-major.nc")

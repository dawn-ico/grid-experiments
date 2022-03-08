#!/usr/bin/env python
from __future__ import annotations

import shutil

from reordering import reorder_pool_folder, fix_hole
from grid_types import GridSet


def row_major_permutation(grid_set: GridSet, fix_hole_in_grid: bool):
    grid_set.copy_to_staging()
    reorder_pool_folder(grid_set, fix_hole_in_grid=fix_hole_in_grid)
    grid_set.copy_to_pool("row-major")


def reset_to_icon(grid_set: GridSet, fix_hole_in_grid: bool):
    grid_set.copy_to_staging()
    grid_set.make_data_sets()
    if fix_hole_in_grid:
        fix_hole(grid_set.grid.data_set, grid_set.grid.schema)
    grid_set.copy_to_pool()


# FIXME add argument parsing
if __name__ == "__main__":
    grid_set = GridSet("/scratch/mroeth/icon-nwp-new/my_pool/data/ICON/mch")

    # row_major_permutation(grid_set, fix_hole_in_grid=True)
    reset_to_icon(grid_set, fix_hole_in_grid=True)

#!/usr/bin/env python
from __future__ import annotations

import argparse

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="reorder icon grids")
    parser.add_argument(
        "--row_major", action="store_true", help="place row major grids in pool folder"
    )
    parser.add_argument(
        "--icon", action="store_true", help="place icon grids in pool folder"
    )
    parser.add_argument(
        "--fix_hole", action="store_true", help="fix hole present in icon grid"
    )
    args = parser.parse_args()

    if not args.icon and not args.row_major:
        parser.error("either chose icon or row major grids")

    grid_set = GridSet("/scratch/mroeth/icon-nwp-new/my_pool/data/ICON/mch")

    if args.icon:
        reset_to_icon(grid_set, fix_hole_in_grid=args.fix_hole)
    else:
        row_major_permutation(grid_set, fix_hole_in_grid=args.fix_hole)

#!/usr/bin/env python
from __future__ import annotations

import argparse

from reordering import reorder_pool_folder, fix_hole
from grid_types import GridSet


def row_major_permutation(
    grid_set: GridSet, fix_hole_in_grid: bool, apply_morton: bool, sort_nbh: bool
):
    grid_set.copy_to_staging()
    reorder_pool_folder(
        grid_set,
        fix_hole_in_grid=fix_hole_in_grid,
        apply_morton=apply_morton,
        sort_tables=sort_nbh,
    )
    grid_set.copy_to_pool("morton" if apply_morton else "row-major")


def reset_to_icon(grid_set: GridSet, fix_hole_in_grid: bool):
    grid_set.copy_to_staging()
    grid_set.make_data_sets("tmp")
    if fix_hole_in_grid:
        fix_hole(grid_set.grid.data_set, grid_set.grid.schema)
    grid_set.sync_data_sets()
    grid_set.copy_to_pool("tmp")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="reorder icon grids")
    parser.add_argument(
        "--row_major", action="store_true", help="place row major grids in pool folder"
    )
    parser.add_argument(
        "--morton", action="store_true", help="place z ordered grids in pool folder"
    )
    parser.add_argument(
        "--icon", action="store_true", help="place icon grids in pool folder"
    )
    parser.add_argument(
        "--fix_hole", action="store_true", help="fix hole present in icon grid"
    )
    parser.add_argument("--sort_tables", action="store_true", help="sort nbh tables")
    args = parser.parse_args()

    arg_flags = [args.icon, args.row_major, args.morton]
    if not any(arg_flags):
        parser.error("chose an ordering method")
    if sum(arg_flags) > 1:
        parser.error("chose only one ordering method")

    grid_set = GridSet("my_pool/data/ICON/mch")

    if args.icon:
        reset_to_icon(grid_set, fix_hole_in_grid=args.fix_hole)

    if args.row_major:
        row_major_permutation(
            grid_set,
            fix_hole_in_grid=args.fix_hole,
            apply_morton=False,
            sort_nbh=args.sort_tables,
        )

    if args.morton:
        row_major_permutation(
            grid_set,
            fix_hole_in_grid=args.fix_hole,
            apply_morton=True,
            sort_nbh=args.sort_tables,
        )

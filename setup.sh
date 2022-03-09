#!/bin/bash
mkdir -p my_pool/data/ICON/mch/grids/ch_r04b09/
mkdir -p my_pool/data/ICON/mch/input/ch_r04b09/
cp /scratch/jenkins/icon/pool/data/ICON/mch/grids/ch_r04b09/grid.nc my_pool/data/ICON/mch/grids/ch_r04b09/grid_icon.nc   
cp /scratch/jenkins/icon/pool/data/ICON/mch/grids/ch_r04b09/grid.parent.nc my_pool/data/ICON/mch/grids/ch_r04b09/grid.parent.nc
cp /scratch/jenkins/icon/pool/data/ICON/mch/grids/ch_r04b09/extpar.nc my_pool/data/ICON/mch/grids/ch_r04b09/extpar_icon.nc
cp /scratch/jenkins/icon/pool/data/ICON/mch/input/ch_r04b09/lateral_boundary.grid.nc my_pool/data/ICON/mch/input/ch_r04b09/lateral_boundary.grid_icon.nc
cp /scratch/jenkins/icon/pool/data/ICON/mch/input/ch_r04b09/igfff00000000.nc my_pool/data/ICON/mch/input/ch_r04b09/igfff00000000_icon.nc
cp /scratch/jenkins/icon/pool/data/ICON/mch/input/ch_r04b09/igfff00000000_lbc.nc my_pool/data/ICON/mch/input/ch_r04b09/igfff00000000_lbc.nc
cp /scratch/jenkins/icon/pool/data/ICON/mch/input/ch_r04b09/igfff00030000_lbc.nc my_pool/data/ICON/mch/input/ch_r04b09/igfff00030000_lbc.nc
cp /scratch/jenkins/icon/pool/data/ICON/mch/input/ch_r04b09/map_file.latbc my_pool/data/ICON/mch/input/ch_r04b09/map_file.latbc
# Grid Experiments

Contains a few scripts to experiment with the memory layout of the icon grid

* `icon_grid.py` to transform the memory layout
* `plotting.py` to visualize the memory layout of a given icon mesh
* `reduce_test.py` to determine whether two given icon meshes are the same, up to a layout transform

# Workflow to Transform an icon grid

Currently, only experiment `ch_r04b09` is supported

* compile icon and generate the runscripts as usual
* in the icon root folder, clone this repo `git clone git@github.com:dawn-ico/grid-experiments.git`
* run the setup script `cd grid-experiments && bash setup.sh`. This copies the contents of `/scratch/jenkins/icon/pool/data/ICON/mch/grids/ch_r04b09/` into a local pool folder called `my_pool`
* change the `icon_data_rootFolder` in `exp.mch_ch_r04b09.run` from the jenkins pool to the local pool, i.e. `icon_data_rootFolder="/scratch/jenkins/icon/pool/data/ICON/" > icon_data_rootFolder="/scratch/mroeth/icon-test/grid-experiments/my_pool/data/ICON"`
* use `icon_grid.py` to perform the desired layout transformation, e.g. `icon_grid.py --row-major --fix-hole`. This manipulates the icon grids in `my_pool`. Experiment `exp.mch_ch_r04b09.run` now uses the transformed grids.

**NOTE** probtest will only succeed if the option `--fix-hole` is used for both layouts to be tested. This is due to an icon bug. 
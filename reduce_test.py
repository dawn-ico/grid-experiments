def test_permutation():
    shutil.copy("../my_pool/data/ICON/mch/grids/ch_r04b09/grid_icon.nc", "grid.nc")

    grid_file = netCDF4.Dataset("grid.nc")
    grid = Grid.from_netCDF4(grid_file)
    shutil.copy("./grid.nc", "./grid_row-major.nc")
    grid_modified_file = netCDF4.Dataset("./grid_row-major.nc", "r+")

    p1 = np.array([[0.18511014, 0.79054856]])
    p2 = np.array([[0.18593181, 0.79048109]])
    right_direction_angle = np.squeeze(get_angle(p2 - p1))

    mapping = create_structured_grid_mapping(grid, right_direction_angle, angle_threshold=np.deg2rad(15))

    v_grf = get_grf_ranges(grid, LocationType.Vertex)
    e_grf = get_grf_ranges(grid, LocationType.Edge)
    c_grf = get_grf_ranges(grid, LocationType.Cell)

    v_perm = argsort_simple(mapping.vertex_mapping, SimpleRowMajorSorting.vertex_compare, v_grf[0])
    e_perm = argsort_simple(mapping.edge_mapping, SimpleRowMajorSorting.edge_compare, e_grf[0])
    c_perm = argsort_simple(mapping.cell_mapping, SimpleRowMajorSorting.cell_compare, c_grf[0])

    # c_perm = np.array(range(0,grid.nc))

    apply_permutation(grid_modified_file, c_perm, ICON_grid_schema_dbg, LocationType.Cell)
    apply_permutation(grid_modified_file, e_perm, ICON_grid_schema_dbg, LocationType.Edge)
    apply_permutation(grid_modified_file, v_perm, ICON_grid_schema_dbg, LocationType.Vertex)

    # perm_dict = {LocationType.Cell : c_perm, LocationType.Edge : e_perm, LocationType.Vertex : v_perm}
    # apply_permutation_nbh_tables(grid_modified_file, perm_dict, ICON_grid_schema_dbg)

    grid_modified_file.sync()

    grid_file = netCDF4.Dataset("grid_row-major.nc")
    grid_modified = Grid.from_netCDF4(grid_file)    

    test_vertex(grid, grid_modified, v_perm, c_perm)
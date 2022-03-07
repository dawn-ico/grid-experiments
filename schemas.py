from __future__ import annotations
import typing
import dataclasses

from grid_types import LocationType


@dataclasses.dataclass
class FieldDescriptor:
    location_type: typing.Optional[LocationType] = None
    indexes_into: typing.Optional[LocationType] = None
    # FIXME: `primary_axis` is quite weird...
    # some fields that have a `location_type` are transposed
    # (only needed with `location_type` and if the field is more than one dimensional)
    primary_axis: typing.Optional[int] = None


GridScheme = typing.Dict[str, FieldDescriptor]


def make_schema(**fields: FieldDescriptor) -> GridScheme:
    return fields


ICON_grid_schema = make_schema(
    # FIXME: make sure this is complete
    ###########################################################################
    # grid topology and lon/lat coordinates
    ###########################################################################
    vlon=FieldDescriptor(location_type=LocationType.Vertex),
    vlat=FieldDescriptor(location_type=LocationType.Vertex),
    # v2v
    vertices_of_vertex=FieldDescriptor(
        primary_axis=1,
        location_type=LocationType.Vertex,
        indexes_into=LocationType.Vertex,
    ),
    # v2e
    edges_of_vertex=FieldDescriptor(
        primary_axis=1,
        location_type=LocationType.Vertex,
        indexes_into=LocationType.Edge,
    ),
    # v2c
    cells_of_vertex=FieldDescriptor(
        primary_axis=1,
        location_type=LocationType.Vertex,
        indexes_into=LocationType.Cell,
    ),
    elon=FieldDescriptor(location_type=LocationType.Edge),
    elat=FieldDescriptor(location_type=LocationType.Edge),
    # e2v
    edge_vertices=FieldDescriptor(
        primary_axis=1,
        location_type=LocationType.Edge,
        indexes_into=LocationType.Vertex,
    ),
    # e2c
    adjacent_cell_of_edge=FieldDescriptor(
        primary_axis=1,
        location_type=LocationType.Edge,
        indexes_into=LocationType.Cell,
    ),
    clon=FieldDescriptor(location_type=LocationType.Cell),
    clat=FieldDescriptor(location_type=LocationType.Cell),
    # c2v
    vertex_of_cell=FieldDescriptor(
        primary_axis=1,
        location_type=LocationType.Cell,
        indexes_into=LocationType.Vertex,
    ),
    # c2e
    edge_of_cell=FieldDescriptor(
        primary_axis=1,
        location_type=LocationType.Cell,
        indexes_into=LocationType.Edge,
    ),
    # c2c
    neighbor_cell_index=FieldDescriptor(
        primary_axis=1,
        location_type=LocationType.Cell,
        indexes_into=LocationType.Cell,
    ),
    ###########################################################################
    # other vertex fields
    ###########################################################################
    cartesian_x_vertices=FieldDescriptor(location_type=LocationType.Vertex),
    cartesian_y_vertices=FieldDescriptor(location_type=LocationType.Vertex),
    cartesian_z_vertices=FieldDescriptor(location_type=LocationType.Vertex),
    dual_area=FieldDescriptor(location_type=LocationType.Vertex),
    longitude_vertices=FieldDescriptor(location_type=LocationType.Vertex),
    latitude_vertices=FieldDescriptor(location_type=LocationType.Vertex),
    dual_area_p=FieldDescriptor(location_type=LocationType.Vertex),
    vlon_vertices=FieldDescriptor(primary_axis=0, location_type=LocationType.Vertex),
    vlat_vertices=FieldDescriptor(primary_axis=0, location_type=LocationType.Vertex),
    # FIXME: will this work just like this?
    edge_orientation=FieldDescriptor(primary_axis=1, location_type=LocationType.Vertex),
    refin_v_ctrl=FieldDescriptor(location_type=LocationType.Vertex),
    parent_vertex_index=FieldDescriptor(location_type=LocationType.Vertex),
    ###########################################################################
    # other edge fields
    ###########################################################################
    lon_edge_centre=FieldDescriptor(location_type=LocationType.Edge),
    lat_edge_centre=FieldDescriptor(location_type=LocationType.Edge),
    edge_length=FieldDescriptor(location_type=LocationType.Edge),
    edge_cell_distance=FieldDescriptor(primary_axis=1, location_type=LocationType.Edge),
    dual_edge_length=FieldDescriptor(location_type=LocationType.Edge),
    edge_vert_distance=FieldDescriptor(primary_axis=1, location_type=LocationType.Edge),
    zonal_normal_primal_edge=FieldDescriptor(location_type=LocationType.Edge),
    meridional_normal_primal_edge=FieldDescriptor(location_type=LocationType.Edge),
    zonal_normal_dual_edge=FieldDescriptor(location_type=LocationType.Edge),
    meridional_normal_dual_edge=FieldDescriptor(location_type=LocationType.Edge),
    edge_system_orientation=FieldDescriptor(location_type=LocationType.Edge),
    elon_vertices=FieldDescriptor(primary_axis=0, location_type=LocationType.Edge),
    elat_vertices=FieldDescriptor(primary_axis=0, location_type=LocationType.Edge),
    quadrilateral_area=FieldDescriptor(location_type=LocationType.Edge),
    refin_e_ctrl=FieldDescriptor(location_type=LocationType.Edge),
    parent_edge_index=FieldDescriptor(location_type=LocationType.Edge),
    ###########################################################################
    # other cell fields
    ###########################################################################
    cell_area=FieldDescriptor(location_type=LocationType.Cell),
    lon_cell_centre=FieldDescriptor(location_type=LocationType.Cell),
    lat_cell_centre=FieldDescriptor(location_type=LocationType.Cell),
    cell_area_p=FieldDescriptor(location_type=LocationType.Cell),
    orientation_of_normal=FieldDescriptor(
        primary_axis=1, location_type=LocationType.Cell
    ),
    clon_vertices=FieldDescriptor(
        primary_axis=0,
        location_type=LocationType.Cell,
    ),
    clat_vertices=FieldDescriptor(
        primary_axis=0,
        location_type=LocationType.Cell,
    ),
    parent_cell_index=FieldDescriptor(location_type=LocationType.Cell),
    refin_c_ctrl=FieldDescriptor(location_type=LocationType.Cell),
)

ICON_grid_schema_dbg = make_schema(
    #   double clon(cell=20340);
    #   :units = "radian";
    #   :standard_name = "grid_longitude";
    #   :bounds = "clon_vertices";
    clon=FieldDescriptor(location_type=LocationType.Cell),
    # double clat(cell=20340);
    #   :units = "radian";
    #   :standard_name = "grid_latitude";
    #   :bounds = "clat_vertices";
    clat=FieldDescriptor(location_type=LocationType.Cell),
    # double vlon(vertex=10376);
    #   :units = "radian";
    #   :standard_name = "grid_longitude";
    #   :bounds = "vlon_vertices";
    vlon=FieldDescriptor(location_type=LocationType.Vertex),
    # double vlat(vertex=10376);
    #   :units = "radian";
    #   :standard_name = "grid_latitude";
    #   :bounds = "vlat_vertices";
    vlat=FieldDescriptor(location_type=LocationType.Vertex),
    # double cartesian_x_vertices(vertex=10376);
    cartesian_x_vertices=FieldDescriptor(location_type=LocationType.Vertex),
    # double cartesian_y_vertices(vertex=10376);
    cartesian_y_vertices=FieldDescriptor(location_type=LocationType.Vertex),
    # double cartesian_z_vertices(vertex=10376);
    cartesian_z_vertices=FieldDescriptor(location_type=LocationType.Vertex),
    # double elon(edge=30715);
    #   :units = "radian";
    #   :standard_name = "grid_longitude";
    #   :bounds = "elon_vertices";
    elon=FieldDescriptor(location_type=LocationType.Edge),
    # double elat(edge=30715);
    #   :units = "radian";
    #   :standard_name = "grid_latitude";
    #   :bounds = "elat_vertices";
    elat=FieldDescriptor(location_type=LocationType.Edge),
    # double cell_area(cell=20340);
    #   :coordinates = "clon clat";
    cell_area=FieldDescriptor(location_type=LocationType.Cell),
    # double dual_area(vertex=10376);
    dual_area=FieldDescriptor(location_type=LocationType.Vertex),
    # double lon_cell_centre(cell=20340);
    lon_cell_centre=FieldDescriptor(location_type=LocationType.Cell),
    # double lat_cell_centre(cell=20340);
    lat_cell_centre=FieldDescriptor(location_type=LocationType.Cell),
    # double longitude_vertices(vertex=10376);
    longitude_vertices=FieldDescriptor(location_type=LocationType.Vertex),
    # double latitude_vertices(vertex=10376);
    latitude_vertices=FieldDescriptor(location_type=LocationType.Vertex),
    # double lon_edge_centre(edge=30715);
    lon_edge_centre=FieldDescriptor(location_type=LocationType.Edge),
    # double lat_edge_centre(edge=30715);
    lat_edge_centre=FieldDescriptor(location_type=LocationType.Edge),
    # int edge_of_cell(nv=3, cell=20340);
    edge_of_cell=FieldDescriptor(
        location_type=LocationType.Cell, indexes_into=LocationType.Edge, primary_axis=1
    ),
    # int vertex_of_cell(nv=3, cell=20340);
    vertex_of_cell=FieldDescriptor(
        location_type=LocationType.Cell,
        indexes_into=LocationType.Vertex,
        primary_axis=1,
    ),
    # int adjacent_cell_of_edge(nc=2, edge=30715);
    adjacent_cell_of_edge=FieldDescriptor(
        location_type=LocationType.Edge, indexes_into=LocationType.Cell, primary_axis=1
    ),
    # int edge_vertices(nc=2, edge=30715);
    edge_vertices=FieldDescriptor(
        location_type=LocationType.Edge,
        indexes_into=LocationType.Vertex,
        primary_axis=1,
    ),
    # int cells_of_vertex(ne=6, vertex=10376);
    cells_of_vertex=FieldDescriptor(
        location_type=LocationType.Vertex,
        indexes_into=LocationType.Cell,
        primary_axis=1,
    ),
    # int edges_of_vertex(ne=6, vertex=10376);
    edges_of_vertex=FieldDescriptor(
        location_type=LocationType.Vertex,
        indexes_into=LocationType.Edge,
        primary_axis=1,
    ),
    # int vertices_of_vertex(ne=6, vertex=10376);
    vertices_of_vertex=FieldDescriptor(
        location_type=LocationType.Vertex,
        indexes_into=LocationType.Vertex,
        primary_axis=1,
    ),
    # double cell_area_p(cell=20340);
    cell_area_p=FieldDescriptor(location_type=LocationType.Cell),
    # double dual_area_p(vertex=10376);
    dual_area_p=FieldDescriptor(location_type=LocationType.Vertex),
    # double edge_length(edge=30715);
    edge_length=FieldDescriptor(location_type=LocationType.Edge),
    # double edge_cell_distance(nc=2, edge=30715);
    edge_cell_distance=FieldDescriptor(location_type=LocationType.Edge, primary_axis=1),
    # double dual_edge_length(edge=30715);
    dual_edge_length=FieldDescriptor(location_type=LocationType.Edge),
    # double edge_vert_distance(nc=2, edge=30715);
    edge_vert_distance=FieldDescriptor(location_type=LocationType.Edge, primary_axis=1),
    # double zonal_normal_primal_edge(edge=30715);
    zonal_normal_primal_edge=FieldDescriptor(location_type=LocationType.Edge),
    # double meridional_normal_primal_edge(edge=30715);
    meridional_normal_primal_edge=FieldDescriptor(location_type=LocationType.Edge),
    # double zonal_normal_dual_edge(edge=30715);
    zonal_normal_dual_edge=FieldDescriptor(location_type=LocationType.Edge),
    # double meridional_normal_dual_edge(edge=30715);
    meridional_normal_dual_edge=FieldDescriptor(location_type=LocationType.Edge),
    # int orientation_of_normal(nv=3, cell=20340);
    orientation_of_normal=FieldDescriptor(
        location_type=LocationType.Cell, primary_axis=1
    ),
    # double clon_vertices(cell=20340, nv=3);
    clon_vertices=FieldDescriptor(location_type=LocationType.Cell, primary_axis=0),
    # double clat_vertices(cell=20340, nv=3);
    clat_vertices=FieldDescriptor(location_type=LocationType.Cell, primary_axis=0),
    # double elon_vertices(edge=30715, no=4);
    elon_vertices=FieldDescriptor(location_type=LocationType.Edge, primary_axis=0),
    # double elat_vertices(edge=30715, no=4);
    elat_vertices=FieldDescriptor(location_type=LocationType.Edge, primary_axis=0),
    # double vlon_vertices(vertex=10376, ne=6);
    vlon_vertices=FieldDescriptor(location_type=LocationType.Vertex, primary_axis=0),
    # double vlat_vertices(vertex=10376, ne=6);
    vlat_vertices=FieldDescriptor(location_type=LocationType.Vertex, primary_axis=0),
    # double quadrilateral_area(edge=30715);
    #   :coordinates = "elon elat";
    quadrilateral_area=FieldDescriptor(location_type=LocationType.Edge),
    # int parent_cell_index(cell=20340);
    #   :long_name = "parent cell index";
    #   :cdi = "ignore";
    parent_cell_index=FieldDescriptor(location_type=LocationType.Cell),
    # int neighbor_cell_index(nv=3, cell=20340);
    neighbor_cell_index=FieldDescriptor(
        location_type=LocationType.Cell, indexes_into=LocationType.Cell, primary_axis=1
    ),
    # int edge_orientation(ne=6, vertex=10376);
    edge_orientation=FieldDescriptor(location_type=LocationType.Vertex, primary_axis=1),
    # int edge_system_orientation(edge=30715);
    edge_system_orientation=FieldDescriptor(location_type=LocationType.Edge),
    # int refin_c_ctrl(cell=20340);
    #   :long_name = "refinement control flag for cells";
    #   :cdi = "ignore";
    refin_c_ctrl=FieldDescriptor(location_type=LocationType.Cell),
    # int start_idx_c(max_chdom=1, cell_grf=14);
    #   :long_name = "list of start indices for each refinement control level for cells";
    #   :cdi = "ignore";
    # int end_idx_c(max_chdom=1, cell_grf=14);
    #   :long_name = "list of end indices for each refinement control level for cells";
    #   :cdi = "ignore";
    end_idx_c=FieldDescriptor(),
    # int refin_e_ctrl(edge=30715);
    #   :long_name = "refinement control flag for edges";
    #   :cdi = "ignore";
    refin_e_ctrl=FieldDescriptor(location_type=LocationType.Edge),
    # int start_idx_e(max_chdom=1, edge_grf=24);
    #   :long_name = "list of start indices for each refinement control level for edges";
    #   :cdi = "ignore";
    # int end_idx_e(max_chdom=1, edge_grf=24);
    #   :long_name = "list of end indices for each refinement control level for edges";
    #   :cdi = "ignore";
    end_idx_e=FieldDescriptor(),
    # int refin_v_ctrl(vertex=10376);
    #   :long_name = "refinement control flag for vertices";
    #   :cdi = "ignore";
    refin_v_ctrl=FieldDescriptor(location_type=LocationType.Vertex),
    # int start_idx_v(max_chdom=1, vert_grf=13);
    #   :long_name = "list of start indices for each refinement control level for vertices";
    #   :cdi = "ignore";
    # int end_idx_v(max_chdom=1, vert_grf=13);
    #   :long_name = "list of end indices for each refinement control level for vertices";
    #   :cdi = "ignore";
    end_idx_v=FieldDescriptor(),
    # int parent_edge_index(edge=30715);
    #   :long_name = "parent edge index";
    #   :cdi = "ignore";
    parent_edge_index=FieldDescriptor(location_type=LocationType.Edge),
    # int parent_vertex_index(vertex=10376);
    parent_vertex_index=FieldDescriptor(location_type=LocationType.Vertex),
)

ICON_grid_schema_parent = make_schema(
    #   double clon(cell=20340);
    #   :units = "radian";
    #   :standard_name = "grid_longitude";
    #   :bounds = "clon_vertices";
    clon=FieldDescriptor(location_type=LocationType.Cell),
    # double clat(cell=20340);
    #   :units = "radian";
    #   :standard_name = "grid_latitude";
    #   :bounds = "clat_vertices";
    clat=FieldDescriptor(location_type=LocationType.Cell),
    # double vlon(vertex=10376);
    #   :units = "radian";
    #   :standard_name = "grid_longitude";
    #   :bounds = "vlon_vertices";
    vlon=FieldDescriptor(location_type=LocationType.Vertex),
    # double vlat(vertex=10376);
    #   :units = "radian";
    #   :standard_name = "grid_latitude";
    #   :bounds = "vlat_vertices";
    vlat=FieldDescriptor(location_type=LocationType.Vertex),
    # double cartesian_x_vertices(vertex=10376);
    cartesian_x_vertices=FieldDescriptor(location_type=LocationType.Vertex),
    # double cartesian_y_vertices(vertex=10376);
    cartesian_y_vertices=FieldDescriptor(location_type=LocationType.Vertex),
    # double cartesian_z_vertices(vertex=10376);
    cartesian_z_vertices=FieldDescriptor(location_type=LocationType.Vertex),
    # double elon(edge=30715);
    #   :units = "radian";
    #   :standard_name = "grid_longitude";
    #   :bounds = "elon_vertices";
    elon=FieldDescriptor(location_type=LocationType.Edge),
    # double elat(edge=30715);
    #   :units = "radian";
    #   :standard_name = "grid_latitude";
    #   :bounds = "elat_vertices";
    elat=FieldDescriptor(location_type=LocationType.Edge),
    # double cell_area(cell=20340);
    #   :coordinates = "clon clat";
    cell_area=FieldDescriptor(location_type=LocationType.Cell),
    # double dual_area(vertex=10376);
    dual_area=FieldDescriptor(location_type=LocationType.Vertex),
    # double lon_cell_centre(cell=20340);
    lon_cell_centre=FieldDescriptor(location_type=LocationType.Cell),
    # double lat_cell_centre(cell=20340);
    lat_cell_centre=FieldDescriptor(location_type=LocationType.Cell),
    # double longitude_vertices(vertex=10376);
    longitude_vertices=FieldDescriptor(location_type=LocationType.Vertex),
    # double latitude_vertices(vertex=10376);
    latitude_vertices=FieldDescriptor(location_type=LocationType.Vertex),
    # double lon_edge_centre(edge=30715);
    lon_edge_centre=FieldDescriptor(location_type=LocationType.Edge),
    # double lat_edge_centre(edge=30715);
    lat_edge_centre=FieldDescriptor(location_type=LocationType.Edge),
    # int edge_of_cell(nv=3, cell=20340);
    edge_of_cell=FieldDescriptor(
        location_type=LocationType.Cell, indexes_into=LocationType.Edge, primary_axis=1
    ),
    # int vertex_of_cell(nv=3, cell=20340);
    vertex_of_cell=FieldDescriptor(
        location_type=LocationType.Cell,
        indexes_into=LocationType.Vertex,
        primary_axis=1,
    ),
    # int adjacent_cell_of_edge(nc=2, edge=30715);
    adjacent_cell_of_edge=FieldDescriptor(
        location_type=LocationType.Edge, indexes_into=LocationType.Cell, primary_axis=1
    ),
    # int edge_vertices(nc=2, edge=30715);
    edge_vertices=FieldDescriptor(
        location_type=LocationType.Edge,
        indexes_into=LocationType.Vertex,
        primary_axis=1,
    ),
    # int cells_of_vertex(ne=6, vertex=10376);
    cells_of_vertex=FieldDescriptor(
        location_type=LocationType.Vertex,
        indexes_into=LocationType.Cell,
        primary_axis=1,
    ),
    # int edges_of_vertex(ne=6, vertex=10376);
    edges_of_vertex=FieldDescriptor(
        location_type=LocationType.Vertex,
        indexes_into=LocationType.Edge,
        primary_axis=1,
    ),
    # int vertices_of_vertex(ne=6, vertex=10376);
    vertices_of_vertex=FieldDescriptor(
        location_type=LocationType.Vertex,
        indexes_into=LocationType.Vertex,
        primary_axis=1,
    ),
    # double cell_area_p(cell=20340);
    cell_area_p=FieldDescriptor(location_type=LocationType.Cell),
    # double dual_area_p(vertex=10376);
    dual_area_p=FieldDescriptor(location_type=LocationType.Vertex),
    # double edge_length(edge=30715);
    edge_length=FieldDescriptor(location_type=LocationType.Edge),
    # double edge_cell_distance(nc=2, edge=30715);
    edge_cell_distance=FieldDescriptor(location_type=LocationType.Edge, primary_axis=1),
    # double dual_edge_length(edge=30715);
    dual_edge_length=FieldDescriptor(location_type=LocationType.Edge),
    # double edge_vert_distance(nc=2, edge=30715);
    edge_vert_distance=FieldDescriptor(location_type=LocationType.Edge, primary_axis=1),
    # double zonal_normal_primal_edge(edge=30715);
    zonal_normal_primal_edge=FieldDescriptor(location_type=LocationType.Edge),
    # double meridional_normal_primal_edge(edge=30715);
    meridional_normal_primal_edge=FieldDescriptor(location_type=LocationType.Edge),
    # double zonal_normal_dual_edge(edge=30715);
    zonal_normal_dual_edge=FieldDescriptor(location_type=LocationType.Edge),
    # double meridional_normal_dual_edge(edge=30715);
    meridional_normal_dual_edge=FieldDescriptor(location_type=LocationType.Edge),
    # int orientation_of_normal(nv=3, cell=20340);
    orientation_of_normal=FieldDescriptor(
        location_type=LocationType.Cell, primary_axis=1
    ),
    # double clon_vertices(cell=20340, nv=3);
    clon_vertices=FieldDescriptor(location_type=LocationType.Cell, primary_axis=0),
    # double clat_vertices(cell=20340, nv=3);
    clat_vertices=FieldDescriptor(location_type=LocationType.Cell, primary_axis=0),
    # double elon_vertices(edge=30715, no=4);
    elon_vertices=FieldDescriptor(location_type=LocationType.Edge, primary_axis=0),
    # double elat_vertices(edge=30715, no=4);
    elat_vertices=FieldDescriptor(location_type=LocationType.Edge, primary_axis=0),
    # double vlon_vertices(vertex=10376, ne=6);
    vlon_vertices=FieldDescriptor(location_type=LocationType.Vertex, primary_axis=0),
    # double vlat_vertices(vertex=10376, ne=6);
    vlat_vertices=FieldDescriptor(location_type=LocationType.Vertex, primary_axis=0),
    # double quadrilateral_area(edge=30715);
    #   :coordinates = "elon elat";
    quadrilateral_area=FieldDescriptor(location_type=LocationType.Edge),
    # int neighbor_cell_index(nv=3, cell=20340);
    neighbor_cell_index=FieldDescriptor(
        location_type=LocationType.Cell, indexes_into=LocationType.Cell, primary_axis=1
    ),
    # int edge_orientation(ne=6, vertex=10376);
    edge_orientation=FieldDescriptor(location_type=LocationType.Vertex, primary_axis=1),
    # int edge_system_orientation(edge=30715);
    edge_system_orientation=FieldDescriptor(location_type=LocationType.Edge),
    # int refin_c_ctrl(cell=20340);
    #   :long_name = "refinement control flag for cells";
    #   :cdi = "ignore";
    refin_c_ctrl=FieldDescriptor(location_type=LocationType.Cell),
    # int start_idx_c(max_chdom=1, cell_grf=14);
    #   :long_name = "list of start indices for each refinement control level for cells";
    #   :cdi = "ignore";
    # int end_idx_c(max_chdom=1, cell_grf=14);
    #   :long_name = "list of end indices for each refinement control level for cells";
    #   :cdi = "ignore";
    # int refin_e_ctrl(edge=30715);
    #   :long_name = "refinement control flag for edges";
    #   :cdi = "ignore";
    refin_e_ctrl=FieldDescriptor(location_type=LocationType.Edge),
    # int start_idx_e(max_chdom=1, edge_grf=24);
    #   :long_name = "list of start indices for each refinement control level for edges";
    #   :cdi = "ignore";
    # int end_idx_e(max_chdom=1, edge_grf=24);
    #   :long_name = "list of end indices for each refinement control level for edges";
    #   :cdi = "ignore";
    # int refin_v_ctrl(vertex=10376);
    #   :long_name = "refinement control flag for vertices";
    #   :cdi = "ignore";
    refin_v_ctrl=FieldDescriptor(location_type=LocationType.Vertex),
    # int start_idx_v(max_chdom=1, vert_grf=13);
    #   :long_name = "list of start indices for each refinement control level for vertices";
    #   :cdi = "ignore";
    # int end_idx_v(max_chdom=1, vert_grf=13);
    #   :long_name = "list of end indices for each refinement control level for vertices";
    #   :cdi = "ignore";
)

ICON_grid_schema_lat_grid = make_schema(
    # double clon(cell=9868);
    #   :units = "radian";
    #   :standard_name = "grid_longitude";
    #   :bounds = "clon_vertices";
    clon=FieldDescriptor(location_type=LocationType.Cell),
    # double clat(cell=9868);
    #   :units = "radian";
    #   :standard_name = "grid_latitude";
    #   :bounds = "clat_vertices";
    clat=FieldDescriptor(location_type=LocationType.Cell),
    # double vlon(vertex=5287);
    #   :units = "radian";
    #   :standard_name = "grid_longitude";
    #   :bounds = "vlon_vertices";
    vlon=FieldDescriptor(location_type=LocationType.Vertex),
    # double vlat(vertex=5287);
    #   :units = "radian";
    #   :standard_name = "grid_latitude";
    #   :bounds = "vlat_vertices";
    vlat=FieldDescriptor(location_type=LocationType.Vertex),
    # double elon(edge=15155);
    #   :units = "radian";
    #   :standard_name = "grid_longitude";
    #   :bounds = "elon_vertices";
    elon=FieldDescriptor(location_type=LocationType.Edge),
    # double elat(edge=15155);
    #   :units = "radian";
    #   :standard_name = "grid_latitude";
    #   :bounds = "elat_vertices";
    elat=FieldDescriptor(location_type=LocationType.Edge),
    # double longitude_vertices(vertex=5287);
    #   :coordinates = "vlon vlat";
    longitude_vertices=FieldDescriptor(location_type=LocationType.Vertex),
    # double latitude_vertices(vertex=5287);
    #   :coordinates = "vlon vlat";
    latitude_vertices=FieldDescriptor(location_type=LocationType.Vertex),
    # double clon_vertices(cell=9868, nv=3);
    clon_vertices=FieldDescriptor(
        primary_axis=0,
        location_type=LocationType.Cell,
    ),
    # double clat_vertices(cell=9868, nv=3);
    clat_vertices=FieldDescriptor(
        primary_axis=0,
        location_type=LocationType.Cell,
    ),
    # double vlon_vertices(vertex=5287, ne=6);
    vlon_vertices=FieldDescriptor(
        primary_axis=0,
        location_type=LocationType.Vertex,
    ),
    # double vlat_vertices(vertex=5287, ne=6);
    vlat_vertices=FieldDescriptor(
        primary_axis=0,
        location_type=LocationType.Vertex,
    ),
    # double cell_area(cell=9868);
    #   :coordinates = "clon clat";
    cell_area=FieldDescriptor(location_type=LocationType.Cell),
    # double lon_edge_centre(edge=15155);
    #   :cdi = "ignore";
    lon_edge_centre=FieldDescriptor(location_type=LocationType.Edge),
    # double lat_edge_centre(edge=15155);
    #   :cdi = "ignore";
    lat_edge_centre=FieldDescriptor(location_type=LocationType.Edge),
    # int edge_of_cell(nv=3, cell=9868);
    #   :cdi = "ignore";
    edge_of_cell=FieldDescriptor(
        primary_axis=1,
        location_type=LocationType.Cell,
        indexes_into=LocationType.Edge,
    ),
    # int vertex_of_cell(nv=3, cell=9868);
    #   :cdi = "ignore";
    vertex_of_cell=FieldDescriptor(
        primary_axis=1,
        location_type=LocationType.Cell,
        indexes_into=LocationType.Vertex,
    ),
    # int adjacent_cell_of_edge(nc=2, edge=15155);
    #   :cdi = "ignore";
    adjacent_cell_of_edge=FieldDescriptor(
        primary_axis=1,
        location_type=LocationType.Edge,
        indexes_into=LocationType.Cell,
    ),
    # int edge_vertices(nc=2, edge=15155);
    #   :cdi = "ignore";
    edge_vertices=FieldDescriptor(
        primary_axis=1,
        location_type=LocationType.Edge,
        indexes_into=LocationType.Vertex,
    ),
    # int cells_of_vertex(ne=6, vertex=5287);
    #   :cdi = "ignore";
    cells_of_vertex=FieldDescriptor(
        primary_axis=1,
        location_type=LocationType.Vertex,
        indexes_into=LocationType.Cell,
    ),
    # double edge_length(edge=15155);
    #   :coordinates = "elon elat";
    edge_length=FieldDescriptor(location_type=LocationType.Edge),
    # double zonal_normal_primal_edge(edge=15155);
    #   :cdi = "ignore";
    zonal_normal_primal_edge=FieldDescriptor(location_type=LocationType.Edge),
    # double meridional_normal_primal_edge(edge=15155);
    #   :cdi = "ignore";
    meridional_normal_primal_edge=FieldDescriptor(location_type=LocationType.Edge),
    # int neighbor_cell_index(nv=3, cell=9868);
    #   :cdi = "ignore";
    neighbor_cell_index=FieldDescriptor(
        primary_axis=1,
        location_type=LocationType.Cell,
        indexes_into=LocationType.Cell,
    ),
    # int global_cell_index(cell=9868);
    #   :cdi = "ignore";
    #   :nglobal = 20340; // int
    global_cell_index=FieldDescriptor(
        location_type=LocationType.Cell,
        indexes_into=LocationType.Cell,  # NOTE: this maps into cells of grid.nc, not lateral_boundary.grid.nc
    ),
    # int global_edge_index(edge=15155);
    #   :cdi = "ignore";
    #   :nglobal = 30715; // int
    global_edge_index=FieldDescriptor(
        location_type=LocationType.Edge,
        indexes_into=LocationType.Edge,  # NOTE: this maps into cells of grid.nc, not lateral_boundary.grid.nc
    ),
)

ICON_grid_schema_ic = make_schema(
    # double clon(ncells=20340);
    #   :standard_name = "longitude";
    #   :long_name = "center longitude";
    #   :units = "radian";
    #   :bounds = "clon_bnds";
    clon=FieldDescriptor(location_type=LocationType.Cell),
    # double clon_bnds(ncells=20340, vertices=3);
    # clon_bnds = FieldDescriptor(location_type=LocationType.Cell, indexes_into=LocationType.Vertex, primary_axis=0),
    clon_bnds=FieldDescriptor(location_type=LocationType.Cell, primary_axis=0),
    # double clat(ncells=20340);
    #   :standard_name = "latitude";
    #   :long_name = "center latitude";
    #   :units = "radian";
    #   :bounds = "clat_bnds";
    clat=FieldDescriptor(location_type=LocationType.Cell),
    # double clat_bnds(ncells=20340, vertices=3);
    # clat_bnds = FieldDescriptor(location_type=LocationType.Cell, indexes_into=LocationType.Vertex, primary_axis=0),
    clat_bnds=FieldDescriptor(location_type=LocationType.Cell, primary_axis=0),
    # float HHL(time=1, height=91, ncells=20340);
    #   :long_name = "Geometric Height of the layer limits above sea level(NN)";
    #   :units = "m";
    #   :param = "6.3.0";
    #   :CDI_grid_type = "unstructured";
    #   :number_of_grid_in_reference = 1; // int
    #   :coordinates = "clat clon";
    #   :_FillValue = -8.9999998E15f; // float
    #   :missing_value = -8.9999998E15f; // float
    HHL=FieldDescriptor(location_type=LocationType.Cell, primary_axis=2),
    # float U(time=1, height_2=90, ncells=20340);
    #   :standard_name = "eastward_wind";
    #   :long_name = "U-Component of Wind";
    #   :units = "m s-1";
    #   :param = "2.2.0";
    #   :CDI_grid_type = "unstructured";
    #   :number_of_grid_in_reference = 1; // int
    #   :coordinates = "clat clon";
    #   :_FillValue = -8.9999998E15f; // float
    #   :missing_value = -8.9999998E15f; // float
    U=FieldDescriptor(location_type=LocationType.Cell, primary_axis=2),
    # float V(time=1, height_2=90, ncells=20340);
    #   :standard_name = "northward_wind";
    #   :long_name = "V-Component of Wind";
    #   :units = "m s-1";
    #   :param = "3.2.0";
    #   :CDI_grid_type = "unstructured";
    #   :number_of_grid_in_reference = 1; // int
    #   :coordinates = "clat clon";
    #   :_FillValue = -8.9999998E15f; // float
    #   :missing_value = -8.9999998E15f; // float
    V=FieldDescriptor(location_type=LocationType.Cell, primary_axis=2),
    # float W(time=1, height_3=91, ncells=20340);
    #   :long_name = "Vertical Velocity (Geometric) (w)";
    #   :units = "m s-1";
    #   :param = "9.2.0";
    #   :CDI_grid_type = "unstructured";
    #   :number_of_grid_in_reference = 1; // int
    #   :coordinates = "clat clon";
    #   :_FillValue = -8.9999998E15f; // float
    #   :missing_value = -8.9999998E15f; // float
    W=FieldDescriptor(location_type=LocationType.Cell, primary_axis=2),
    # float T(time=1, height_2=90, ncells=20340);
    #   :standard_name = "air_temperature";
    #   :long_name = "Temperature";
    #   :units = "K";
    #   :param = "0.0.0";
    #   :CDI_grid_type = "unstructured";
    #   :number_of_grid_in_reference = 1; // int
    #   :coordinates = "clat clon";
    #   :_FillValue = -8.9999998E15f; // float
    #   :missing_value = -8.9999998E15f; // float
    T=FieldDescriptor(location_type=LocationType.Cell, primary_axis=2),
    # float P(time=1, height_2=90, ncells=20340);
    #   :long_name = "Pressure";
    #   :units = "Pa";
    #   :param = "0.3.0";
    #   :CDI_grid_type = "unstructured";
    #   :number_of_grid_in_reference = 1; // int
    #   :coordinates = "clat clon";
    #   :_FillValue = -8.9999998E15f; // float
    #   :missing_value = -8.9999998E15f; // float
    P=FieldDescriptor(location_type=LocationType.Cell, primary_axis=2),
    # float QV(time=1, height_2=90, ncells=20340);
    #   :standard_name = "specific_humidity";
    #   :long_name = "Specific Humidity";
    #   :units = "kg kg-1";
    #   :param = "0.1.0";
    #   :CDI_grid_type = "unstructured";
    #   :number_of_grid_in_reference = 1; // int
    #   :coordinates = "clat clon";
    #   :_FillValue = -8.9999998E15f; // float
    #   :missing_value = -8.9999998E15f; // float
    QV=FieldDescriptor(location_type=LocationType.Cell, primary_axis=2),
    # float QC(time=1, height_2=90, ncells=20340);
    #   :long_name = "Cloud Mixing Ratio";
    #   :units = "kg kg-1";
    #   :param = "22.1.0";
    #   :CDI_grid_type = "unstructured";
    #   :number_of_grid_in_reference = 1; // int
    #   :coordinates = "clat clon";
    #   :_FillValue = -8.9999998E15f; // float
    #   :missing_value = -8.9999998E15f; // float
    QC=FieldDescriptor(location_type=LocationType.Cell, primary_axis=2),
    # float QI(time=1, height_2=90, ncells=20340);
    #   :long_name = "Cloud Ice Mixing Ratio";
    #   :units = "kg kg-1";
    #   :param = "82.1.0";
    #   :CDI_grid_type = "unstructured";
    #   :number_of_grid_in_reference = 1; // int
    #   :coordinates = "clat clon";
    #   :_FillValue = -8.9999998E15f; // float
    #   :missing_value = -8.9999998E15f; // float
    QI=FieldDescriptor(location_type=LocationType.Cell, primary_axis=2),
    # float QR(time=1, height_2=90, ncells=20340);
    #   :long_name = "Rain mixing ratio";
    #   :units = "kg kg-1";
    #   :param = "24.1.0";
    #   :CDI_grid_type = "unstructured";
    #   :number_of_grid_in_reference = 1; // int
    #   :coordinates = "clat clon";
    #   :_FillValue = -8.9999998E15f; // float
    #   :missing_value = -8.9999998E15f; // float
    QR=FieldDescriptor(location_type=LocationType.Cell, primary_axis=2),
    # float QS(time=1, height_2=90, ncells=20340);
    #   :long_name = "Snow mixing ratio";
    #   :units = "kg kg-1";
    #   :param = "25.1.0";
    #   :CDI_grid_type = "unstructured";
    #   :number_of_grid_in_reference = 1; // int
    #   :coordinates = "clat clon";
    #   :_FillValue = -8.9999998E15f; // float
    #   :missing_value = -8.9999998E15f; // float
    QS=FieldDescriptor(location_type=LocationType.Cell, primary_axis=2),
    # float T_G(time=1, ncells=20340);
    #   :standard_name = "air_temperature";
    #   :long_name = "Temperature (G)";
    #   :units = "K";
    #   :param = "0.0.0";
    #   :CDI_grid_type = "unstructured";
    #   :number_of_grid_in_reference = 1; // int
    #   :coordinates = "clat clon";
    #   :_FillValue = -8.9999998E15f; // float
    #   :missing_value = -8.9999998E15f; // float
    T_G=FieldDescriptor(location_type=LocationType.Cell, primary_axis=1),
    # float T_ICE(time=1, ncells=20340);
    #   :long_name = "Sea Ice Temperature";
    #   :units = "K";
    #   :param = "8.2.10";
    #   :CDI_grid_type = "unstructured";
    #   :number_of_grid_in_reference = 1; // int
    #   :coordinates = "clat clon";
    #   :_FillValue = -8.9999998E15f; // float
    #   :missing_value = -8.9999998E15f; // float
    T_ICE=FieldDescriptor(location_type=LocationType.Cell, primary_axis=1),
    # float H_ICE(time=1, ncells=20340);
    #   :long_name = "Sea Ice Thickness";
    #   :units = "m";
    #   :param = "1.2.10";
    #   :CDI_grid_type = "unstructured";
    #   :number_of_grid_in_reference = 1; // int
    #   :coordinates = "clat clon";
    #   :_FillValue = -8.9999998E15f; // float
    #   :missing_value = -8.9999998E15f; // float
    H_ICE=FieldDescriptor(location_type=LocationType.Cell, primary_axis=1),
    # float T_MNW_LK(time=1, ncells=20340);
    #   :long_name = "Mean temperature of the water column";
    #   :units = "K";
    #   :param = "1.2.1";
    #   :CDI_grid_type = "unstructured";
    #   :number_of_grid_in_reference = 1; // int
    #   :coordinates = "clat clon";
    #   :_FillValue = -8.9999998E15f; // float
    #   :missing_value = -8.9999998E15f; // float
    T_MNW_LK=FieldDescriptor(location_type=LocationType.Cell, primary_axis=1),
    # float T_WML_LK(time=1, ncells=20340);
    #   :long_name = "Mixed-layer temperature";
    #   :units = "K";
    #   :param = "1.2.1";
    #   :CDI_grid_type = "unstructured";
    #   :number_of_grid_in_reference = 1; // int
    #   :coordinates = "clat clon";
    #   :_FillValue = -8.9999998E15f; // float
    #   :missing_value = -8.9999998E15f; // float
    T_WML_LK=FieldDescriptor(location_type=LocationType.Cell, primary_axis=1),
    # float H_ML_LK(time=1, ncells=20340);
    #   :long_name = "Mixed-layer depth";
    #   :units = "m";
    #   :param = "0.2.1";
    #   :CDI_grid_type = "unstructured";
    #   :number_of_grid_in_reference = 1; // int
    #   :coordinates = "clat clon";
    #   :_FillValue = -8.9999998E15f; // float
    #   :missing_value = -8.9999998E15f; // float
    H_ML_LK=FieldDescriptor(location_type=LocationType.Cell, primary_axis=1),
    # float T_BOT_LK(time=1, ncells=20340);
    #   :long_name = "Bottom temperature (temperature at the water-bottom sediment interface)";
    #   :units = "K";
    #   :param = "1.2.1";
    #   :CDI_grid_type = "unstructured";
    #   :number_of_grid_in_reference = 1; // int
    #   :coordinates = "clat clon";
    #   :_FillValue = -8.9999998E15f; // float
    #   :missing_value = -8.9999998E15f; // float
    #   :level_type = "lakebottom";
    T_BOT_LK=FieldDescriptor(location_type=LocationType.Cell, primary_axis=1),
    # float C_T_LK(time=1, ncells=20340);
    #   :long_name = "Shape factor with respect to the temperature profile in the thermocline";
    #   :units = "Numeric";
    #   :param = "10.2.1";
    #   :CDI_grid_type = "unstructured";
    #   :number_of_grid_in_reference = 1; // int
    #   :coordinates = "clat clon";
    #   :_FillValue = -8.9999998E15f; // float
    #   :missing_value = -8.9999998E15f; // float
    #   :level_type = "mixlayer";
    C_T_LK=FieldDescriptor(location_type=LocationType.Cell, primary_axis=1),
    # float QV_S(time=1, ncells=20340);
    #   :standard_name = "specific_humidity";
    #   :long_name = "Specific Humidity (S)";
    #   :units = "kg kg-1";
    #   :param = "0.1.0";
    #   :CDI_grid_type = "unstructured";
    #   :number_of_grid_in_reference = 1; // int
    #   :coordinates = "clat clon";
    #   :_FillValue = -8.9999998E15f; // float
    #   :missing_value = -8.9999998E15f; // float
    QV_S=FieldDescriptor(location_type=LocationType.Cell, primary_axis=1),
    # float T_SO(time=1, depth=9, ncells=20340);
    #   :long_name = "Soil Temperature (multilayer model)";
    #   :units = "K";
    #   :param = "18.3.2";
    #   :CDI_grid_type = "unstructured";
    #   :number_of_grid_in_reference = 1; // int
    #   :coordinates = "clat clon";
    #   :_FillValue = -8.9999998E15f; // float
    #   :missing_value = -8.9999998E15f; // float
    T_SO=FieldDescriptor(location_type=LocationType.Cell, primary_axis=2),
    # float W_SO(time=1, depth_2=8, ncells=20340);
    #   :long_name = "Column-integrated Soil Moisture (multilayers)";
    #   :units = "kg m-2";
    #   :param = "20.3.2";
    #   :CDI_grid_type = "unstructured";
    #   :number_of_grid_in_reference = 1; // int
    #   :coordinates = "clat clon";
    #   :_FillValue = -8.9999998E15f; // float
    #   :missing_value = -8.9999998E15f; // float
    W_SO=FieldDescriptor(location_type=LocationType.Cell, primary_axis=2),
    # float W_SO_ICE(time=1, depth_2=8, ncells=20340);
    #   :long_name = "soil ice content (multilayers)";
    #   :units = "kg m-2";
    #   :param = "22.3.2";
    #   :CDI_grid_type = "unstructured";
    #   :number_of_grid_in_reference = 1; // int
    #   :coordinates = "clat clon";
    #   :_FillValue = -8.9999998E15f; // float
    #   :missing_value = -8.9999998E15f; // float
    W_SO_ICE=FieldDescriptor(location_type=LocationType.Cell, primary_axis=2),
    # float ALB_SEAICE(time=1, ncells=20340);
    #   :long_name = "sea ice albedo - diffusive solar (0.3 - 5.0 m-6)";
    #   :units = "%";
    #   :param = "234.19.0";
    #   :CDI_grid_type = "unstructured";
    #   :number_of_grid_in_reference = 1; // int
    #   :coordinates = "clat clon";
    #   :_FillValue = -8.9999998E15f; // float
    #   :missing_value = -8.9999998E15f; // float
    ALB_SEAICE=FieldDescriptor(location_type=LocationType.Cell, primary_axis=1),
    # float EVAP_PL(time=1, ncells=20340);
    #   :long_name = "evaporation of plants (integrated since \"nightly reset\")";
    #   :units = "kgm-2";
    #   :param = "198.0.2";
    #   :CDI_grid_type = "unstructured";
    #   :number_of_grid_in_reference = 1; // int
    #   :coordinates = "clat clon";
    #   :_FillValue = -8.9999998E15f; // float
    #   :missing_value = -8.9999998E15f; // float
    EVAP_PL=FieldDescriptor(location_type=LocationType.Cell, primary_axis=1),
    # float SMI(time=1, depth_2=8, ncells=20340);
    #   :long_name = "soil moisture index (multilayers)";
    #   :units = "Numeric";
    #   :param = "200.3.2";
    #   :CDI_grid_type = "unstructured";
    #   :number_of_grid_in_reference = 1; // int
    #   :coordinates = "clat clon";
    #   :_FillValue = -8.9999998E15f; // float
    #   :missing_value = -8.9999998E15f; // float
    SMI=FieldDescriptor(location_type=LocationType.Cell, primary_axis=2),
    # float RHO_SNOW(time=1, ncells=20340);
    #   :long_name = "Snow density";
    #   :units = "kg m-3";
    #   :param = "61.1.0";
    #   :CDI_grid_type = "unstructured";
    #   :number_of_grid_in_reference = 1; // int
    #   :coordinates = "clat clon";
    #   :_FillValue = -8.9999998E15f; // float
    #   :missing_value = -8.9999998E15f; // float
    RHO_SNOW=FieldDescriptor(location_type=LocationType.Cell, primary_axis=1),
    # float T_SNOW(time=1, ncells=20340);
    #   :long_name = "Snow temperature (top of snow)";
    #   :units = "K";
    #   :param = "18.0.0";
    #   :CDI_grid_type = "unstructured";
    #   :number_of_grid_in_reference = 1; // int
    #   :coordinates = "clat clon";
    #   :_FillValue = -8.9999998E15f; // float
    #   :missing_value = -8.9999998E15f; // float
    T_SNOW=FieldDescriptor(location_type=LocationType.Cell, primary_axis=1),
    # float W_SNOW(time=1, ncells=20340);
    #   :long_name = "Snow depth water equivalent";
    #   :units = "kg m-2";
    #   :param = "60.1.0";
    #   :CDI_grid_type = "unstructured";
    #   :number_of_grid_in_reference = 1; // int
    #   :coordinates = "clat clon";
    #   :_FillValue = -8.9999998E15f; // float
    #   :missing_value = -8.9999998E15f; // float
    W_SNOW=FieldDescriptor(location_type=LocationType.Cell, primary_axis=1),
    # float H_SNOW(time=1, ncells=20340);
    #   :standard_name = "lwe_thickness_of_surface_snow_amount";
    #   :long_name = "Snow Depth";
    #   :units = "m";
    #   :param = "11.1.0";
    #   :CDI_grid_type = "unstructured";
    #   :number_of_grid_in_reference = 1; // int
    #   :coordinates = "clat clon";
    #   :_FillValue = -8.9999998E15f; // float
    #   :missing_value = -8.9999998E15f; // float
    H_SNOW=FieldDescriptor(location_type=LocationType.Cell, primary_axis=1),
    # float W_I(time=1, ncells=20340);
    #   :long_name = "Plant Canopy Surface Water";
    #   :units = "kg m-2";
    #   :param = "13.0.2";
    #   :CDI_grid_type = "unstructured";
    #   :number_of_grid_in_reference = 1; // int
    #   :coordinates = "clat clon";
    #   :_FillValue = -8.9999998E15f; // float
    #   :missing_value = -8.9999998E15f; // float
    W_I=FieldDescriptor(location_type=LocationType.Cell, primary_axis=1),
    # float FRESHSNW(time=1, ncells=20340);
    #   :long_name = "Fresh snow factor (weighting function for albedo indicating freshness of snow)";
    #   :units = "Numeric";
    #   :param = "203.1.0";
    #   :CDI_grid_type = "unstructured";
    #   :number_of_grid_in_reference = 1; // int
    #   :coordinates = "clat clon";
    #   :_FillValue = -8.9999998E15f; // float
    #   :missing_value = -8.9999998E15f; // float
    FRESHSNW=FieldDescriptor(location_type=LocationType.Cell, primary_axis=1),
    # float Z0(time=1, ncells=20340);
    #   :standard_name = "surface_roughness_length";
    #   :long_name = "Surface Roughness length Surface Roughness";
    #   :units = "m";
    #   :param = "1.0.2";
    #   :CDI_grid_type = "unstructured";
    #   :number_of_grid_in_reference = 1; // int
    #   :coordinates = "clat clon";
    #   :_FillValue = -8.9999998E15f; // float
    #   :missing_value = -8.9999998E15f; // float
    Z0=FieldDescriptor(location_type=LocationType.Cell, primary_axis=1),
    # float FR_ICE(time=1, ncells=20340);
    #   :standard_name = "sea_ice_area_fraction";
    #   :long_name = "Sea Ice Cover ( 0= free, 1=cover)";
    #   :units = "Proportion";
    #   :param = "0.2.10";
    #   :CDI_grid_type = "unstructured";
    #   :number_of_grid_in_reference = 1; // int
    #   :coordinates = "clat clon";
    #   :_FillValue = -8.9999998E15f; // float
    #   :missing_value = -8.9999998E15f; // float
    FR_ICE=FieldDescriptor(location_type=LocationType.Cell, primary_axis=1),
)

# NOTE: this is probably not needed since we are not reordering lateral_boundary.grid.nc (but the indexes therein)
ICON_grid_schema_lbc = make_schema(
    # double clon(ncells=9868);
    #   :standard_name = "longitude";
    #   :long_name = "center longitude";
    #   :units = "radian";
    #   :bounds = "clon_bnds";
    clon=FieldDescriptor(location_type=LocationType.Cell),
    # double clon_bnds(ncells=9868, vertices=3);
    # clon_bnds = FieldDescriptor(location_type=LocationType.Cell, indexes_into=LocationType.Vertex, primary_axis=0),
    clon_bnds=FieldDescriptor(location_type=LocationType.Cell, primary_axis=0),
    # double clat(ncells=9868);
    #   :standard_name = "latitude";
    #   :long_name = "center latitude";
    #   :units = "radian";
    #   :bounds = "clat_bnds";
    clat=FieldDescriptor(location_type=LocationType.Cell),
    # double clat_bnds(ncells=9868, vertices=3);
    # clat_bnds = FieldDescriptor(location_type=LocationType.Cell, indexes_into=LocationType.Vertex, primary_axis=0),
    clat_bnds=FieldDescriptor(location_type=LocationType.Cell, primary_axis=0),
    # float HHL(time=1, height=91, ncells=9868);
    #   :long_name = "Geometric Height of the layer limits above sea level(NN)";
    #   :units = "m";
    #   :param = "6.3.0";
    #   :CDI_grid_type = "unstructured";
    #   :number_of_grid_in_reference = 1; // int
    #   :coordinates = "clat clon";
    #   :_FillValue = -8.9999998E15f; // float
    #   :missing_value = -8.9999998E15f; // float
    HHL=FieldDescriptor(location_type=LocationType.Cell, primary_axis=2),
    # float U(time=1, height_2=90, ncells=9868);
    #   :standard_name = "eastward_wind";
    #   :long_name = "U-Component of Wind";
    #   :units = "m s-1";
    #   :param = "2.2.0";
    #   :CDI_grid_type = "unstructured";
    #   :number_of_grid_in_reference = 1; // int
    #   :coordinates = "clat clon";
    #   :_FillValue = -8.9999998E15f; // float
    #   :missing_value = -8.9999998E15f; // float
    U=FieldDescriptor(location_type=LocationType.Cell, primary_axis=2),
    # float V(time=1, height_2=90, ncells=9868);
    #   :standard_name = "northward_wind";
    #   :long_name = "V-Component of Wind";
    #   :units = "m s-1";
    #   :param = "3.2.0";
    #   :CDI_grid_type = "unstructured";
    #   :number_of_grid_in_reference = 1; // int
    #   :coordinates = "clat clon";
    #   :_FillValue = -8.9999998E15f; // float
    #   :missing_value = -8.9999998E15f; // float
    V=FieldDescriptor(location_type=LocationType.Cell, primary_axis=2),
    # float W(time=1, height_3=91, ncells=9868);
    #   :long_name = "Vertical Velocity (Geometric) (w)";
    #   :units = "m s-1";
    #   :param = "9.2.0";
    #   :CDI_grid_type = "unstructured";
    #   :number_of_grid_in_reference = 1; // int
    #   :coordinates = "clat clon";
    #   :_FillValue = -8.9999998E15f; // float
    #   :missing_value = -8.9999998E15f; // float
    W=FieldDescriptor(location_type=LocationType.Cell, primary_axis=2),
    # float T(time=1, height_2=90, ncells=9868);
    #   :standard_name = "air_temperature";
    #   :long_name = "Temperature";
    #   :units = "K";
    #   :param = "0.0.0";
    #   :CDI_grid_type = "unstructured";
    #   :number_of_grid_in_reference = 1; // int
    #   :coordinates = "clat clon";
    #   :_FillValue = -8.9999998E15f; // float
    #   :missing_value = -8.9999998E15f; // float
    T=FieldDescriptor(location_type=LocationType.Cell, primary_axis=2),
    # float P(time=1, height_2=90, ncells=9868);
    #   :long_name = "Pressure";
    #   :units = "Pa";
    #   :param = "0.3.0";
    #   :CDI_grid_type = "unstructured";
    #   :number_of_grid_in_reference = 1; // int
    #   :coordinates = "clat clon";
    #   :_FillValue = -8.9999998E15f; // float
    #   :missing_value = -8.9999998E15f; // float
    P=FieldDescriptor(location_type=LocationType.Cell, primary_axis=2),
    # float QV(time=1, height_2=90, ncells=9868);
    #   :standard_name = "specific_humidity";
    #   :long_name = "Specific Humidity";
    #   :units = "kg kg-1";
    #   :param = "0.1.0";
    #   :CDI_grid_type = "unstructured";
    #   :number_of_grid_in_reference = 1; // int
    #   :coordinates = "clat clon";
    #   :_FillValue = -8.9999998E15f; // float
    #   :missing_value = -8.9999998E15f; // float
    QV=FieldDescriptor(location_type=LocationType.Cell, primary_axis=2),
    # float QC(time=1, height_2=90, ncells=9868);
    #   :long_name = "Cloud Mixing Ratio";
    #   :units = "kg kg-1";
    #   :param = "22.1.0";
    #   :CDI_grid_type = "unstructured";
    #   :number_of_grid_in_reference = 1; // int
    #   :coordinates = "clat clon";
    #   :_FillValue = -8.9999998E15f; // float
    #   :missing_value = -8.9999998E15f; // float
    QC=FieldDescriptor(location_type=LocationType.Cell, primary_axis=2),
    # float QI(time=1, height_2=90, ncells=9868);
    #   :long_name = "Cloud Ice Mixing Ratio";
    #   :units = "kg kg-1";
    #   :param = "82.1.0";
    #   :CDI_grid_type = "unstructured";
    #   :number_of_grid_in_reference = 1; // int
    #   :coordinates = "clat clon";
    #   :_FillValue = -8.9999998E15f; // float
    #   :missing_value = -8.9999998E15f; // float
    QI=FieldDescriptor(location_type=LocationType.Cell, primary_axis=2),
    # float QR(time=1, height_2=90, ncells=9868);
    #   :long_name = "Rain mixing ratio";
    #   :units = "kg kg-1";
    #   :param = "24.1.0";
    #   :CDI_grid_type = "unstructured";
    #   :number_of_grid_in_reference = 1; // int
    #   :coordinates = "clat clon";
    #   :_FillValue = -8.9999998E15f; // float
    #   :missing_value = -8.9999998E15f; // float
    QR=FieldDescriptor(location_type=LocationType.Cell, primary_axis=2),
    # float QS(time=1, height_2=90, ncells=9868);
    #   :long_name = "Snow mixing ratio";
    #   :units = "kg kg-1";
    #   :param = "25.1.0";
    #   :CDI_grid_type = "unstructured";
    #   :number_of_grid_in_reference = 1; // int
    #   :coordinates = "clat clon";
    #   :_FillValue = -8.9999998E15f; // float
    #   :missing_value = -8.9999998E15f; // float
    QS=FieldDescriptor(location_type=LocationType.Cell, primary_axis=2),
)

ICON_grid_schema_extpar = make_schema(
    #     int SOILTYP(cell=20340);
    #   :standard_name = "soil_type";
    #   :long_name = "soil type derived from FAO Digital Soil Map of the World";
    #   :_FillValue = -999; // int
    #   :missing_value = -999; // int
    #   :data_set = "FAO Digital Soil Map of the World";
    #   :CDI_grid_type = "unstructured";
    SOILTYP=FieldDescriptor(location_type=LocationType.Cell),
    # float FR_LAND(cell=20340);
    #   :standard_name = "land_area_fraction";
    #   :long_name = "Fraction land";
    #   :_FillValue = -1.0E20f; // float
    #   :missing_value = -1.0E20f; // float
    #   :CDI_grid_type = "unstructured";
    FR_LAND=FieldDescriptor(location_type=LocationType.Cell),
    # float ICE(cell=20340);
    #   :standard_name = "-";
    #   :long_name = "Ice fraction due to GLOBCOVER2009 Data";
    #   :_FillValue = -1.0E20f; // float
    #   :missing_value = -1.0E20f; // float
    #   :CDI_grid_type = "unstructured";
    ICE=FieldDescriptor(location_type=LocationType.Cell),
    # float PLCOV_MX(cell=20340);
    #   :standard_name = "vegetation_area_fraction";
    #   :long_name = "Plant cover maximum due to GLOBCOVER2009 land use data";
    #   :_FillValue = -1.0E20f; // float
    #   :missing_value = -1.0E20f; // float
    #   :CDI_grid_type = "unstructured";
    PLCOV_MX=FieldDescriptor(location_type=LocationType.Cell),
    # float LAI_MX(cell=20340);
    #   :standard_name = "leaf_area_index";
    #   :long_name = "Leaf Area Index Maximum";
    #   :_FillValue = -1.0E20f; // float
    #   :missing_value = -1.0E20f; // float
    #   :CDI_grid_type = "unstructured";
    LAI_MX=FieldDescriptor(location_type=LocationType.Cell),
    # float RSMIN(cell=20340);
    #   :standard_name = "-";
    #   :long_name = "Minimal stomata resistence";
    #   :units = "s/m";
    #   :_FillValue = -1.0E20f; // float
    #   :missing_value = -1.0E20f; // float
    #   :CDI_grid_type = "unstructured";
    RSMIN=FieldDescriptor(location_type=LocationType.Cell),
    # float URBAN(cell=20340);
    #   :standard_name = "-";
    #   :long_name = "urban area fraction";
    #   :_FillValue = -1.0E20f; // float
    #   :missing_value = -1.0E20f; // float
    #   :CDI_grid_type = "unstructured";
    URBAN=FieldDescriptor(location_type=LocationType.Cell),
    # float FOR_D(cell=20340);
    #   :standard_name = "-";
    #   :long_name = "Fraction of deciduous forest";
    #   :_FillValue = -1.0E20f; // float
    #   :missing_value = -1.0E20f; // float
    #   :CDI_grid_type = "unstructured";
    FOR_D=FieldDescriptor(location_type=LocationType.Cell),
    # float FOR_E(cell=20340);
    #   :standard_name = "-";
    #   :long_name = "Fraction of evergreen forest";
    #   :_FillValue = -1.0E20f; // float
    #   :missing_value = -1.0E20f; // float
    #   :CDI_grid_type = "unstructured";
    FOR_E=FieldDescriptor(location_type=LocationType.Cell),
    # float SKC(cell=20340);
    #   :standard_name = "skin_conductivity";
    #   :long_name = "Skin conductivity";
    #   :_FillValue = -1.0E20f; // float
    #   :missing_value = -1.0E20f; // float
    #   :CDI_grid_type = "unstructured";
    SKC=FieldDescriptor(location_type=LocationType.Cell),
    # float EMIS_RAD(cell=20340);
    #   :standard_name = "-";
    #   :long_name = "longwave surface emissivity";
    #   :_FillValue = -1.0E20f; // float
    #   :missing_value = -1.0E20f; // float
    #   :CDI_grid_type = "unstructured";
    EMIS_RAD=FieldDescriptor(location_type=LocationType.Cell),
    # float ROOTDP(cell=20340);
    #   :standard_name = "root_depth";
    #   :long_name = "Root depth";
    #   :units = "m";
    #   :_FillValue = -1.0E20f; // float
    #   :missing_value = -1.0E20f; // float
    #   :CDI_grid_type = "unstructured";
    ROOTDP=FieldDescriptor(location_type=LocationType.Cell),
    # float Z0(cell=20340);
    #   :standard_name = "surface_roughness_length";
    #   :long_name = "Roughness length";
    #   :_FillValue = -1.0E20f; // float
    #   :missing_value = -1.0E20f; // float
    #   :CDI_grid_type = "unstructured";
    Z0=FieldDescriptor(location_type=LocationType.Cell),
    # float lon(cell=20340);
    #   :standard_name = "longitude";
    #   :long_name = "geographical longitude";
    #   :units = "degrees_east";
    #   :_FillValue = -1.0E20f; // float
    #   :missing_value = -1.0E20f; // float
    #   :CDI_grid_type = "unstructured";
    lon=FieldDescriptor(location_type=LocationType.Cell),
    # float lat(cell=20340);
    #   :standard_name = "latitude";
    #   :long_name = "geographical latitude";
    #   :units = "degrees_north";
    #   :_FillValue = -1.0E20f; // float
    #   :missing_value = -1.0E20f; // float
    #   :CDI_grid_type = "unstructured";
    lat=FieldDescriptor(location_type=LocationType.Cell),
    # float NDVI_MAX(cell=20340);
    #   :standard_name = "-";
    #   :long_name = "NDVI yearly maximum for climatology 1998-2003";
    #   :_FillValue = -1.0E20f; // float
    #   :missing_value = -1.0E20f; // float
    #   :data_set = "NASA/GSFS climatology 1998-2003";
    #   :CDI_grid_type = "unstructured";
    NDVI_MAX=FieldDescriptor(location_type=LocationType.Cell),
    # float topography_c(cell=20340);
    #   :standard_name = "surface_altitude";
    #   :long_name = "geometric height of the earths surface above sea level";
    #   :units = "m";
    #   :_FillValue = -1.0E20f; // float
    #   :missing_value = -1.0E20f; // float
    #   :data_set = "ASTER";
    #   :CDI_grid_type = "unstructured";
    topography_c=FieldDescriptor(location_type=LocationType.Cell),
    # float SSO_OROMAX(cell=20340);
    #   :standard_name = "maximum_contributing_surface_height";
    #   :long_name = "maximum geometric height of contributing raw data height points";
    #   :units = "m";
    #   :_FillValue = -1.0E20f; // float
    #   :missing_value = -1.0E20f; // float
    #   :data_set = "";
    #   :CDI_grid_type = "unstructured";
    SSO_OROMAX=FieldDescriptor(location_type=LocationType.Cell),
    # float SSO_OROMIN(cell=20340);
    #   :standard_name = "minimum_contributing_surface_height";
    #   :long_name = "minimum geometric height of contributing raw data height points";
    #   :units = "m";
    #   :_FillValue = -1.0E20f; // float
    #   :missing_value = -1.0E20f; // float
    #   :data_set = "";
    #   :CDI_grid_type = "unstructured";
    SSO_OROMIN=FieldDescriptor(location_type=LocationType.Cell),
    # float SSO_STDH(cell=20340);
    #   :standard_name = "-";
    #   :long_name = "standard deviation of subgrid scale orography";
    #   :units = "m";
    #   :_FillValue = -1.0E20f; // float
    #   :missing_value = -1.0E20f; // float
    #   :data_set = "ASTER";
    #   :CDI_grid_type = "unstructured";
    SSO_STDH=FieldDescriptor(location_type=LocationType.Cell),
    # float SSO_THETA(cell=20340);
    #   :standard_name = "-";
    #   :long_name = "Angle of sub-gridscale orography";
    #   :units = "rad";
    #   :_FillValue = -1.0E20f; // float
    #   :missing_value = -1.0E20f; // float
    #   :data_set = "ASTER";
    #   :CDI_grid_type = "unstructured";
    SSO_THETA=FieldDescriptor(location_type=LocationType.Cell),
    # float SSO_GAMMA(cell=20340);
    #   :standard_name = "-";
    #   :long_name = "anisotropy of sub-gridscale orography";
    #   :_FillValue = -1.0E20f; // float
    #   :missing_value = -1.0E20f; // float
    #   :data_set = "ASTER";
    #   :CDI_grid_type = "unstructured";
    SSO_GAMMA=FieldDescriptor(location_type=LocationType.Cell),
    # float SSO_SIGMA(cell=20340);
    #   :standard_name = "-";
    #   :long_name = "Slope of sub-gridscale orography";
    #   :_FillValue = -1.0E20f; // float
    #   :missing_value = -1.0E20f; // float
    #   :data_set = "ASTER";
    #   :CDI_grid_type = "unstructured";
    SSO_SIGMA=FieldDescriptor(location_type=LocationType.Cell),
    # float T_CL(cell=20340);
    #   :standard_name = "soil_temperature";
    #   :long_name = "CRU near surface temperature climatology";
    #   :units = "K";
    #   :_FillValue = -1.0E20f; // float
    #   :missing_value = -1.0E20f; // float
    #   :data_set = "CRU";
    #   :CDI_grid_type = "unstructured";
    T_CL=FieldDescriptor(location_type=LocationType.Cell),
    # float FR_LAKE(cell=20340);
    #   :standard_name = "-";
    #   :long_name = "fraction lake";
    #   :units = "1";
    #   :_FillValue = -1.0E20f; // float
    #   :missing_value = -1.0E20f; // float
    #   :data_set = "DWD/RSHU/MeteoFrance";
    #   :CDI_grid_type = "unstructured";
    FR_LAKE=FieldDescriptor(location_type=LocationType.Cell),
    # float DEPTH_LK(cell=20340);
    #   :standard_name = "sea_floor_depth_below_sea_level";
    #   :long_name = "Lake depth";
    #   :units = "m";
    #   :_FillValue = -1.0E20f; // float
    #   :missing_value = -1.0E20f; // float
    #   :data_set = "DWD/RSHU/MeteoFrance";
    #   :CDI_grid_type = "unstructured";
    DEPTH_LK=FieldDescriptor(location_type=LocationType.Cell),
    # float LU_CLASS_FRACTION(nclass_lu=23, cell=20340);
    #   :standard_name = "-";
    #   :long_name = "Fraction of GLOBCOVER2009 land use classes in target grid element";
    #   :_FillValue = -1.0E20f; // float
    #   :missing_value = -1.0E20f; // float
    #   :CDI_grid_type = "unstructured";
    LU_CLASS_FRACTION=FieldDescriptor(location_type=LocationType.Cell, primary_axis=1),
    # float NDVI(time=12, cell=20340);
    #   :standard_name = "-";
    #   :long_name = "monthly mean NDVI climatology 1998-2003";
    #   :_FillValue = -1.0E20f; // float
    #   :missing_value = -1.0E20f; // float
    #   :data_set = " NASA/GSFS climatology 1998-2003";
    #   :CDI_grid_type = "unstructured";
    NDVI=FieldDescriptor(location_type=LocationType.Cell, primary_axis=1),
    # float NDVI_MRAT(time=12, cell=20340);
    #   :standard_name = "-";
    #   :long_name = "(monthly) proportion of actual value/maximum normalized differential vegetation index";
    #   :_FillValue = -1.0E20f; // float
    #   :missing_value = -1.0E20f; // float
    #   :data_set = " NASA/GSFS climatology 1998-2003";
    #   :CDI_grid_type = "unstructured";
    NDVI_MRAT=FieldDescriptor(location_type=LocationType.Cell, primary_axis=1),
    # float AER_BC(time=12, cell=20340);
    #   :standard_name = "atmosphere_absorption_optical_thickness_due_to_black_carbon_ambient_aerosol";
    #   :long_name = "aerosol optical thickness of black carbon";
    #   :_FillValue = -1.0E20f; // float
    #   :missing_value = -1.0E20f; // float
    #   :data_set = "Tegen JGR 1997 (NASA/GISS)";
    #   :CDI_grid_type = "unstructured";
    AER_BC=FieldDescriptor(location_type=LocationType.Cell, primary_axis=1),
    # float AER_DUST(time=12, cell=20340);
    #   :standard_name = "-";
    #   :long_name = "atmosphere_absorption_optical_thickness_due_to_dust_ambient_aerosol";
    #   :_FillValue = -1.0E20f; // float
    #   :missing_value = -1.0E20f; // float
    #   :data_set = "Tegen JGR 1997 (NASA/GISS)";
    #   :CDI_grid_type = "unstructured";
    AER_DUST=FieldDescriptor(location_type=LocationType.Cell, primary_axis=1),
    # float AER_ORG(time=12, cell=20340);
    #   :standard_name = "-";
    #   :long_name = "atmosphere_absorption_optical_thickness_due_to_particulate_organic_matter_ambient_aerosol";
    #   :_FillValue = -1.0E20f; // float
    #   :missing_value = -1.0E20f; // float
    #   :data_set = "Tegen JGR 1997 (NASA/GISS)";
    #   :CDI_grid_type = "unstructured";
    AER_ORG=FieldDescriptor(location_type=LocationType.Cell, primary_axis=1),
    # float AER_SO4(time=12, cell=20340);
    #   :standard_name = "-";
    #   :long_name = "atmosphere_absorption_optical_thickness_due_to_sulfate_ambient_aerosol";
    #   :_FillValue = -1.0E20f; // float
    #   :missing_value = -1.0E20f; // float
    #   :data_set = "Tegen JGR 1997 (NASA/GISS)";
    #   :CDI_grid_type = "unstructured";
    AER_SO4=FieldDescriptor(location_type=LocationType.Cell, primary_axis=1),
    # float AER_SS(time=12, cell=20340);
    #   :standard_name = "-";
    #   :long_name = "atmosphere_absorption_optical_thickness_due_to_seasalt_ambient_aerosol";
    #   :_FillValue = -1.0E20f; // float
    #   :missing_value = -1.0E20f; // float
    #   :data_set = "Tegen JGR 1997 (NASA/GISS)";
    #   :CDI_grid_type = "unstructured";
    AER_SS=FieldDescriptor(location_type=LocationType.Cell, primary_axis=1),
    # float ALB(time=12, cell=20340);
    #   :standard_name = "surface_albedo";
    #   :long_name = "Albedo";
    #   :units = "%";
    #   :_FillValue = -1.0E20f; // float
    #   :missing_value = -1.0E20f; // float
    #   :data_set = "MODIS";
    #   :CDI_grid_type = "unstructured";
    ALB=FieldDescriptor(location_type=LocationType.Cell, primary_axis=1),
    # float ALNID(time=12, cell=20340);
    #   :standard_name = "-";
    #   :long_name = "NI_Albedo";
    #   :units = "%";
    #   :_FillValue = -1.0E20f; // float
    #   :missing_value = -1.0E20f; // float
    #   :data_set = "MODIS";
    #   :CDI_grid_type = "unstructured";
    ALNID=FieldDescriptor(location_type=LocationType.Cell, primary_axis=1),
    # float ALUVD(time=12, cell=20340);
    #   :standard_name = "-";
    #   :long_name = "UV_Albedo";
    #   :units = "%";
    #   :_FillValue = -1.0E20f; // float
    #   :missing_value = -1.0E20f; // float
    #   :data_set = "MODIS";
    #   :CDI_grid_type = "unstructured";
    ALUVD=FieldDescriptor(location_type=LocationType.Cell, primary_axis=1),
    # float T_SEA(time=12, cell=20340);
    #   :standard_name = "T_SEA";
    #   :long_name = "monthly mean SST climatology 1986-2015";
    #   :_FillValue = -1.0E20f; // float
    #   :missing_value = -1.0E20f; // float
    #   :data_set = "";
    #   :CDI_grid_type = "unstructured";
    T_SEA=FieldDescriptor(location_type=LocationType.Cell, primary_axis=1),
    # float W_SNOW(time=12, cell=20340);
    #   :standard_name = "W_SNOW";
    #   :long_name = "monthly mean WSNOW climatology 1986-2015";
    #   :_FillValue = -1.0E20f; // float
    #   :missing_value = -1.0E20f; // float
    #   :data_set = "";
    #   :CDI_grid_type = "unstructured";
    W_SNOW=FieldDescriptor(location_type=LocationType.Cell, primary_axis=1),
    # float T_2M_CLIM(time=12, cell=20340);
    #   :standard_name = "T_2M_CLIM";
    #   :long_name = "monthly mean T2M climatology 1986-2015";
    #   :_FillValue = -1.0E20f; // float
    #   :missing_value = -1.0E20f; // float
    #   :data_set = "";
    #   :CDI_grid_type = "unstructured";
    T_2M_CLIM=FieldDescriptor(location_type=LocationType.Cell, primary_axis=1),
    # float TOPO_CLIM(cell=20340);
    #   :standard_name = "TOPO_CLIM";
    #   :long_name = "TOPO_CLIM for climatology 1986-2015";
    #   :_FillValue = -1.0E20f; // float
    #   :missing_value = -1.0E20f; // float
    #   :data_set = "";
    #   :CDI_grid_type = "unstructured";
    TOPO_CLIM=FieldDescriptor(location_type=LocationType.Cell),
    # float clon(cell=20340);
    #   :standard_name = "grid_longitude";
    #   :long_name = "longitude of icon grid cell centre";
    #   :units = "radians";
    #   :_FillValue = -1.0E20f; // float
    #   :missing_value = -1.0E20f; // float
    #   :CDI_grid_type = "unstructured";
    clon=FieldDescriptor(location_type=LocationType.Cell),
    # float clat(cell=20340);
    #   :standard_name = "grid_latitude";
    #   :long_name = "latitude of icon grid cell centre";
    #   :units = "radians";
    #   :_FillValue = -1.0E20f; // float
    #   :missing_value = -1.0E20f; // float
    #   :CDI_grid_type = "unstructured";
    clat=FieldDescriptor(location_type=LocationType.Cell),
)

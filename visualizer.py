#!/usr/bin/env python
from __future__ import annotations
import typing

from functools import cmp_to_key
import enum
import dataclasses
import os
import shutil
import numpy as np
from matplotlib import (
    pyplot as plt,
    patches,
    collections,
)
import netCDF4


NaN = float("nan")
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
            v_grf=np.concatenate((get_var("start_idx_v"), get_var("end_idx_v")), axis=1) - 1,
            e_grf=np.concatenate((get_var("start_idx_e"), get_var("end_idx_e")), axis=1) - 1,
            c_grf=np.concatenate((get_var("start_idx_c"), get_var("end_idx_c")), axis=1) - 1,
        )
    
    @staticmethod
    def lat_from_netCDF4(ncf) -> Grid:

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
            v2e=0,
            v2c=get_var("cells_of_vertex") - 1,
            e2v=get_var("edge_vertices") - 1,
            e2c=get_var("adjacent_cell_of_edge") - 1,
            c2v=get_var("vertex_of_cell") - 1,
            c2e=0,
            v_grf=0,
            e_grf=0,
            c_grf=0,
        )


class LocationType(enum.Enum):
    Vertex = enum.auto()
    Edge = enum.auto()
    Cell = enum.auto()


@dataclasses.dataclass
class FieldDescriptor:
    location_type: typing.Optional[LocationType] = None
    indexes_into: typing.Optional[LocationType] = None
    #FIXME: `primary_axis` is quite weird...
    # some fields that have a `location_type` are transposed
    # (only needed with `location_type` and if the field is more than one dimensional)
    primary_axis: typing.Optional[int] = None


GridScheme = typing.Dict[str, FieldDescriptor]
def make_schema(**fields: FieldDescriptor) -> GridScheme:
    return fields


ICON_grid_schema = make_schema(
    #FIXME: make sure this is complete
    ###########################################################################
    # grid topology and lon/lat coordinates
    ###########################################################################
    vlon=FieldDescriptor(
        location_type=LocationType.Vertex
    ),
    vlat=FieldDescriptor(
        location_type=LocationType.Vertex
    ),
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
    elon=FieldDescriptor(
        location_type=LocationType.Edge
    ),
    elat=FieldDescriptor(
        location_type=LocationType.Edge
    ),
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
    clon=FieldDescriptor(
        location_type=LocationType.Cell
    ),
    clat=FieldDescriptor(
        location_type=LocationType.Cell
    ),
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
    cartesian_x_vertices=FieldDescriptor(
        location_type=LocationType.Vertex
    ),
    cartesian_y_vertices=FieldDescriptor(
        location_type=LocationType.Vertex
    ),
    cartesian_z_vertices=FieldDescriptor(
        location_type=LocationType.Vertex
    ),
    dual_area=FieldDescriptor(
        location_type=LocationType.Vertex
    ),
    longitude_vertices=FieldDescriptor(
        location_type=LocationType.Vertex
    ),
    latitude_vertices=FieldDescriptor(
        location_type=LocationType.Vertex
    ),
    dual_area_p=FieldDescriptor(
        location_type=LocationType.Vertex
    ),
    vlon_vertices=FieldDescriptor(
        primary_axis=0,
        location_type=LocationType.Vertex
    ),
    vlat_vertices=FieldDescriptor(
        primary_axis=0,
        location_type=LocationType.Vertex
    ),
    #FIXME: will this work just like this?
    edge_orientation=FieldDescriptor(
        primary_axis=1,
        location_type=LocationType.Vertex
    ),
    refin_v_ctrl=FieldDescriptor(
        location_type=LocationType.Vertex
    ),
    parent_vertex_index=FieldDescriptor(
        location_type=LocationType.Vertex
    ),
    ###########################################################################
    # other edge fields
    ###########################################################################
    lon_edge_centre=FieldDescriptor(
        location_type=LocationType.Edge
    ),
    lat_edge_centre=FieldDescriptor(
        location_type=LocationType.Edge
    ),
    edge_length=FieldDescriptor(
        location_type=LocationType.Edge
    ),
    edge_cell_distance=FieldDescriptor(
        primary_axis=1,
        location_type=LocationType.Edge
    ),
    dual_edge_length=FieldDescriptor(
        location_type=LocationType.Edge
    ),
    edge_vert_distance=FieldDescriptor(
        primary_axis=1,
        location_type=LocationType.Edge
    ),
    zonal_normal_primal_edge=FieldDescriptor(
        location_type=LocationType.Edge
    ),
    meridional_normal_primal_edge=FieldDescriptor(
        location_type=LocationType.Edge
    ),
    zonal_normal_dual_edge=FieldDescriptor(
        location_type=LocationType.Edge
    ),
    meridional_normal_dual_edge=FieldDescriptor(
        location_type=LocationType.Edge
    ),
    edge_system_orientation=FieldDescriptor(
        location_type=LocationType.Edge
    ),
    elon_vertices=FieldDescriptor(
        primary_axis=0,
        location_type=LocationType.Edge
    ),
    elat_vertices=FieldDescriptor(
        primary_axis=0,
        location_type=LocationType.Edge
    ),
    quadrilateral_area=FieldDescriptor(
        location_type=LocationType.Edge
    ),
    refin_e_ctrl=FieldDescriptor(
        location_type=LocationType.Edge
    ),
    parent_edge_index=FieldDescriptor(
        location_type=LocationType.Edge
    ),
    ###########################################################################
    # other cell fields
    ###########################################################################
    cell_area=FieldDescriptor(
        location_type=LocationType.Cell
    ),
    lon_cell_centre=FieldDescriptor(
        location_type=LocationType.Cell
    ),
    lat_cell_centre=FieldDescriptor(
        location_type=LocationType.Cell
    ),
    cell_area_p=FieldDescriptor(
        location_type=LocationType.Cell
    ),
    orientation_of_normal=FieldDescriptor(
        primary_axis=1,
        location_type=LocationType.Cell
    ),
    clon_vertices=FieldDescriptor(
        primary_axis=0,
        location_type=LocationType.Cell,
    ),
    clat_vertices=FieldDescriptor(
        primary_axis=0,
        location_type=LocationType.Cell,
    ),
    parent_cell_index=FieldDescriptor(
        location_type=LocationType.Cell
    ),
    refin_c_ctrl=FieldDescriptor(
        location_type=LocationType.Cell
    ),
)

ICON_grid_schema_dbg = make_schema(
    #   double clon(cell=20340);
    #   :units = "radian";
    #   :standard_name = "grid_longitude";
    #   :bounds = "clon_vertices";
    clon = FieldDescriptor(location_type=LocationType.Cell),

    # double clat(cell=20340);
    #   :units = "radian";
    #   :standard_name = "grid_latitude";
    #   :bounds = "clat_vertices";
    clat = FieldDescriptor(location_type=LocationType.Cell),

    # double vlon(vertex=10376);
    #   :units = "radian";
    #   :standard_name = "grid_longitude";
    #   :bounds = "vlon_vertices";
    vlon = FieldDescriptor(location_type=LocationType.Vertex),

    # double vlat(vertex=10376);
    #   :units = "radian";
    #   :standard_name = "grid_latitude";
    #   :bounds = "vlat_vertices";
    vlat = FieldDescriptor(location_type=LocationType.Vertex),

    # double cartesian_x_vertices(vertex=10376);
    cartesian_x_vertices = FieldDescriptor(location_type=LocationType.Vertex),

    # double cartesian_y_vertices(vertex=10376);
    cartesian_y_vertices = FieldDescriptor(location_type=LocationType.Vertex),

    # double cartesian_z_vertices(vertex=10376);
    cartesian_z_vertices = FieldDescriptor(location_type=LocationType.Vertex),

    # double elon(edge=30715);
    #   :units = "radian";
    #   :standard_name = "grid_longitude";
    #   :bounds = "elon_vertices";
    elon = FieldDescriptor(location_type=LocationType.Edge),

    # double elat(edge=30715);
    #   :units = "radian";
    #   :standard_name = "grid_latitude";
    #   :bounds = "elat_vertices";
    elat = FieldDescriptor(location_type=LocationType.Edge),

    # double cell_area(cell=20340);
    #   :coordinates = "clon clat";
    cell_area = FieldDescriptor(location_type=LocationType.Cell),

    # double dual_area(vertex=10376);
    dual_area = FieldDescriptor(location_type=LocationType.Vertex),

    # double lon_cell_centre(cell=20340);
    lon_cell_centre = FieldDescriptor(location_type=LocationType.Cell),

    # double lat_cell_centre(cell=20340);
    lat_cell_centre = FieldDescriptor(location_type=LocationType.Cell),

    # double longitude_vertices(vertex=10376);
    longitude_vertices = FieldDescriptor(location_type=LocationType.Vertex),

    # double latitude_vertices(vertex=10376);
    latitude_vertices = FieldDescriptor(location_type=LocationType.Vertex),

    # double lon_edge_centre(edge=30715);
    lon_edge_centre = FieldDescriptor(location_type=LocationType.Edge),

    # double lat_edge_centre(edge=30715);
    lat_edge_centre = FieldDescriptor(location_type=LocationType.Edge),

    # int edge_of_cell(nv=3, cell=20340);
    edge_of_cell = FieldDescriptor(location_type=LocationType.Cell, indexes_into=LocationType.Edge, primary_axis=1),

    # int vertex_of_cell(nv=3, cell=20340);
    vertex_of_cell = FieldDescriptor(location_type=LocationType.Cell, indexes_into=LocationType.Vertex, primary_axis=1),

    # int adjacent_cell_of_edge(nc=2, edge=30715);
    adjacent_cell_of_edge = FieldDescriptor(location_type=LocationType.Edge, indexes_into=LocationType.Cell, primary_axis=1),

    # int edge_vertices(nc=2, edge=30715);
    edge_vertices = FieldDescriptor(location_type=LocationType.Edge, indexes_into=LocationType.Vertex, primary_axis=1),

    # int cells_of_vertex(ne=6, vertex=10376);
    cells_of_vertex = FieldDescriptor(location_type=LocationType.Vertex, indexes_into=LocationType.Cell, primary_axis=1),

    # int edges_of_vertex(ne=6, vertex=10376);
    edges_of_vertex = FieldDescriptor(location_type=LocationType.Vertex, indexes_into=LocationType.Edge, primary_axis=1),

    # int vertices_of_vertex(ne=6, vertex=10376);
    vertices_of_vertex = FieldDescriptor(location_type=LocationType.Vertex, indexes_into=LocationType.Vertex, primary_axis=1),

    # double cell_area_p(cell=20340);
    cell_area_p = FieldDescriptor(location_type=LocationType.Cell),

    # double dual_area_p(vertex=10376);
    dual_area_p = FieldDescriptor(location_type=LocationType.Vertex),

    # double edge_length(edge=30715);
    edge_length = FieldDescriptor(location_type=LocationType.Edge),

    # double edge_cell_distance(nc=2, edge=30715);
    edge_cell_distance = FieldDescriptor(location_type=LocationType.Edge, primary_axis=1),

    # double dual_edge_length(edge=30715);
    dual_edge_length = FieldDescriptor(location_type=LocationType.Edge),

    # double edge_vert_distance(nc=2, edge=30715);
    edge_vert_distance = FieldDescriptor(location_type=LocationType.Edge, primary_axis=1),

    # double zonal_normal_primal_edge(edge=30715);
    zonal_normal_primal_edge = FieldDescriptor(location_type=LocationType.Edge),

    # double meridional_normal_primal_edge(edge=30715);
    meridional_normal_primal_edge = FieldDescriptor(location_type=LocationType.Edge),

    # double zonal_normal_dual_edge(edge=30715);
    zonal_normal_dual_edge = FieldDescriptor(location_type=LocationType.Edge),

    # double meridional_normal_dual_edge(edge=30715);
    meridional_normal_dual_edge = FieldDescriptor(location_type=LocationType.Edge),

    # int orientation_of_normal(nv=3, cell=20340);
    orientation_of_normal = FieldDescriptor(location_type=LocationType.Cell, primary_axis=1),

    # double clon_vertices(cell=20340, nv=3);
    clon_vertices = FieldDescriptor(location_type=LocationType.Cell, primary_axis=0),

    # double clat_vertices(cell=20340, nv=3);
    clat_vertices = FieldDescriptor(location_type=LocationType.Cell, primary_axis=0),

    # double elon_vertices(edge=30715, no=4);
    elon_vertices = FieldDescriptor(location_type=LocationType.Edge, primary_axis=0),

    # double elat_vertices(edge=30715, no=4);
    elat_vertices = FieldDescriptor(location_type=LocationType.Edge, primary_axis=0),

    # double vlon_vertices(vertex=10376, ne=6);
    vlon_vertices = FieldDescriptor(location_type=LocationType.Vertex, primary_axis=0),

    # double vlat_vertices(vertex=10376, ne=6);
    vlat_vertices = FieldDescriptor(location_type=LocationType.Vertex, primary_axis=0),

    # double quadrilateral_area(edge=30715);
    #   :coordinates = "elon elat";
    quadrilateral_area = FieldDescriptor(location_type=LocationType.Edge),

    # int parent_cell_index(cell=20340);
    #   :long_name = "parent cell index";
    #   :cdi = "ignore";
    parent_cell_index = FieldDescriptor(location_type=LocationType.Cell),

    # int neighbor_cell_index(nv=3, cell=20340);
    neighbor_cell_index = FieldDescriptor(location_type=LocationType.Cell, indexes_into=LocationType.Cell, primary_axis=1),

    # int edge_orientation(ne=6, vertex=10376);
    edge_orientation = FieldDescriptor(location_type=LocationType.Vertex, primary_axis=1),

    # int edge_system_orientation(edge=30715);
    edge_system_orientation = FieldDescriptor(location_type=LocationType.Edge),

    # int refin_c_ctrl(cell=20340);
    #   :long_name = "refinement control flag for cells";
    #   :cdi = "ignore";
    refin_c_ctrl = FieldDescriptor(location_type=LocationType.Cell),

    # int start_idx_c(max_chdom=1, cell_grf=14);
    #   :long_name = "list of start indices for each refinement control level for cells";
    #   :cdi = "ignore";

    # int end_idx_c(max_chdom=1, cell_grf=14);
    #   :long_name = "list of end indices for each refinement control level for cells";
    #   :cdi = "ignore";

    # int refin_e_ctrl(edge=30715);
    #   :long_name = "refinement control flag for edges";
    #   :cdi = "ignore";
    refin_e_ctrl = FieldDescriptor(location_type=LocationType.Edge),

    # int start_idx_e(max_chdom=1, edge_grf=24);
    #   :long_name = "list of start indices for each refinement control level for edges";
    #   :cdi = "ignore";

    # int end_idx_e(max_chdom=1, edge_grf=24);
    #   :long_name = "list of end indices for each refinement control level for edges";
    #   :cdi = "ignore";

    # int refin_v_ctrl(vertex=10376);
    #   :long_name = "refinement control flag for vertices";
    #   :cdi = "ignore";
    refin_v_ctrl = FieldDescriptor(location_type=LocationType.Vertex),

    # int start_idx_v(max_chdom=1, vert_grf=13);
    #   :long_name = "list of start indices for each refinement control level for vertices";
    #   :cdi = "ignore";

    # int end_idx_v(max_chdom=1, vert_grf=13);
    #   :long_name = "list of end indices for each refinement control level for vertices";
    #   :cdi = "ignore";

    # int parent_edge_index(edge=30715);
    #   :long_name = "parent edge index";
    #   :cdi = "ignore";
    parent_edge_index = FieldDescriptor(location_type=LocationType.Edge),

    # int parent_vertex_index(vertex=10376);
    parent_vertex_index = FieldDescriptor(location_type=LocationType.Vertex),
)

ICON_grid_schema_parent = make_schema(
    #   double clon(cell=20340);
    #   :units = "radian";
    #   :standard_name = "grid_longitude";
    #   :bounds = "clon_vertices";
    clon = FieldDescriptor(location_type=LocationType.Cell),

    # double clat(cell=20340);
    #   :units = "radian";
    #   :standard_name = "grid_latitude";
    #   :bounds = "clat_vertices";
    clat = FieldDescriptor(location_type=LocationType.Cell),

    # double vlon(vertex=10376);
    #   :units = "radian";
    #   :standard_name = "grid_longitude";
    #   :bounds = "vlon_vertices";
    vlon = FieldDescriptor(location_type=LocationType.Vertex),

    # double vlat(vertex=10376);
    #   :units = "radian";
    #   :standard_name = "grid_latitude";
    #   :bounds = "vlat_vertices";
    vlat = FieldDescriptor(location_type=LocationType.Vertex),

    # double cartesian_x_vertices(vertex=10376);
    cartesian_x_vertices = FieldDescriptor(location_type=LocationType.Vertex),

    # double cartesian_y_vertices(vertex=10376);
    cartesian_y_vertices = FieldDescriptor(location_type=LocationType.Vertex),

    # double cartesian_z_vertices(vertex=10376);
    cartesian_z_vertices = FieldDescriptor(location_type=LocationType.Vertex),

    # double elon(edge=30715);
    #   :units = "radian";
    #   :standard_name = "grid_longitude";
    #   :bounds = "elon_vertices";
    elon = FieldDescriptor(location_type=LocationType.Edge),

    # double elat(edge=30715);
    #   :units = "radian";
    #   :standard_name = "grid_latitude";
    #   :bounds = "elat_vertices";
    elat = FieldDescriptor(location_type=LocationType.Edge),

    # double cell_area(cell=20340);
    #   :coordinates = "clon clat";
    cell_area = FieldDescriptor(location_type=LocationType.Cell),

    # double dual_area(vertex=10376);
    dual_area = FieldDescriptor(location_type=LocationType.Vertex),

    # double lon_cell_centre(cell=20340);
    lon_cell_centre = FieldDescriptor(location_type=LocationType.Cell),

    # double lat_cell_centre(cell=20340);
    lat_cell_centre = FieldDescriptor(location_type=LocationType.Cell),

    # double longitude_vertices(vertex=10376);
    longitude_vertices = FieldDescriptor(location_type=LocationType.Vertex),

    # double latitude_vertices(vertex=10376);
    latitude_vertices = FieldDescriptor(location_type=LocationType.Vertex),

    # double lon_edge_centre(edge=30715);
    lon_edge_centre = FieldDescriptor(location_type=LocationType.Edge),

    # double lat_edge_centre(edge=30715);
    lat_edge_centre = FieldDescriptor(location_type=LocationType.Edge),

    # int edge_of_cell(nv=3, cell=20340);
    edge_of_cell = FieldDescriptor(location_type=LocationType.Cell, indexes_into=LocationType.Edge, primary_axis=1),

    # int vertex_of_cell(nv=3, cell=20340);
    vertex_of_cell = FieldDescriptor(location_type=LocationType.Cell, indexes_into=LocationType.Vertex, primary_axis=1),

    # int adjacent_cell_of_edge(nc=2, edge=30715);
    adjacent_cell_of_edge = FieldDescriptor(location_type=LocationType.Edge, indexes_into=LocationType.Cell, primary_axis=1),

    # int edge_vertices(nc=2, edge=30715);
    edge_vertices = FieldDescriptor(location_type=LocationType.Edge, indexes_into=LocationType.Vertex, primary_axis=1),

    # int cells_of_vertex(ne=6, vertex=10376);
    cells_of_vertex = FieldDescriptor(location_type=LocationType.Vertex, indexes_into=LocationType.Cell, primary_axis=1),

    # int edges_of_vertex(ne=6, vertex=10376);
    edges_of_vertex = FieldDescriptor(location_type=LocationType.Vertex, indexes_into=LocationType.Edge, primary_axis=1),

    # int vertices_of_vertex(ne=6, vertex=10376);
    vertices_of_vertex = FieldDescriptor(location_type=LocationType.Vertex, indexes_into=LocationType.Vertex, primary_axis=1),

    # double cell_area_p(cell=20340);
    cell_area_p = FieldDescriptor(location_type=LocationType.Cell),

    # double dual_area_p(vertex=10376);
    dual_area_p = FieldDescriptor(location_type=LocationType.Vertex),

    # double edge_length(edge=30715);
    edge_length = FieldDescriptor(location_type=LocationType.Edge),

    # double edge_cell_distance(nc=2, edge=30715);
    edge_cell_distance = FieldDescriptor(location_type=LocationType.Edge, primary_axis=1),

    # double dual_edge_length(edge=30715);
    dual_edge_length = FieldDescriptor(location_type=LocationType.Edge),

    # double edge_vert_distance(nc=2, edge=30715);
    edge_vert_distance = FieldDescriptor(location_type=LocationType.Edge, primary_axis=1),

    # double zonal_normal_primal_edge(edge=30715);
    zonal_normal_primal_edge = FieldDescriptor(location_type=LocationType.Edge),

    # double meridional_normal_primal_edge(edge=30715);
    meridional_normal_primal_edge = FieldDescriptor(location_type=LocationType.Edge),

    # double zonal_normal_dual_edge(edge=30715);
    zonal_normal_dual_edge = FieldDescriptor(location_type=LocationType.Edge),

    # double meridional_normal_dual_edge(edge=30715);
    meridional_normal_dual_edge = FieldDescriptor(location_type=LocationType.Edge),

    # int orientation_of_normal(nv=3, cell=20340);
    orientation_of_normal = FieldDescriptor(location_type=LocationType.Cell, primary_axis=1),

    # double clon_vertices(cell=20340, nv=3);
    clon_vertices = FieldDescriptor(location_type=LocationType.Cell, primary_axis=0),

    # double clat_vertices(cell=20340, nv=3);
    clat_vertices = FieldDescriptor(location_type=LocationType.Cell, primary_axis=0),

    # double elon_vertices(edge=30715, no=4);
    elon_vertices = FieldDescriptor(location_type=LocationType.Edge, primary_axis=0),

    # double elat_vertices(edge=30715, no=4);
    elat_vertices = FieldDescriptor(location_type=LocationType.Edge, primary_axis=0),

    # double vlon_vertices(vertex=10376, ne=6);
    vlon_vertices = FieldDescriptor(location_type=LocationType.Vertex, primary_axis=0),

    # double vlat_vertices(vertex=10376, ne=6);
    vlat_vertices = FieldDescriptor(location_type=LocationType.Vertex, primary_axis=0),

    # double quadrilateral_area(edge=30715);
    #   :coordinates = "elon elat";
    quadrilateral_area = FieldDescriptor(location_type=LocationType.Edge),

    # int neighbor_cell_index(nv=3, cell=20340);
    neighbor_cell_index = FieldDescriptor(location_type=LocationType.Cell, indexes_into=LocationType.Cell, primary_axis=1),

    # int edge_orientation(ne=6, vertex=10376);
    edge_orientation = FieldDescriptor(location_type=LocationType.Vertex, primary_axis=1),

    # int edge_system_orientation(edge=30715);
    edge_system_orientation = FieldDescriptor(location_type=LocationType.Edge),

    # int refin_c_ctrl(cell=20340);
    #   :long_name = "refinement control flag for cells";
    #   :cdi = "ignore";
    refin_c_ctrl = FieldDescriptor(location_type=LocationType.Cell),

    # int start_idx_c(max_chdom=1, cell_grf=14);
    #   :long_name = "list of start indices for each refinement control level for cells";
    #   :cdi = "ignore";

    # int end_idx_c(max_chdom=1, cell_grf=14);
    #   :long_name = "list of end indices for each refinement control level for cells";
    #   :cdi = "ignore";

    # int refin_e_ctrl(edge=30715);
    #   :long_name = "refinement control flag for edges";
    #   :cdi = "ignore";
    refin_e_ctrl = FieldDescriptor(location_type=LocationType.Edge),

    # int start_idx_e(max_chdom=1, edge_grf=24);
    #   :long_name = "list of start indices for each refinement control level for edges";
    #   :cdi = "ignore";

    # int end_idx_e(max_chdom=1, edge_grf=24);
    #   :long_name = "list of end indices for each refinement control level for edges";
    #   :cdi = "ignore";

    # int refin_v_ctrl(vertex=10376);
    #   :long_name = "refinement control flag for vertices";
    #   :cdi = "ignore";
    refin_v_ctrl = FieldDescriptor(location_type=LocationType.Vertex),

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
    clon=FieldDescriptor(
        location_type=LocationType.Cell
    ),

    # double clat(cell=9868);
    #   :units = "radian";
    #   :standard_name = "grid_latitude";
    #   :bounds = "clat_vertices";
    clat=FieldDescriptor(
        location_type=LocationType.Cell
    ),

    # double vlon(vertex=5287);
    #   :units = "radian";
    #   :standard_name = "grid_longitude";
    #   :bounds = "vlon_vertices";
    vlon=FieldDescriptor(
        location_type=LocationType.Vertex
    ),

    # double vlat(vertex=5287);
    #   :units = "radian";
    #   :standard_name = "grid_latitude";
    #   :bounds = "vlat_vertices";
    vlat=FieldDescriptor(
        location_type=LocationType.Vertex
    ),

    # double elon(edge=15155);
    #   :units = "radian";
    #   :standard_name = "grid_longitude";
    #   :bounds = "elon_vertices";
    elon=FieldDescriptor(
        location_type=LocationType.Edge
    ),

    # double elat(edge=15155);
    #   :units = "radian";
    #   :standard_name = "grid_latitude";
    #   :bounds = "elat_vertices";
    elat=FieldDescriptor(
        location_type=LocationType.Edge
    ),

    # double longitude_vertices(vertex=5287);
    #   :coordinates = "vlon vlat";
    longitude_vertices=FieldDescriptor(
        location_type=LocationType.Vertex
    ),


    # double latitude_vertices(vertex=5287);
    #   :coordinates = "vlon vlat";
    latitude_vertices=FieldDescriptor(
        location_type=LocationType.Vertex
    ),

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
    cell_area=FieldDescriptor(
        location_type=LocationType.Cell
    ),

    # double lon_edge_centre(edge=15155);
    #   :cdi = "ignore";
    lon_edge_centre=FieldDescriptor(
        location_type=LocationType.Edge
    ),

    # double lat_edge_centre(edge=15155);
    #   :cdi = "ignore";
    lat_edge_centre=FieldDescriptor(
        location_type=LocationType.Edge
    ),

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
    edge_length=FieldDescriptor(
        location_type=LocationType.Edge
    ),

    # double zonal_normal_primal_edge(edge=15155);
    #   :cdi = "ignore";
    zonal_normal_primal_edge=FieldDescriptor(
        location_type=LocationType.Edge
    ),

    # double meridional_normal_primal_edge(edge=15155);
    #   :cdi = "ignore";
    meridional_normal_primal_edge=FieldDescriptor(
        location_type=LocationType.Edge
    ),

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
        indexes_into=LocationType.Cell  #NOTE: this maps into cells of grid.nc, not lateral_boundary.grid.nc
    ),

    # int global_edge_index(edge=15155);
    #   :cdi = "ignore";
    #   :nglobal = 30715; // int
    global_edge_index=FieldDescriptor(
        location_type=LocationType.Edge,
        indexes_into=LocationType.Edge #NOTE: this maps into cells of grid.nc, not lateral_boundary.grid.nc
    ),
)

ICON_grid_schema_ic = make_schema(    
    # double clon(ncells=20340);
    #   :standard_name = "longitude";
    #   :long_name = "center longitude";
    #   :units = "radian";
    #   :bounds = "clon_bnds";
    clon = FieldDescriptor(location_type=LocationType.Cell),

    # double clon_bnds(ncells=20340, vertices=3);
    # clon_bnds = FieldDescriptor(location_type=LocationType.Cell, indexes_into=LocationType.Vertex, primary_axis=0),
    clon_bnds = FieldDescriptor(location_type=LocationType.Cell, primary_axis=0),
    

    # double clat(ncells=20340);
    #   :standard_name = "latitude";
    #   :long_name = "center latitude";
    #   :units = "radian";
    #   :bounds = "clat_bnds";
    clat = FieldDescriptor(location_type=LocationType.Cell),

    # double clat_bnds(ncells=20340, vertices=3);
    # clat_bnds = FieldDescriptor(location_type=LocationType.Cell, indexes_into=LocationType.Vertex, primary_axis=0),    
    clat_bnds = FieldDescriptor(location_type=LocationType.Cell, primary_axis=0),    

    # float HHL(time=1, height=91, ncells=20340);
    #   :long_name = "Geometric Height of the layer limits above sea level(NN)";
    #   :units = "m";
    #   :param = "6.3.0";
    #   :CDI_grid_type = "unstructured";
    #   :number_of_grid_in_reference = 1; // int
    #   :coordinates = "clat clon";
    #   :_FillValue = -8.9999998E15f; // float
    #   :missing_value = -8.9999998E15f; // float
    HHL = FieldDescriptor(location_type=LocationType.Cell, primary_axis=2),

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
    U = FieldDescriptor(location_type=LocationType.Cell, primary_axis=2),

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
    V = FieldDescriptor(location_type=LocationType.Cell, primary_axis=2),

    # float W(time=1, height_3=91, ncells=20340);
    #   :long_name = "Vertical Velocity (Geometric) (w)";
    #   :units = "m s-1";
    #   :param = "9.2.0";
    #   :CDI_grid_type = "unstructured";
    #   :number_of_grid_in_reference = 1; // int
    #   :coordinates = "clat clon";
    #   :_FillValue = -8.9999998E15f; // float
    #   :missing_value = -8.9999998E15f; // float
    W = FieldDescriptor(location_type=LocationType.Cell, primary_axis=2),

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
    T = FieldDescriptor(location_type=LocationType.Cell, primary_axis=2),

    # float P(time=1, height_2=90, ncells=20340);
    #   :long_name = "Pressure";
    #   :units = "Pa";
    #   :param = "0.3.0";
    #   :CDI_grid_type = "unstructured";
    #   :number_of_grid_in_reference = 1; // int
    #   :coordinates = "clat clon";
    #   :_FillValue = -8.9999998E15f; // float
    #   :missing_value = -8.9999998E15f; // float
    P = FieldDescriptor(location_type=LocationType.Cell, primary_axis=2),

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
    QV = FieldDescriptor(location_type=LocationType.Cell, primary_axis=2),

    # float QC(time=1, height_2=90, ncells=20340);
    #   :long_name = "Cloud Mixing Ratio";
    #   :units = "kg kg-1";
    #   :param = "22.1.0";
    #   :CDI_grid_type = "unstructured";
    #   :number_of_grid_in_reference = 1; // int
    #   :coordinates = "clat clon";
    #   :_FillValue = -8.9999998E15f; // float
    #   :missing_value = -8.9999998E15f; // float
    QC = FieldDescriptor(location_type=LocationType.Cell, primary_axis=2),

    # float QI(time=1, height_2=90, ncells=20340);
    #   :long_name = "Cloud Ice Mixing Ratio";
    #   :units = "kg kg-1";
    #   :param = "82.1.0";
    #   :CDI_grid_type = "unstructured";
    #   :number_of_grid_in_reference = 1; // int
    #   :coordinates = "clat clon";
    #   :_FillValue = -8.9999998E15f; // float
    #   :missing_value = -8.9999998E15f; // float
    QI = FieldDescriptor(location_type=LocationType.Cell, primary_axis=2),

    # float QR(time=1, height_2=90, ncells=20340);
    #   :long_name = "Rain mixing ratio";
    #   :units = "kg kg-1";
    #   :param = "24.1.0";
    #   :CDI_grid_type = "unstructured";
    #   :number_of_grid_in_reference = 1; // int
    #   :coordinates = "clat clon";
    #   :_FillValue = -8.9999998E15f; // float
    #   :missing_value = -8.9999998E15f; // float
    QR = FieldDescriptor(location_type=LocationType.Cell, primary_axis=2),

    # float QS(time=1, height_2=90, ncells=20340);
    #   :long_name = "Snow mixing ratio";
    #   :units = "kg kg-1";
    #   :param = "25.1.0";
    #   :CDI_grid_type = "unstructured";
    #   :number_of_grid_in_reference = 1; // int
    #   :coordinates = "clat clon";
    #   :_FillValue = -8.9999998E15f; // float
    #   :missing_value = -8.9999998E15f; // float
    QS = FieldDescriptor(location_type=LocationType.Cell, primary_axis=2),

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
    T_G = FieldDescriptor(location_type=LocationType.Cell, primary_axis=1),

    # float T_ICE(time=1, ncells=20340);
    #   :long_name = "Sea Ice Temperature";
    #   :units = "K";
    #   :param = "8.2.10";
    #   :CDI_grid_type = "unstructured";
    #   :number_of_grid_in_reference = 1; // int
    #   :coordinates = "clat clon";
    #   :_FillValue = -8.9999998E15f; // float
    #   :missing_value = -8.9999998E15f; // float
    T_ICE = FieldDescriptor(location_type=LocationType.Cell, primary_axis=1),

    # float H_ICE(time=1, ncells=20340);
    #   :long_name = "Sea Ice Thickness";
    #   :units = "m";
    #   :param = "1.2.10";
    #   :CDI_grid_type = "unstructured";
    #   :number_of_grid_in_reference = 1; // int
    #   :coordinates = "clat clon";
    #   :_FillValue = -8.9999998E15f; // float
    #   :missing_value = -8.9999998E15f; // float
    H_ICE = FieldDescriptor(location_type=LocationType.Cell, primary_axis=1),

    # float T_MNW_LK(time=1, ncells=20340);
    #   :long_name = "Mean temperature of the water column";
    #   :units = "K";
    #   :param = "1.2.1";
    #   :CDI_grid_type = "unstructured";
    #   :number_of_grid_in_reference = 1; // int
    #   :coordinates = "clat clon";
    #   :_FillValue = -8.9999998E15f; // float
    #   :missing_value = -8.9999998E15f; // float
    T_MNW_LK = FieldDescriptor(location_type=LocationType.Cell, primary_axis=1),

    # float T_WML_LK(time=1, ncells=20340);
    #   :long_name = "Mixed-layer temperature";
    #   :units = "K";
    #   :param = "1.2.1";
    #   :CDI_grid_type = "unstructured";
    #   :number_of_grid_in_reference = 1; // int
    #   :coordinates = "clat clon";
    #   :_FillValue = -8.9999998E15f; // float
    #   :missing_value = -8.9999998E15f; // float
    T_WML_LK = FieldDescriptor(location_type=LocationType.Cell, primary_axis=1),

    # float H_ML_LK(time=1, ncells=20340);
    #   :long_name = "Mixed-layer depth";
    #   :units = "m";
    #   :param = "0.2.1";
    #   :CDI_grid_type = "unstructured";
    #   :number_of_grid_in_reference = 1; // int
    #   :coordinates = "clat clon";
    #   :_FillValue = -8.9999998E15f; // float
    #   :missing_value = -8.9999998E15f; // float
    H_ML_LK = FieldDescriptor(location_type=LocationType.Cell, primary_axis=1),

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
    T_BOT_LK = FieldDescriptor(location_type=LocationType.Cell, primary_axis=1),

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
    C_T_LK = FieldDescriptor(location_type=LocationType.Cell, primary_axis=1),

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
    QV_S = FieldDescriptor(location_type=LocationType.Cell, primary_axis=1),

    # float T_SO(time=1, depth=9, ncells=20340);
    #   :long_name = "Soil Temperature (multilayer model)";
    #   :units = "K";
    #   :param = "18.3.2";
    #   :CDI_grid_type = "unstructured";
    #   :number_of_grid_in_reference = 1; // int
    #   :coordinates = "clat clon";
    #   :_FillValue = -8.9999998E15f; // float
    #   :missing_value = -8.9999998E15f; // float
    T_SO = FieldDescriptor(location_type=LocationType.Cell, primary_axis=2),

    # float W_SO(time=1, depth_2=8, ncells=20340);
    #   :long_name = "Column-integrated Soil Moisture (multilayers)";
    #   :units = "kg m-2";
    #   :param = "20.3.2";
    #   :CDI_grid_type = "unstructured";
    #   :number_of_grid_in_reference = 1; // int
    #   :coordinates = "clat clon";
    #   :_FillValue = -8.9999998E15f; // float
    #   :missing_value = -8.9999998E15f; // float
    W_SO = FieldDescriptor(location_type=LocationType.Cell, primary_axis=2),

    # float W_SO_ICE(time=1, depth_2=8, ncells=20340);
    #   :long_name = "soil ice content (multilayers)";
    #   :units = "kg m-2";
    #   :param = "22.3.2";
    #   :CDI_grid_type = "unstructured";
    #   :number_of_grid_in_reference = 1; // int
    #   :coordinates = "clat clon";
    #   :_FillValue = -8.9999998E15f; // float
    #   :missing_value = -8.9999998E15f; // float
    W_SO_ICE = FieldDescriptor(location_type=LocationType.Cell, primary_axis=2),

    # float ALB_SEAICE(time=1, ncells=20340);
    #   :long_name = "sea ice albedo - diffusive solar (0.3 - 5.0 m-6)";
    #   :units = "%";
    #   :param = "234.19.0";
    #   :CDI_grid_type = "unstructured";
    #   :number_of_grid_in_reference = 1; // int
    #   :coordinates = "clat clon";
    #   :_FillValue = -8.9999998E15f; // float
    #   :missing_value = -8.9999998E15f; // float
    ALB_SEAICE = FieldDescriptor(location_type=LocationType.Cell, primary_axis=1),

    # float EVAP_PL(time=1, ncells=20340);
    #   :long_name = "evaporation of plants (integrated since \"nightly reset\")";
    #   :units = "kgm-2";
    #   :param = "198.0.2";
    #   :CDI_grid_type = "unstructured";
    #   :number_of_grid_in_reference = 1; // int
    #   :coordinates = "clat clon";
    #   :_FillValue = -8.9999998E15f; // float
    #   :missing_value = -8.9999998E15f; // float
    EVAP_PL = FieldDescriptor(location_type=LocationType.Cell, primary_axis=1),

    # float SMI(time=1, depth_2=8, ncells=20340);
    #   :long_name = "soil moisture index (multilayers)";
    #   :units = "Numeric";
    #   :param = "200.3.2";
    #   :CDI_grid_type = "unstructured";
    #   :number_of_grid_in_reference = 1; // int
    #   :coordinates = "clat clon";
    #   :_FillValue = -8.9999998E15f; // float
    #   :missing_value = -8.9999998E15f; // float
    SMI = FieldDescriptor(location_type=LocationType.Cell, primary_axis=2),

    # float RHO_SNOW(time=1, ncells=20340);
    #   :long_name = "Snow density";
    #   :units = "kg m-3";
    #   :param = "61.1.0";
    #   :CDI_grid_type = "unstructured";
    #   :number_of_grid_in_reference = 1; // int
    #   :coordinates = "clat clon";
    #   :_FillValue = -8.9999998E15f; // float
    #   :missing_value = -8.9999998E15f; // float
    RHO_SNOW = FieldDescriptor(location_type=LocationType.Cell, primary_axis=1),

    # float T_SNOW(time=1, ncells=20340);
    #   :long_name = "Snow temperature (top of snow)";
    #   :units = "K";
    #   :param = "18.0.0";
    #   :CDI_grid_type = "unstructured";
    #   :number_of_grid_in_reference = 1; // int
    #   :coordinates = "clat clon";
    #   :_FillValue = -8.9999998E15f; // float
    #   :missing_value = -8.9999998E15f; // float
    T_SNOW = FieldDescriptor(location_type=LocationType.Cell, primary_axis=1),

    # float W_SNOW(time=1, ncells=20340);
    #   :long_name = "Snow depth water equivalent";
    #   :units = "kg m-2";
    #   :param = "60.1.0";
    #   :CDI_grid_type = "unstructured";
    #   :number_of_grid_in_reference = 1; // int
    #   :coordinates = "clat clon";
    #   :_FillValue = -8.9999998E15f; // float
    #   :missing_value = -8.9999998E15f; // float
    W_SNOW = FieldDescriptor(location_type=LocationType.Cell, primary_axis=1),

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
    H_SNOW = FieldDescriptor(location_type=LocationType.Cell, primary_axis=1),

    # float W_I(time=1, ncells=20340);
    #   :long_name = "Plant Canopy Surface Water";
    #   :units = "kg m-2";
    #   :param = "13.0.2";
    #   :CDI_grid_type = "unstructured";
    #   :number_of_grid_in_reference = 1; // int
    #   :coordinates = "clat clon";
    #   :_FillValue = -8.9999998E15f; // float
    #   :missing_value = -8.9999998E15f; // float
    W_I = FieldDescriptor(location_type=LocationType.Cell, primary_axis=1),

    # float FRESHSNW(time=1, ncells=20340);
    #   :long_name = "Fresh snow factor (weighting function for albedo indicating freshness of snow)";
    #   :units = "Numeric";
    #   :param = "203.1.0";
    #   :CDI_grid_type = "unstructured";
    #   :number_of_grid_in_reference = 1; // int
    #   :coordinates = "clat clon";
    #   :_FillValue = -8.9999998E15f; // float
    #   :missing_value = -8.9999998E15f; // float
    FRESHSNW = FieldDescriptor(location_type=LocationType.Cell, primary_axis=1),

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
    Z0 = FieldDescriptor(location_type=LocationType.Cell, primary_axis=1),

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
    FR_ICE = FieldDescriptor(location_type=LocationType.Cell, primary_axis=1),
)

#NOTE: this is probably not needed since we are not reordering lateral_boundary.grid.nc (but the indexes therein)
ICON_grid_schema_lbc = make_schema(
    # double clon(ncells=9868);
    #   :standard_name = "longitude";
    #   :long_name = "center longitude";
    #   :units = "radian";
    #   :bounds = "clon_bnds";
    clon = FieldDescriptor(location_type=LocationType.Cell),

    # double clon_bnds(ncells=9868, vertices=3);
    # clon_bnds = FieldDescriptor(location_type=LocationType.Cell, indexes_into=LocationType.Vertex, primary_axis=0),
    clon_bnds = FieldDescriptor(location_type=LocationType.Cell, primary_axis=0),

    # double clat(ncells=9868);
    #   :standard_name = "latitude";
    #   :long_name = "center latitude";
    #   :units = "radian";
    #   :bounds = "clat_bnds";
    clat = FieldDescriptor(location_type=LocationType.Cell),

    # double clat_bnds(ncells=9868, vertices=3);
    # clat_bnds = FieldDescriptor(location_type=LocationType.Cell, indexes_into=LocationType.Vertex, primary_axis=0),    
    clat_bnds = FieldDescriptor(location_type=LocationType.Cell, primary_axis=0),    

    # float HHL(time=1, height=91, ncells=9868);
    #   :long_name = "Geometric Height of the layer limits above sea level(NN)";
    #   :units = "m";
    #   :param = "6.3.0";
    #   :CDI_grid_type = "unstructured";
    #   :number_of_grid_in_reference = 1; // int
    #   :coordinates = "clat clon";
    #   :_FillValue = -8.9999998E15f; // float
    #   :missing_value = -8.9999998E15f; // float
    HHL = FieldDescriptor(location_type=LocationType.Cell, primary_axis=2),

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
    U = FieldDescriptor(location_type=LocationType.Cell, primary_axis=2),

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
    V = FieldDescriptor(location_type=LocationType.Cell, primary_axis=2),

    # float W(time=1, height_3=91, ncells=9868);
    #   :long_name = "Vertical Velocity (Geometric) (w)";
    #   :units = "m s-1";
    #   :param = "9.2.0";
    #   :CDI_grid_type = "unstructured";
    #   :number_of_grid_in_reference = 1; // int
    #   :coordinates = "clat clon";
    #   :_FillValue = -8.9999998E15f; // float
    #   :missing_value = -8.9999998E15f; // float
    W = FieldDescriptor(location_type=LocationType.Cell, primary_axis=2),

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
    T = FieldDescriptor(location_type=LocationType.Cell, primary_axis=2),

    # float P(time=1, height_2=90, ncells=9868);
    #   :long_name = "Pressure";
    #   :units = "Pa";
    #   :param = "0.3.0";
    #   :CDI_grid_type = "unstructured";
    #   :number_of_grid_in_reference = 1; // int
    #   :coordinates = "clat clon";
    #   :_FillValue = -8.9999998E15f; // float
    #   :missing_value = -8.9999998E15f; // float
    P = FieldDescriptor(location_type=LocationType.Cell, primary_axis=2),

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
    QV = FieldDescriptor(location_type=LocationType.Cell, primary_axis=2),

    # float QC(time=1, height_2=90, ncells=9868);
    #   :long_name = "Cloud Mixing Ratio";
    #   :units = "kg kg-1";
    #   :param = "22.1.0";
    #   :CDI_grid_type = "unstructured";
    #   :number_of_grid_in_reference = 1; // int
    #   :coordinates = "clat clon";
    #   :_FillValue = -8.9999998E15f; // float
    #   :missing_value = -8.9999998E15f; // float
    QC = FieldDescriptor(location_type=LocationType.Cell, primary_axis=2),

    # float QI(time=1, height_2=90, ncells=9868);
    #   :long_name = "Cloud Ice Mixing Ratio";
    #   :units = "kg kg-1";
    #   :param = "82.1.0";
    #   :CDI_grid_type = "unstructured";
    #   :number_of_grid_in_reference = 1; // int
    #   :coordinates = "clat clon";
    #   :_FillValue = -8.9999998E15f; // float
    #   :missing_value = -8.9999998E15f; // float
    QI = FieldDescriptor(location_type=LocationType.Cell, primary_axis=2),

    # float QR(time=1, height_2=90, ncells=9868);
    #   :long_name = "Rain mixing ratio";
    #   :units = "kg kg-1";
    #   :param = "24.1.0";
    #   :CDI_grid_type = "unstructured";
    #   :number_of_grid_in_reference = 1; // int
    #   :coordinates = "clat clon";
    #   :_FillValue = -8.9999998E15f; // float
    #   :missing_value = -8.9999998E15f; // float
    QR = FieldDescriptor(location_type=LocationType.Cell, primary_axis=2),

    # float QS(time=1, height_2=90, ncells=9868);
    #   :long_name = "Snow mixing ratio";
    #   :units = "kg kg-1";
    #   :param = "25.1.0";
    #   :CDI_grid_type = "unstructured";
    #   :number_of_grid_in_reference = 1; // int
    #   :coordinates = "clat clon";
    #   :_FillValue = -8.9999998E15f; // float
    #   :missing_value = -8.9999998E15f; // float
    QS = FieldDescriptor(location_type=LocationType.Cell, primary_axis=2),
)

def apply_permutation(
    ncf,
    perm: np.ndarray,
    schema: GridScheme,
    location_type: LocationType
) -> None:
    rev_perm = revert_permutation(perm)

    for field_name, descr in schema.items():

        if descr.indexes_into is not None:
            continue

        field = ncf.variables[field_name]
        array = np.copy(field[:])

        if descr.location_type is location_type:
            if 1 < len(array.shape):
                assert descr.primary_axis is not None
                array = np.take(array, perm, axis=descr.primary_axis)
            else:
                array = array[perm]

        if descr.indexes_into is location_type:
            # go from fortran's 1-based indexing to python's 0-based indexing
            array = array - 1

            # remap indices
            missing_values = array == DEVICE_MISSING_VALUE
            array = rev_perm[array]
            array[missing_values] = DEVICE_MISSING_VALUE

            array = array + 1

        field[:] = array

def apply_permutation_nbh_tables(
    ncf,
    perm_dict,
    schema: GridScheme,    
) -> None:    

    for field_name, descr in schema.items():

        if descr.indexes_into is not None:

            print(field_name)

            field = ncf.variables[field_name]
            original_table = np.copy(field[:]) - 1      
            
            outer_perm = perm_dict[descr.location_type]
            outer_perm_rev = revert_permutation(outer_perm)

            inner_perm = perm_dict[descr.indexes_into]
            inner_perm_rev = revert_permutation(inner_perm)
            
            perm_table = np.take(original_table, outer_perm_rev, axis=descr.primary_axis)  
            
            #FIXME: bad code, assumes shape of nbh table, invalidates primary axis
            for i in range (0, np.shape(perm_table)[1]):
                for j in range(0, np.shape(perm_table)[0]):
                    if perm_table[j][i] == DEVICE_MISSING_VALUE:
                        continue
                    perm_table[j][i] = inner_perm_rev[perm_table[j][i]]
            
            field[:] = perm_table + 1

def apply_permutation_latbc_grid(
    ncf,
    perm_cells: np.ndarray,
    perm_edges: np.ndarray,
    schema: GridScheme,    
) -> None:
    for field_name, descr in schema.items():           
        field = ncf.variables[field_name]
        array = np.copy(field[:])

        if field_name == "global_edge_index":            
            rev_perm = revert_permutation(perm_edges)
            array = array - 1
            field[:] = rev_perm[array] + 1            
            # for i in range(0, len(array)):
            #     array[i] = perm_edges[array[i]]
            # field[:] = array + 1

        if field_name == "global_cell_index":
            rev_perm = revert_permutation(perm_cells)
            array = array - 1
            field[:] = rev_perm[array] + 1            
            # for i in range(0, len(array)):
            #     array[i] = perm_cells[array[i]]
            # field[:] = array + 1

def order_around(center, points):
    centered_points = points - center
    return points[np.argsort(np.arctan2(centered_points[:, 0], centered_points[:, 1]))]


def plot_cells(ax, grid: Grid, field=None):

    if field is None:
        field = np.zeros(grid.nc)

    triangles = []
    for ci in range(grid.nc):

        vs = grid.c2v[ci]
        assert np.all((0 <= vs) & (vs < grid.nv))

        vs = grid.v_lon_lat[vs]
        triangles.append(patches.Polygon(vs))

    collection = collections.PatchCollection(triangles)
    collection.set_array(field)
    ax.add_collection(collection)

    return collection


def plot_vertices(ax, grid: Grid, field=None):

    if field is None:
        field = np.zeros(grid.nv)

    hexagons = []
    for vi in range(grid.nv):

        cs = grid.v2c[vi]
        if np.all((0 <= cs) & (cs < grid.nc)):
            points = grid.c_lon_lat[cs]
            points = order_around(grid.v_lon_lat[vi].reshape(1, 2), points)
        else:
            # some of our neighbors are outside the field
            # we have to draw a partial hexagon
            cs = cs[cs != DEVICE_MISSING_VALUE]
            es = grid.v2e[vi]
            es = es[es != DEVICE_MISSING_VALUE]

            vertex = grid.v_lon_lat[vi].reshape(1, 2)
            points = np.concatenate((grid.c_lon_lat[cs], grid.e_lon_lat[es], vertex), axis=0)
            center = np.average(points, axis=0)
            points = order_around(center, points)

        hexagons.append(patches.Polygon(points))

    collection = collections.PatchCollection(hexagons)
    collection.set_array(field)
    ax.add_collection(collection)

    return collection


def plot_edges(ax, grid: Grid, field=None):

    if field is None:
        field = np.zeros(grid.nv)

    diamonds = []
    for ei in range(grid.ne):

        vs = grid.e2v[ei]
        cs = grid.e2c[ei]

        if cs[1] == DEVICE_MISSING_VALUE:
            # one of our neighboring cells is outside the grid
            # we just draw half a diamond
            cs = cs[:1]

        assert np.all((0 <= vs) & (vs < grid.nv)) and np.all((0 <= cs) & (cs < grid.nc))

        points = np.concatenate((grid.v_lon_lat[vs], grid.c_lon_lat[cs]), axis=0)
        points = order_around(grid.e_lon_lat[ei].reshape(1, 2), points)
        diamonds.append(patches.Polygon(points))

    collection = collections.PatchCollection(diamonds)
    collection.set_array(field)
    ax.add_collection(collection)

    return collection


def get_grf_ranges(grid: Grid, location_type: LocationType = LocationType.Cell):

    # returns the index ranges of the grid refinement valid_regions
    # region 0 is the compute domain.
    # all other regions are the lateral boundary layers starting from most outer
    # and going to most inner.

    if location_type is LocationType.Vertex:
        n = grid.nv
        start, end = grid.v_grf[:, 0], grid.v_grf[:, 1]
    elif location_type is LocationType.Edge:
        n = grid.ne
        start, end = grid.e_grf[:, 0], grid.e_grf[:, 1]
    elif location_type is LocationType.Cell:
        n = grid.nc
        start, end = grid.c_grf[:, 0], grid.c_grf[:, 1]
    else:
        raise ValueError

    valid_regions = start <= end
    start = start[valid_regions]
    end = end[valid_regions]
    end = end + 1  # end is exclusive

    assert np.min(start) == 0
    assert np.max(end) <= n

    # There's something very weird going on:
    # Some few vertices/edges/cells (at the end) aren't in any valid region,
    # but without them, there will be a hole in the compute domain.
    # We fix this by assigning them to region `0` by default.

    end[0] = n

    return list(zip(start, end))


def range_to_slice(range: typing.Tuple[typing.Optional[int], typing.Optional[int]]):
    return slice(range[0], range[1])


def normalize_angle(angle):
    return np.fmod(angle, 2*np.pi)


def get_angle(p):
    return np.arctan2(p[:, 1], p[:, 0])


def rotate(points, angle, origin=np.array([[0, 0]])):
    points = points - origin
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle) ]
    ])
    points = (rotation_matrix @ points.T).T
    return points + origin


class UnsupportedPentagonException(Exception):
    pass


def neighbor_array_to_set(array):
    nbhs = np.unique(array)
    return nbhs[nbhs != DEVICE_MISSING_VALUE]


###############################################################################
# Each vertex is the crossing point of 6 rays. Two of twos rays are defined as
# the cartesian x- and y-axis (marked as double lines below).
# With this, we can give each vertex a unique x/y coordinate, as shown below.
#
#                 2                                       1
#                   \                                  //
#                     \                              //
#                       \                          //
#                 [-1, 1] *-----------------------* [0, 1]
#                       /   \                  //   \
#                     /       \              //       \
#                   /           \          //           \
#                 /               \      //               \
#       [-1, 0] /                   \  // [0, 0]            \ [1, 0]
# 3 ===========*======================*======================*=============== 0
#               \                   //  \                   /
#                 \               //      \               /
#                   \           //          \           /
#                     \       //              \       /
#                       \   //                  \   /
#                 [0, -1] *-----------------------* [1, -1]
#                       //                          \
#                     //                              \
#                   //                                  \
#                 4                                       5
#
###############################################################################
structured_v2v_offsets = np.array([
    # neighbor id/ray 0
    [+1, +0],
    # neighbor id/ray 1
    [+0, +1],
    # neighbor id/ray 2
    [-1, +1],
    # neighbor id/ray 3
    [-1, +0],
    # neighbor id/ray 4
    [+0, -1],
    # neighbor id/ray 5
    [+1, -1],
], dtype=int)

###############################################################################
# Once each vertex has a unique x/y coordinate, we use those to assign
# to each edge & cell a x/y coordinate and a color. For each edge & cell
# we look for the closest vertex in the bottom left direction. This vertex
# determines the x/y coordinate of each edge & cell. Then the coloring is done
# from left to right in a counter clock-wise direction.
# (This is similar to dawn's `ICOChainSize`, but uses a slightly different ordering)
#
#        /                          \       /                                  /
# \    /                             \    /                                  /
#  \ /                                \ /                                  /
#  -*------------------------ [x, y+1] *==================================* [x+1, y+1]
#  / \                               // \                               // \
#     \                            //    \                            //    \
#      \                         //       \        [x, y, 1]        //       \
#       \                      //          \                      //          \
#        \                   //             \                   //             \
#         \             [x, y, 0]        [x, y, 1]            //                \
#          \             //                   \             //                   \
#           \          //                      \          //                      \
#            \       //       [x, y, 0]         \       //
#             \    //                            \    //
#              \ //                               \ //
#   ---- [x, y] *============[x, y, 2]=============* [x+1, y] ---------------------
#              / \                                / \
#            /    \                             /    \
#          /       \                          /       \
#        /          \                       /          \
#
###############################################################################
structured_v2e_offsets = np.array([
    # neighbor id/ray 0
    [+0, +0, +2],
    # neighbor id/ray 1
    [+0, +0, +0],
    # neighbor id/ray 2
    [-1, +0, +1],
    # neighbor id/ray 3
    [-1, +0, +2],
    # neighbor id/ray 4
    [+0, -1, +0],
    # neighbor id/ray 5
    [+0, -1, +1],
], dtype=int)
# (for the cells, we shift the rays 15 degrees counter clock-wise)
structured_v2c_offsets = np.array([
    # neighbor id/ray 0
    [+0, +0, +0],
    # neighbor id/ray 1
    [-1, +0, +1],
    # neighbor id/ray 2
    [-1, +0, +0],
    # neighbor id/ray 3
    [-1, -1, +1],
    # neighbor id/ray 4
    [+0, -1, +0],
    # neighbor id/ray 5
    [+0, -1, +1],
], dtype=int)


@dataclasses.dataclass
class GridMapping:
    vertex_mapping: np.ndarray
    edge_mapping: np.ndarray
    cell_mapping: np.ndarray


def create_structured_grid_mapping(
    grid: Grid,
    right_direction_angle,
    start_vertex=None,
    angle_threshold=np.deg2rad(30)
) -> GridMapping:
    # doesn't support pentagons!

    if start_vertex is None:
        start_vertex = 0

    if isinstance(right_direction_angle, np.ndarray):
        right_direction_angle = float(right_direction_angle)

    vertex_mapping = np.full((grid.nv, 2), NaN)
    edge_mapping = np.full((grid.ne, 3), NaN)
    cell_mapping = np.full((grid.nc, 3), NaN)

    vertex_mapping[start_vertex] = [0, 0]

    # This algorithms works as follows:
    #
    # * Carry out a breadth-first search starting from `start_vertex`.
    # * For each vertex:
    #   * Determine for each neighbor edge, cell, vertex what is their relative id.
    #     (see `structured_<...>_offsets`)
    #   * For each neighbor edge, cell, vertex check which ones have no coordinates assigned yet:
    #     * Assign new coordinates to neighbors if they don't have coordinates yet.
    #     * Check if coordinates are consistent if they already have coordinates.
    #   * Update the right direction angle based on the neighboring vertices of the vertex
    #     (this way the algorithm can handle a small local curvature)
    #   * Continue the bsf with the vertices that have newly assigned coordinates.

    def bfs(vertex_id, right_direction_angle):

        # neighbor cells, edges, vertices
        cell_ids = neighbor_array_to_set(grid.v2c[vertex_id])
        edge_ids = neighbor_array_to_set(grid.v2e[vertex_id])
        vertex_ids = neighbor_array_to_set(grid.e2v[edge_ids])
        vertex_ids = vertex_ids[vertex_ids != vertex_id]

        # some sanity checks
        if len(edge_ids) == 5 and len(cell_ids) == 5:
            raise UnsupportedPentagonException

        assert len(edge_ids) == len(cell_ids) == 6 or len(cell_ids) + 1 == len(edge_ids)
        assert len(vertex_ids) == len(edge_ids)
        assert 0 < len(cell_ids) <= len(edge_ids) <= 6

        # get the coordinates of this vertex
        x, y = vertex_mapping[vertex_id]

        assert not np.isnan(x) and not np.isnan(y)

        self_lon_lat = grid.v_lon_lat[vertex_id]

        # compute angles of neighbor vertices
        vertices_angle = normalize_angle(get_angle(grid.v_lon_lat[vertex_ids] - self_lon_lat) - right_direction_angle)
        vertices_nbh_ids = np.around(vertices_angle/(np.pi/3)).astype(int)
        assert np.all(np.fabs(vertices_angle - vertices_nbh_ids*np.pi/3) <= angle_threshold)

        # compute angles of neighbor edges
        edges_angle = normalize_angle(get_angle(grid.e_lon_lat[edge_ids] - self_lon_lat) - right_direction_angle)
        edges_nbh_ids = np.around(edges_angle/(np.pi/3)).astype(int)
        assert np.all(np.fabs(edges_angle - edges_nbh_ids*np.pi/3) <= angle_threshold)

        # compute angles of neighbor cells
        # (we rotate the cells by 30 degrees clock-wise (`-np.pi/6`) to get the angle id)
        cells_angle = normalize_angle(get_angle(grid.c_lon_lat[cell_ids] - self_lon_lat) - right_direction_angle - np.pi/6)
        cells_nbh_ids = np.around(cells_angle/(np.pi/3)).astype(int)
        assert np.all(np.fabs(cells_angle - cells_nbh_ids*np.pi/3) <= angle_threshold)

        # update right direction angle
        self_right_direction_angle = np.average(vertices_angle - vertices_nbh_ids*np.pi/3) + right_direction_angle

        # assign coordinates to vertex neighbors that don't have a coordinate yet
        vertices_nbh_structured_coords = structured_v2v_offsets[vertices_nbh_ids] + np.array([[x, y]], dtype=int)
        new_vertex_ids = np.all(np.isnan(vertex_mapping[vertex_ids, :]), axis=-1)
        vertex_mapping[vertex_ids[new_vertex_ids], :] = vertices_nbh_structured_coords[new_vertex_ids]
        # check vertex neighbors that already had a coordinate, that they are consistent with the ones we computed here
        assert np.all(vertex_mapping[vertex_ids, :] == vertices_nbh_structured_coords)

        # assign coordinates to edge neighbors that don't have a coordinate yet
        edges_nbh_structured_coords = structured_v2e_offsets[edges_nbh_ids] + np.array([[x, y, 0]], dtype=int)
        new_edge_ids = np.all(np.isnan(edge_mapping[edge_ids, :]), axis=-1)
        edge_mapping[edge_ids[new_edge_ids], :] = edges_nbh_structured_coords[new_edge_ids]
        # check edge neighbors that already had a coordinate, that they are consistent with the ones we computed here
        assert np.all(edge_mapping[edge_ids, :] == edges_nbh_structured_coords)

        # assign coordinates to cell neighbors that don't have a coordinate yet
        cells_nbh_structured_coords = structured_v2c_offsets[cells_nbh_ids] + np.array([[x, y, 0]], dtype=int)
        new_cell_ids = np.all(np.isnan(cell_mapping[cell_ids, :]), axis=-1)
        cell_mapping[cell_ids[new_cell_ids], :] = cells_nbh_structured_coords[new_cell_ids]
        # check cell neighbors that already had a coordinate, that they are consistent with the ones we computed here
        assert np.all(cell_mapping[cell_ids, :] == cells_nbh_structured_coords)

        # continue bfs with vertices that have newly assigned coordinates
        # (use the updated right direction angle for them)
        return {(int(next_vertex_id), self_right_direction_angle) for next_vertex_id in vertex_ids[new_vertex_ids]}

    current = set()
    next = {(start_vertex, right_direction_angle)}

    while 0 < len(next):
        # swap
        current, next = next, current
        next.clear()

        for vertex_args in current:
            next.update(bfs(*vertex_args))

    assert not np.any(np.isnan(vertex_mapping))
    assert not np.any(np.isnan(edge_mapping))
    assert not np.any(np.isnan(cell_mapping))

    return GridMapping(
        vertex_mapping=vertex_mapping,
        edge_mapping=edge_mapping,
        cell_mapping=cell_mapping,
    )


def argsort_simple(
    mapping: np.ndarray,
    cmp: typing.Callable[[typing.Any, typing.Any], int],
    idx_range: typing.Tuple[typing.Optional[int], typing.Optional[int]] = (None, None),
) -> np.ndarray:
    # Sorts the first axis based on a `cmp` function within the range [start_idx:end_idx].
    # Returns the permutation array for the whole array.
    #
    # A permutation is an array `a` such that: `a[old_index] == new_index`

    total_end_idx = mapping.shape[0]
    start_idx, end_idx = idx_range

    if start_idx is None:
        start_idx = 0

    if end_idx is None:
        end_idx = total_end_idx

    ids = list(range(start_idx, end_idx))
    ids.sort(key=cmp_to_key(lambda a, b: cmp(mapping[a, :], mapping[b, :])))
    return np.concatenate((np.arange(start_idx), np.array(ids), np.arange(end_idx, total_end_idx)))


def revert_permutation(perm: np.ndarray) -> np.ndarray:
    perm_rev = np.arange(perm.shape[0])
    perm_rev[perm] = np.copy(perm_rev)
    return perm_rev


class SimpleRowMajorSorting:
    # Provides comparison functions for mappings from `create_structured_grid_mapping`.

    @staticmethod
    def vertex_compare(a, b) -> int:
        return a[0] - b[0] if b[1] == a[1] else b[1] - a[1]

    @staticmethod
    def edge_compare(a, b) -> int:
        if a[2] == 2 and b[2] != 2:
            return b[1] - a[1] + 1/2
        if a[2] != 2 and b[2] == 2:
            return b[1] - a[1] - 1/2
        return (a[2] - b[2] if a[0] == b[0] else a[0] - b[0]) if b[1] == a[1] else b[1] - a[1]

    @staticmethod
    def cell_compare(a, b) -> int:
        return (a[2] - b[2] if a[0] == b[0] else a[0] - b[0]) if b[1] == a[1] else b[1] - a[1]


TEMP_FILE_NAME = "./temp.nc"
TEMP_FILE_NAME_LAT = "./temp_lat.nc"

def test_vertex(grid: Grid, grid_modified: Grid, v_perm: np.ndarray, c_perm: np.ndarray):

    c_perm_rev = revert_permutation(c_perm)
    v_perm_rev = revert_permutation(v_perm)

    v0 = np.take(grid.v2c, v_perm, axis=0)  
    v1 = np.take(grid.v2c, v_perm_rev, axis=0) 
    v2 = grid.v2c[v_perm]
    v3 = grid.v2c[v_perm_rev]
    #
    # v0 == v2 and v1 == v3 
    
    my_v2c = np.take(grid.v2c, v_perm_rev, axis=0)  #this leads to correct results, but why?   
    test = grid.v2c[v_perm]    
    for i in range (0, grid.nv):
        for j in range(0, 6):
            if my_v2c[i][j] == DEVICE_MISSING_VALUE:
                continue
            my_v2c[i][j] = c_perm_rev[my_v2c[i][j]]

    num_errors = 0
    max_errors = 10
    for i in range (0,grid.nv):         

        sum_original = 0
        nbhood = grid.v2c[i]
        for j in range(0,6):
            nbh = nbhood[j]
            if nbh == DEVICE_MISSING_VALUE:
                continue
            sum_original += grid.c_lon_lat[nbh][0]

        sum_modified = 0
        new_idx = v_perm[i]
        nbhood_modifed = grid_modified.v2c[new_idx]
        for j in range(0,6):
            nbh = nbhood_modifed[j]
            if nbh == DEVICE_MISSING_VALUE:
                continue
            sum_modified += grid_modified.c_lon_lat[nbh][0]

        my_sum = 0
        new_idx = v_perm[i]       
        for j in range(0,6):
            nbh = my_v2c[new_idx][j]
            if nbh == DEVICE_MISSING_VALUE:
                continue
            my_sum += grid_modified.c_lon_lat[nbh][0]
        
        if not np.isclose(sum_original,sum_modified):
            print(f"idx: {i} org: {sum_original} mod: {sum_modified} my: {my_sum}")
            num_errors = num_errors + 1

        if num_errors > max_errors:
            break

        # assert np.isclose(sum_original,sum_modified)



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

    perm_dict = {LocationType.Cell : c_perm, LocationType.Edge : e_perm, LocationType.Vertex : v_perm}

    apply_permutation_nbh_tables(grid_modified_file, perm_dict, ICON_grid_schema_dbg)

    grid_modified_file.sync()

    grid_file = netCDF4.Dataset("grid_row-major.nc")
    grid_modified = Grid.from_netCDF4(grid_file)    

    test_vertex(grid, grid_modified, v_perm, c_perm)


def main():
    # copy files from pool directory
    shutil.copy("../my_pool/data/ICON/mch/grids/ch_r04b09/grid_icon.nc", "grid.nc")
    shutil.copy("../my_pool/data/ICON/mch/grids/ch_r04b09/grid.parent_icon.nc", "grid.parent.nc")
    shutil.copy("../my_pool/data/ICON/mch/input/ch_r04b09/lateral_boundary.grid_icon.nc", "lateral_boundary.grid.nc")
    shutil.copy("../my_pool/data/ICON/mch/input/ch_r04b09/igfff00000000_icon.nc", "igfff00000000.nc")
    # shutil.copy("../my_pool/data/ICON/mch/input/ch_r04b09/igfff00000000_lbc_icon.nc", "igfff00000000_lbc.nc")
    # shutil.copy("../my_pool/data/ICON/mch/input/ch_r04b09/igfff00030000_lbc_icon.nc", "igfff00030000_lbc.nc")

    grid_file = netCDF4.Dataset("grid.nc")
    grid = Grid.from_netCDF4(grid_file)
    parent_grid_file = netCDF4.Dataset("grid.parent.nc")
    parent_grid = Grid.from_netCDF4(parent_grid_file)
    # lateral_grid_file = netCDF4.Dataset("lateral_boundary.grid.nc")
    # lateral_grid = Grid.lat_from_netCDF4(lateral_grid_file)
    
    # init_con_grid_file = netCDF4.Dataset("igfff00000000.nc")
    # lbc_00000000_file = netCDF4.Dataset("igfff00000000_lbc.nc")
    # lbc_00030000_file = netCDF4.Dataset("igfff00030000_lbc.nc")   
   
    write_back = True
    if write_back:
        shutil.copy("./grid.nc", "./grid_row-major.nc")
        grid_modified_file = netCDF4.Dataset("./grid_row-major.nc", "r+")
        shutil.copy("./grid.parent.nc", "./grid.parent_row-major.nc")
        parent_grid_modified_file = netCDF4.Dataset("./grid.parent_row-major.nc", "r+")
        shutil.copy("./lateral_boundary.grid.nc", "./lateral_boundary.grid_row-major.nc")
        lateral_grid_modified_file = netCDF4.Dataset("./lateral_boundary.grid_row-major.nc", "r+")
        shutil.copy("./igfff00000000.nc", "./igfff00000000_row-major.nc")
        init_con_grid_file_modified_file = netCDF4.Dataset("./igfff00000000_row-major.nc", "r+")
        # shutil.copy("./igfff00000000_lbc.nc", "./igfff00000000_lbc_row-major.nc")
        # lbc_00000000_file_modified_file = netCDF4.Dataset("./lateral_boundary.grid_row-major.nc", "r+")
        # shutil.copy("./igfff00030000_lbc.nc", "./igfff00030000_lbc_row-major.nc")
        # lbc_00030000_file_modified_file = netCDF4.Dataset("./lateral_boundary.grid_row-major.nc", "r+")
    else:
        # we don't want to write back, but we still want to be able to modify
        # so we just use a temporary file that we can open in read/write mode
        shutil.copy("./grid.nc", TEMP_FILE_NAME)
        grid_modified_file = netCDF4.Dataset(TEMP_FILE_NAME, "r+")
        shutil.copy("./lateral_boundary.grid.nc", TEMP_FILE_NAME_LAT)
        lateral_grid_modified_file = netCDF4.Dataset(TEMP_FILE_NAME, "r+")

    # the line of the right direction angle for vertex #0:
    p1 = np.array([[0.18511014, 0.79054856]])
    p2 = np.array([[0.18593181, 0.79048109]])
    right_direction_angle = np.squeeze(get_angle(p2 - p1))

    mapping = create_structured_grid_mapping(grid, right_direction_angle, angle_threshold=np.deg2rad(15))
    parent_mapping = create_structured_grid_mapping(parent_grid, right_direction_angle, angle_threshold=np.deg2rad(15))

    v_grf = get_grf_ranges(grid, LocationType.Vertex)
    e_grf = get_grf_ranges(grid, LocationType.Edge)
    c_grf = get_grf_ranges(grid, LocationType.Cell)

    parent_v_grf = get_grf_ranges(parent_grid, LocationType.Vertex)
    parent_e_grf = get_grf_ranges(parent_grid, LocationType.Edge)
    parent_c_grf = get_grf_ranges(parent_grid, LocationType.Cell)

    v_perm = argsort_simple(mapping.vertex_mapping, SimpleRowMajorSorting.vertex_compare, v_grf[0])
    e_perm = argsort_simple(mapping.edge_mapping, SimpleRowMajorSorting.edge_compare, e_grf[0])
    c_perm = argsort_simple(mapping.cell_mapping, SimpleRowMajorSorting.cell_compare, c_grf[0])   

    parent_v_perm = argsort_simple(parent_mapping.vertex_mapping, SimpleRowMajorSorting.vertex_compare, parent_v_grf[0])
    parent_e_perm = argsort_simple(parent_mapping.edge_mapping, SimpleRowMajorSorting.edge_compare, parent_e_grf[0])
    parent_c_perm = argsort_simple(parent_mapping.cell_mapping, SimpleRowMajorSorting.cell_compare, parent_c_grf[0])   

    apply_permutation(grid_modified_file, c_perm, ICON_grid_schema_dbg, LocationType.Cell)
    apply_permutation(grid_modified_file, e_perm, ICON_grid_schema_dbg, LocationType.Edge)
    apply_permutation(grid_modified_file, v_perm, ICON_grid_schema_dbg, LocationType.Vertex)

    apply_permutation(parent_grid_modified_file, parent_c_perm, ICON_grid_schema_parent, LocationType.Cell)
    apply_permutation(parent_grid_modified_file, parent_e_perm, ICON_grid_schema_parent, LocationType.Edge)
    apply_permutation(parent_grid_modified_file, parent_v_perm, ICON_grid_schema_parent, LocationType.Vertex)
    
    perm_dict = {LocationType.Cell : c_perm, LocationType.Edge : e_perm, LocationType.Vertex : v_perm}
    apply_permutation_nbh_tables(grid_modified_file, perm_dict, ICON_grid_schema_dbg)

    perm_dict = {LocationType.Cell : parent_c_perm, LocationType.Edge : parent_e_perm, LocationType.Vertex : parent_v_perm}
    apply_permutation_nbh_tables(parent_grid_modified_file, perm_dict, ICON_grid_schema_parent)

    apply_permutation_latbc_grid(lateral_grid_modified_file, c_perm, e_perm, ICON_grid_schema_lat_grid)    

    apply_permutation(init_con_grid_file_modified_file, c_perm, ICON_grid_schema_ic, LocationType.Cell)
    apply_permutation(init_con_grid_file_modified_file, e_perm, ICON_grid_schema_ic, LocationType.Edge)
    apply_permutation(init_con_grid_file_modified_file, v_perm, ICON_grid_schema_ic, LocationType.Vertex)

    # fig, ax = plt.subplots()

    # take_branch: typing.Optional[LocationType] = LocationType.Cell

    # if take_branch is LocationType.Vertex:
    #     vertices = plot_vertices(ax, grid, field=np.arange(grid.nv))
    #     # transparent edges
    #     vertices.set_edgecolor((0, 0, 0, 0))
    #     fig.colorbar(vertices, ax=ax)

    # elif take_branch is LocationType.Edge:
    #     edges = plot_edges(ax, grid, field=np.arange(grid.ne))
    #     # transparent edges
    #     edges.set_edgecolor((0, 0, 0, 0))
    #     fig.colorbar(edges, ax=ax)

    # if take_branch is LocationType.Cell:
    #     cells = plot_cells(ax, grid, field=np.arange(grid.nc))
    #     # transparent edges
    #     cells.set_edgecolor((0, 0, 0, 0))
    #     fig.colorbar(cells, ax=ax)

    # elif take_branch is None:
    #     pass # this is the plot none branch
    # else:
    #     assert False

    #ax.plot(grid.v_lon_lat[range_to_slice(v_grf[0]), 0], grid.v_lon_lat[range_to_slice(v_grf[0]), 1], 'o-')
    #ax.plot(grid.e_lon_lat[range_to_slice(e_grf[0]), 0], grid.e_lon_lat[range_to_slice(e_grf[0]), 1], 'o-')
    #ax.plot(grid.c_lon_lat[range_to_slice(c_grf[0]), 0], grid.c_lon_lat[range_to_slice(c_grf[0]), 1], 'o-')

    if write_back:
        grid_modified_file.sync()
        parent_grid_modified_file.sync()
        lateral_grid_modified_file.sync()
        init_con_grid_file_modified_file.sync()          
    else:
        os.remove(TEMP_FILE_NAME)

    # ax.autoscale()
    # plt.show()

    # copy files back
    shutil.copy("grid_row-major.nc", "../my_pool/data/ICON/mch/grids/ch_r04b09/grid.nc")   
    shutil.copy("grid.parent_row-major.nc", "../my_pool/data/ICON/mch/grids/ch_r04b09/grid.parent.nc")   
    shutil.copy("lateral_boundary.grid_row-major.nc", "../my_pool/data/ICON/mch/input/ch_r04b09/lateral_boundary.grid.nc")
    shutil.copy("igfff00000000_row-major.nc", "../my_pool/data/ICON/mch/input/ch_r04b09/igfff00000000.nc")

if __name__ == "__main__":
    main()
    # test_permutation()

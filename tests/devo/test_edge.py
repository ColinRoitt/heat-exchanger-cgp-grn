"""Tests for the Edge topology class."""

import pytest
import numpy as np

from OCC.Core.TopoDS import TopoDS_Edge

from src.devo.geometry import Line, Point
from src.devo.topology import Edge, CSType
from src.environment.materials import Material


@pytest.fixture
def material():
    """Fixture for the material of the edge."""
    return Material()


@pytest.fixture
def points():
    """Fixture for the points of the edge."""
    point_0 = Point(x=0.0, y=0.0, z=0.0)
    point_1 = Point(x=50.0, y=0.0, z=0.0)
    return point_0, point_1


@pytest.fixture
def line(points):
    """Fixture for the lines of the edge."""
    return Line(start_point=points[0], end_point=points[1])


@pytest.fixture
def edge(material, line):
    """Fixture for the edge."""
    return Edge(material=material, line=line, cs_area=1.0)

@pytest.fixture
def rectangular_edge(material, line):
    """Fixture for the edge."""
    return Edge(material=material, line=line, dim_1=2.0, dim_2=1.0, cs_type=CSType.rectangle)

def test_id_of_edge(material, line):
    """Test edge id increments when a new edge object is instantiated."""
    Edge.reset_instance_count()
    assert Edge(material=material, line=line, cs_area=1.0).id == 0
    assert Edge(material=material, line=line, cs_area=1.0).id == 1


def test_edge_id_resets(material, line):
    """Test edge class instance count and id can be reset."""
    Edge.reset_instance_count()
    assert Edge(material=material, line=line, cs_area=1.0).id == 0
    assert Edge(material=material, line=line, cs_area=1.0).id == 1
    Edge.reset_instance_count()
    assert Edge(material=material, line=line, cs_area=1.0).id == 0


def test_edge_volume(edge):
    """Test edge object returns its volume."""
    assert edge.volume == 50.0


def test_edge_default_cs_area_can_be_set(material, line):
    """Test the edge class can have its default cs area set for when instantiated during cell division."""
    Edge.default_cs_area = 100.0
    Edge.min_cs_area
    assert Edge.default_cs_area == 100.0
    assert Edge(material=material, line=line).cs_area == 100.0
    assert Edge(material=material, line=line, cs_area=1.0).cs_area == 1.0


def test_edge_minimum_cs_area_can_be_set():
    """Test the edge class can have its minimum cs area set."""
    Edge.min_cs_area = 0.5
    assert Edge.min_cs_area == 0.5


def test_edge_has_a_midpoint_property(edge):
    """Test the edge object has a property to retrieve its midpoint."""
    assert np.array_equal(edge.midpoint, np.array([25.0, 0.0, 0.0]))


def test_edge_as_opencascade_object(edge):
    """Test that the edge object can be represented as an OpenCascade TopoDS_Edge object."""
    edge_occ = edge.as_occ
    assert type(edge_occ == TopoDS_Edge)

def test_edge_cs_type(rectangular_edge):
    """Test the default cs type for edges can be changed"""
    assert rectangular_edge.cs_type == CSType.rectangle

def test_rectangle_edge_min_max_dims(rectangular_edge):
    """Test rectangular edge dims can be changed from minimum"""
    rectangular_edge.cs_type = CSType.rectangle
    assert rectangular_edge.min_dim1 == 0.1
    assert rectangular_edge.min_dim2 == 0.1
    
    rectangular_edge.min_dim1 = 0.05
    rectangular_edge.max_dim1 = 1.0
    rectangular_edge.min_dim2 = 1.0
    rectangular_edge.max_dim2 = 2.0

    assert rectangular_edge.min_dim1 == 0.05
    assert rectangular_edge.max_dim1 == 1.0
    assert rectangular_edge.min_dim2 == 1.0
    assert rectangular_edge.max_dim2 == 2.0

def test_default_rectangle_edge_cs_area(rectangular_edge):
    """Test calculation for default cs area for rectangular edge"""
    assert rectangular_edge.cs_area == 2.0

def test_rectangular_edge_volume(rectangular_edge):
    """Test rectangular edge object returns its volume."""
    assert rectangular_edge.volume == 100.0

def test_rectangular_edge_dims_can_be_updated(rectangular_edge):
    """Test to see if the parameters of the rectabgular edge can be updated with a delta."""
    dim1_delta = 1.0
    dim2_delta = 0.5
    assert rectangular_edge.dim_1 == 2.0
    assert rectangular_edge.dim_2 == 1.0
    rectangular_edge.update_edge_depth(dim1_delta)
    rectangular_edge.update_edge_width(dim2_delta)
    assert rectangular_edge.dim_1 == 3.0
    assert rectangular_edge.dim_2 == 1.5


def test_rectangular_edge_cs_ratio(rectangular_edge):
    """Test to get the ratio of dim1 to dim2 of rectangular edge"""
    assert rectangular_edge.cs_ratio == 2
    dim1_delta = 1.0
    dim2_delta = 3.0
    rectangular_edge.update_edge_depth(dim1_delta)
    rectangular_edge.update_edge_width(dim2_delta)
    assert rectangular_edge.cs_ratio == 0.75

def test_rectangular_edge_second_moment_of_area(rectangular_edge):
    """Test to get the inertia value of rectangular edge"""
    assert round(rectangular_edge.second_moment_of_area, 3) == 0.667
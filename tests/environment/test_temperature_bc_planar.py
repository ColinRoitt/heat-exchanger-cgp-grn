"""Tests for the Planar Temperature Boundary Condition Class."""

import pytest
import numpy as np

from src.environment.stimulus import Stimulus, BoundaryCondition, TemperaturePlanar
from src.environment.materials import Material
from src.devo.geometry import Point, Line, Plane
from src.devo.topology import Edge, Face


@pytest.fixture
def temperature():
    """Fixture for instance of temperature boundary condition"""
    return TemperaturePlanar(Plane(point=Point(5.0, 5.0, 0.0), dir_x=-1.0, dir_y=0.0, dir_z=0.0), 1000)

@pytest.fixture
def line():
    """Line object fixture."""
    return Line(start_point=Point(5.0, 0.0, 0.0), end_point=Point(5.0, 15.0, 0.0))


@pytest.fixture
def material():
    """Fixture for the material of the face."""
    return Material()


@pytest.fixture
def points():
    """Fixture for the points of the edge."""
    Point.reset_instance_count()
    point_0 = Point(x=5.0, y=1.0, z=0.0)
    point_1 = Point(x=5.0, y=8.0, z=2.0)
    point_2 = Point(x=5.0, y=2.0, z=1.0)
    return point_0, point_1, point_2


@pytest.fixture
def lines(points):
    """Fixture for the lines of the face."""
    Line.reset_instance_count()
    line_0 = Line(start_point=points[2], end_point=points[1])
    line_1 = Line(start_point=points[1], end_point=points[0])
    line_2 = Line(start_point=points[0], end_point=points[2])
    return [line_0, line_1, line_2]


@pytest.fixture
def edges(material, lines):
    """Fixture for the edges of face."""
    Edge.reset_instance_count()
    edge_0 = Edge(material=material, line=lines[0], cs_area=1.0)
    edge_1 = Edge(material=material, line=lines[1], cs_area=1.0)
    edge_2 = Edge(material=material, line=lines[2], cs_area=1.0)
    return [edge_0, edge_1, edge_2]


@pytest.fixture
def face(material, edges):
    """Fixture for instance of face"""  
    Face.reset_instance_count()
    face_0 = Face(material=material, edges=edges, thickness=1.0)
    return face_0    


def test_id_increment():
    """test the id of stimulus increments with each initialisation"""
    Stimulus.reset_instance_count()
    stim_0 = TemperaturePlanar(Plane(point=Point(5.0, 5.0, 0.0), dir_x=-1.0, dir_y=0.0, dir_z=0.0), 500)
    stim_1 = TemperaturePlanar(Plane(point=Point(0.0, 5.0, 0.0), dir_x=1.0, dir_y=0.0, dir_z=0.0), 100)
    assert stim_0.id == 0
    assert stim_1.id == 1


def test_temperature_is_boundary_condition(temperature):
    """Test temperature is an instance of a boundary condition"""
    assert isinstance(temperature, BoundaryCondition)


def test_planar_displacement_is_a_stimulus(temperature):
    """Check the temperature is an instance of the Stimulus class."""
    assert isinstance(temperature, Stimulus)


def test_normal(temperature):
    """Test that the plane direction is created"""
    assert np.array_equal(temperature.plane.direction, np.array([-1.0, 0.0, 0.0]))


def test_is_edge_on_boundary_condition(temperature):
    """Test that a edge is on the temperature boundary condition"""
    this_edge = {0:Edge(Material, Line(start_point=Point(5.0, 0.0, 0.0), end_point=Point(5.0, 15.0, 0.0)), 0.0)}
    assert temperature.is_edge_on_plane(this_edge)[0] == this_edge[0]   


def test_is_edge_not_on_boundary_condition(temperature):
    """Test that a edge is not on the temperature boundary condition"""
    this_edge = {1: Edge(Material, Line(start_point=Point(0.0, 0.0, 0.0), end_point=Point(15.0, 0.0, 0.0)), 0.0)}
    assert len(temperature.is_edge_on_plane(this_edge))== 0


def test_is_face_on_boundary_condition(temperature, face):
    """Test that a face is on the temperature boundary condition"""
    assert temperature.is_face_on_plane(face)


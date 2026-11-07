"""Tests for the Planar Temperature Boundary Condition Class."""

import pytest
import numpy as np

from src.environment.stimulus import Stimulus, BoundaryCondition, TemperaturePoint
from src.environment.materials import Material
from src.devo.geometry import Point, Line, Plane
from src.devo.topology import Edge, Face


@pytest.fixture
def temperature():
    """Fixture for instance of temperature boundary condition"""
    return TemperaturePoint(Point(5.0, 5.0, 0.0), magnitude=1000)

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



def test_id_increment():
    """test the id of stimulus increments with each initialisation"""
    Stimulus.reset_instance_count()
    stim_0 = TemperaturePoint(Point(5.0, 5.0, 0.0), magnitude=500)
    stim_1 = TemperaturePoint(Point(0.0, 5.0, 0.0), magnitude=100)
    assert stim_0.id == 0
    assert stim_1.id == 1


def test_temperature_is_boundary_condition(temperature):
    """Test temperature is an instance of a boundary condition"""
    assert isinstance(temperature, BoundaryCondition)


def test_planar_displacement_is_a_stimulus(temperature):
    """Check the temperature is an instance of the Stimulus class."""
    assert isinstance(temperature, Stimulus)


def test_is_nearest_point_to_boundary_condition(temperature):
    """Test that a edge is on the temperature boundary condition"""
    points = {0:Point(5.0, 5.0, 0.0), 1: Point(0.0, 0.0, 0.0), 2: Point(17.0, 25.0, 0.0)}
    nearest_point = temperature.get_nearest_point_on_bc(points)
    assert nearest_point == points[0]

"""Tests for the Planar Fixed Support class."""

import pytest
import numpy as np

from src.environment.stimulus import Stimulus, Support, FixedSupportPlanar
from src.devo.geometry import Point, Line, Plane
from src.devo.topology import Edge
from src.environment.materials import Material

@pytest.fixture
def fixed_support_planar():
    """Fixture for instance of planar fixed support"""
    return FixedSupportPlanar(Plane(point=Point(5.0, 5.0, 0.0), dir_x=0.0, dir_y=0.0, dir_z=-1.0))

def test_id_increment():
    """test the id of constraints increments with each initialisation"""
    Stimulus.reset_instance_count()
    constraint_0 = FixedSupportPlanar(Plane(point=Point(5.0, 5.0, 10.0), dir_x=0.0, dir_y=0.0, dir_z=-1.0))
    constraint_1 = FixedSupportPlanar(Plane(point=Point(5.0, 5.0, 0.0), dir_x=0.0, dir_y=0.0, dir_z=-1.0))
    assert constraint_0.id == 0
    assert constraint_1.id == 1

def test_fixed_support_planar_is_support(fixed_support_planar):
    """Test fixed support planar is an instance of a support"""
    assert isinstance(fixed_support_planar, Support)

def test_planar_displacement_is_a_stimulus(fixed_support_planar):
    """Check the fixed support planar is an instance of the Stimulus class."""
    assert isinstance(fixed_support_planar, Stimulus)

def test_normal(fixed_support_planar):
    """Test that the plane direction is created"""
    assert np.array_equal(fixed_support_planar.plane.direction, np.array([0.0, 0.0, -1.0]))

def test_is_point_on_support(fixed_support_planar):
    """Test that a point is on the support"""
    this_point = Point(x=3.0, y=6.0, z=0.0)
    assert fixed_support_planar.is_point_on_plane(this_point.as_array)   

def test_is_not_point_on_support(fixed_support_planar):
    """Test that a point is not on the support"""
    this_point = Point(x=3.0, y=6.0, z=5.0)
    assert not fixed_support_planar.is_point_on_plane(this_point.as_array)

def test_if_edge_on_support(fixed_support_planar):
    this_edge = Edge(Material, Line(Point(5.0, 0.0, 0.0), Point(5.0, 15.0, 0.0)), 1.0)
    return fixed_support_planar.is_edge_on_plane(this_edge)

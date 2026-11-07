"""Tests for the Planar Displacement class."""

import pytest
import numpy as np
from typing import Type

from src.environment.stimulus import Stimulus, BoundaryCondition, PlanarDisplacement
from src.devo.geometry import Point, Plane


@pytest.fixture
def planar_displacement():
    """Fixture for Planar Displacement object."""
    return PlanarDisplacement(Plane(point=Point(5.0, 5.0, 10.0), dir_x=0.0, dir_y=0.0, dir_z=-1.0), magnitude=1.0)


def test_id_increment():
    """test the id of constraints increments with each initialisation"""
    Stimulus.reset_instance_count()
    constraint_0 = PlanarDisplacement(Plane(point=Point(5.0, 5.0, 10.0), dir_x=0.0, dir_y=0.0, dir_z=-1.0), magnitude=1.0)
    constraint_1 = PlanarDisplacement(Plane(point=Point(5.0, 5.0, 0.0), dir_x=0.0, dir_y=0.0, dir_z=1.0), magnitude=1.0)
    assert constraint_0.id == 0
    assert constraint_1.id == 1


def test_planar_displacement_is_boundary_condition(planar_displacement):
    """Test planar displacement is an instance of a boundary condition"""
    assert isinstance(planar_displacement, BoundaryCondition)


def test_planar_displacement_is_a_stimulus(planar_displacement):
    """Check the planar_displacement is an instance of the Stimulus class."""
    assert isinstance(planar_displacement, Stimulus)


def test_direction(planar_displacement):
    """Test that the plane direction is created"""
    assert np.array_equal(planar_displacement.plane.direction, np.array([0.0, 0.0, -1.0]))


def test_point_on_plane(planar_displacement):
    """Test that a point lies on the plane"""
    this_point = Point(x=3.0, y=6.0, z=10.0)
    assert planar_displacement.is_point_on_plane(this_point.as_array)


def test_not_point_on_plane(planar_displacement):
    """Test that a point does not lie on the plane"""
    this_point = Point(x=3.0, y=6.0, z=5.0)
    assert not planar_displacement.is_point_on_plane(this_point.as_array)


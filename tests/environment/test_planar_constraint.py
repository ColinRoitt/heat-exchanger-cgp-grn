"""Tests for the Planar Constraint class."""

import pytest
import numpy as np

from src.environment.stimulus import Stimulus, Constraint, PlanarConstraint
from src.devo.geometry import Point, Plane

@pytest.fixture
def planar_constraint():
    """Fixture for planar constraint"""
    return PlanarConstraint(Plane(point=Point(5.0, 0.0, 5.0), dir_x=0.0, dir_y=1.0, dir_z=0.0))

def test_id_increment():
    """test the id of constraints increments with each initialisation"""
    Stimulus.reset_instance_count()
    constraint_0 = PlanarConstraint(Plane(point=Point(5.0, 5.0, 10.0), dir_x=0.0, dir_y=0.0, dir_z=0.0))
    constraint_1 = PlanarConstraint(Plane(point=Point(5.0, 5.0, 0.0), dir_x=0.0, dir_y=0.0, dir_z=0.0))
    assert constraint_0.id == 0
    assert constraint_1.id == 1

def test_fixed_support_planar_is_constraint(planar_constraint):
    """Test planar constraint is an instance of a constraint"""
    assert isinstance(planar_constraint, Constraint)

def test_planar_displacement_is_a_stimulus(planar_constraint):
    """Check the planar constraint is an instance of the Stimulus class."""
    assert isinstance(planar_constraint, Stimulus)

def test_normal(planar_constraint):
    """Test that the plane normal is created"""
    assert np.array_equal(planar_constraint.plane.direction, np.array([0.0, 1.0, 0.0]))

def test_if_planar_constraint_constrains_coordinate_delta(planar_constraint):
    """Test if Planar Constraint will set the coordinate delta to be zero-valued."""
    coord_delta = np.array([1.0, 1.0, 1.0])
    updated_delta = planar_constraint.constrain_coord_delta(coord_delta)
    assert np.array_equal(updated_delta, np.array([0.0, 0.0, 0.0]))

def test_is_point_at_constraint(planar_constraint):
    """Test that a point is at the constraint"""
    this_point = Point(x=3.0, y=0.0, z=6.0)
    assert planar_constraint.plane.is_point_on_plane(this_point.as_array)   

def test_is_not_point_at_support(planar_constraint):
    """Test that a point is not at the constraint"""
    this_point = Point(x=3.0, y=6.0, z=5.0)
    assert not planar_constraint.plane.is_point_on_plane(this_point.as_array)
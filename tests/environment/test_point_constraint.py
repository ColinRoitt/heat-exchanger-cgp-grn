"""Tests for the PointConstraint class."""

import pytest
import numpy as np

from src.environment.stimulus import Stimulus, Constraint, PointConstraint


@pytest.fixture
def point_constraint():
    """Fixture for PointConstraint object."""
    return PointConstraint(pos_x=10.0, pos_y=0.0, pos_z=0.0)


def test_id_increments():
    """Test the PointConstraint id increments when a new stimulus is created."""
    Stimulus.reset_instance_count()
    constraint_0 = PointConstraint(pos_x=9.0, pos_y=4.0, pos_z=2.0)
    constraint_1 = PointConstraint(pos_x=-10.0, pos_y=-2.0, pos_z=0.0)
    assert constraint_0.id == 0
    assert constraint_1.id == 1


def test_point_constraint_is_a_constraint(point_constraint):
    """Check the PointConstraint is an instance of the Constraint class."""
    assert isinstance(point_constraint, Constraint)


def test_point_constraint_is_a_stimulus(point_constraint):
    """Check the PointConstraint is an instance of the Stimulus class."""
    assert isinstance(point_constraint, Stimulus)


def test_if_point_constraint_constrains_coordinate_delta(point_constraint):
    """Test if PointConstraint will set the coordinate delta to be zero-valued."""
    coord_delta = np.array([1.0, 1.0, 1.0])
    updated_delta = point_constraint.constrain_coord_delta(coord_delta)
    assert np.array_equal(updated_delta, np.array([0.0, 0.0, 0.0]))

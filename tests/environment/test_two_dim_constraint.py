"""Tests for the TwoDimConstraint class."""

import pytest
import numpy as np

from src.environment.stimulus import Stimulus, Constraint, TwoDimConstraint


@pytest.fixture
def two_dim_constraint():
    """Fixture for 2D constraint."""
    return TwoDimConstraint()


def test_id_increments():
    """Test the 2D constraint id increments when a new stimulus is created."""
    Stimulus.reset_instance_count()
    constraint_0 = TwoDimConstraint()
    constraint_1 = TwoDimConstraint()
    assert constraint_0.id == 0
    assert constraint_1.id == 1


def test_2D_constraint_is_a_constraint(two_dim_constraint):
    """Check the 2D constraint is an instance of the Constraint class."""
    assert isinstance(two_dim_constraint, Constraint)


def test_2D_constraint_is_a_stimulus(two_dim_constraint):
    """Check the 2D constraint is an instance of the Stimulus class."""
    assert isinstance(two_dim_constraint, Stimulus)


def test_2D_constraint_inp_deck():
    """Test the 2D constraint returns the correct inp deck."""
    Stimulus.reset_instance_count()
    constraint_0 = TwoDimConstraint()
    assert constraint_0.get_inp_deck() == '*BOUNDARY\nALLNODES, 3\n'


def test_if_2D_constraint_constrains_coordinate_delta(two_dim_constraint):
    """Test if 2D constraint will set the coordinate delta to be zero-valued."""
    coord_delta = np.array([1.0, 1.0, 1.0])
    updated_delta = two_dim_constraint.constrain_coord_delta(coord_delta)
    assert np.array_equal(updated_delta, np.array([1.0, 1.0, 0.0]))

"""Tests for the RollerSupport class."""

import pytest
import numpy as np

from src.environment.stimulus import Stimulus, BoundaryCondition, Support, RollerSupport
from src.devo.geometry import Point


@pytest.fixture
def roller_support():
    """Fixture for roller support."""
    return RollerSupport(pos_x=0.0, pos_y=0.0, pos_z=0.0)


def test_id_increments():
    """Test the roller support id increments when a new stimulus is created."""
    Stimulus.reset_instance_count()
    support_0 = RollerSupport(pos_x=0.0, pos_y=0.0, pos_z=0.0)
    support_1 = RollerSupport(pos_x=0.0, pos_y=1.0, pos_z=1.0)
    assert support_0.id == 0
    assert support_1.id == 1


def test_num_reactions_of_roller_support(roller_support):
    """Test the roller support object can return the correct number of reactions."""
    assert roller_support.num_reactions == 1


def test_roller_support_is_a_support(roller_support):
    """Check the roller support is an instance of the Support class."""
    assert isinstance(roller_support, Support)


def test_roller_support_is_a_boundary_condition(roller_support):
    """Check the roller support is an instance of the BoundaryCondition class."""
    assert isinstance(roller_support, BoundaryCondition)


def test_roller_support_is_a_stimulus(roller_support):
    """Check the roller support is an instance of the Stimulus class."""
    assert isinstance(roller_support, Stimulus)


def test_roller_support_inp_deck():
    """Test the roller support returns the correct inp deck."""
    Stimulus.reset_instance_count()
    support_0 = RollerSupport(pos_x=0.0, pos_y=0.0, pos_z=0.0)
    assert support_0.get_inp_deck() == '*BOUNDARY\nroller_support_0, 2\n'


def test_roller_support_location(roller_support):
    """Test the roller support returns the correct location based on its position coordinates."""
    assert roller_support.location == [0.0, 0.0, 0.0]


def test_if_point_is_at_roller_support(roller_support):
    """Test if a point is at same location as the roller support."""
    assert roller_support.is_point_at_support(Point(x=0.0, y=0.0, z=0.0)) == True
    assert roller_support.is_point_at_support(Point(x=1.0, y=0.0, z=0.0)) == False


def test_if_roller_support_constrains_coordinate_delta(roller_support):
    """Test if roller support will set the coordinate delta to be zero-valued."""
    coord_delta = np.array([1.0, 1.0, 1.0])
    updated_delta = roller_support.constrain_coord_delta(coord_delta)
    assert np.array_equal(updated_delta, np.array([0.0, 0.0, 0.0]))
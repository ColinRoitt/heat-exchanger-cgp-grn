"""Tests for the SlidingSupport class."""

import pytest
import numpy as np

from src.environment.stimulus import Stimulus, BoundaryCondition, Support, SlidingSupport
from src.devo.geometry import Point


@pytest.fixture
def sliding_support():
    """Fixture for sliding support."""
    return SlidingSupport(pos_x=0.0, pos_y=0.0, pos_z=0.0)


def test_id_increments():
    """Test the sliding support id increments when a new stimulus is created."""
    Stimulus.reset_instance_count()
    support_0 = SlidingSupport(pos_x=0.0, pos_y=0.0, pos_z=0.0)
    support_1 = SlidingSupport(pos_x=0.0, pos_y=1.0, pos_z=1.0)
    assert support_0.id == 0
    assert support_1.id == 1


def test_num_reactions_of_sliding_support(sliding_support):
    """Test the sliding support object can return the correct number of reactions."""
    assert sliding_support.num_reactions == 2


def test_sliding_support_is_a_support(sliding_support):
    """Check the sliding support is an instance of the Support class."""
    assert isinstance(sliding_support, Support)


def test_roller_support_is_a_boundary_condition(sliding_support):
    """Check the sliding support is an instance of the BoundaryCondition class."""
    assert isinstance(sliding_support, BoundaryCondition)


def test_sliding_support_is_a_stimulus(sliding_support):
    """Check the sliding support is an instance of the Stimulus class."""
    assert isinstance(sliding_support, Stimulus)


def test_sliding_support_inp_deck():
    """Test the sliding support returns the correct inp deck."""
    Stimulus.reset_instance_count()
    support_0 = SlidingSupport(pos_x=0.0, pos_y=0.0, pos_z=0.0)
    assert support_0.get_inp_deck() == '*BOUNDARY\nsliding_support_0, 1\nsliding_support_0, 3\n'


def test_sliding_support_location(sliding_support):
    """Test the sliding support returns the correct location based on its position coordinates."""
    assert sliding_support.location == [0.0, 0.0, 0.0]


def test_if_point_is_at_sliding_support(sliding_support):
    """Test if a point is at same location as the sliding support."""
    assert sliding_support.is_point_at_support(Point(x=0.0, y=0.0, z=0.0)) == True
    assert sliding_support.is_point_at_support(Point(x=1.0, y=0.0, z=0.0)) == False


def test_if_sliding_support_constrains_coordinate_delta(sliding_support):
    """Test if sliding support will set the coordinate delta to be zero-valued."""
    coord_delta = np.array([1.0, 1.0, 1.0])
    updated_delta = sliding_support.constrain_coord_delta(coord_delta)
    assert np.array_equal(updated_delta, np.array([0.0, 0.0, 0.0]))
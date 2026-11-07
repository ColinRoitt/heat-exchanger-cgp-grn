"""Tests for the FixedSupport class."""

import pytest
import numpy as np

from src.environment.stimulus import Stimulus, BoundaryCondition, Support, FixedSupport
from src.devo.geometry import Point


@pytest.fixture
def fixed_support():
    """Fixture for fixed support."""
    return FixedSupport(pos_x=0.0, pos_y=0.0, pos_z=0.0)


def test_id_increments():
    """Test the fixed support id increments when a new stimulus is created."""
    Stimulus.reset_instance_count()
    support_0 = FixedSupport(pos_x=0.0, pos_y=0.0, pos_z=0.0)
    support_1 = FixedSupport(pos_x=0.0, pos_y=1.0, pos_z=1.0)
    assert support_0.id == 0
    assert support_1.id == 1


def test_num_reactions_of_fixed_support(fixed_support):
    """Test the fixed support object can return the correct number of reactions."""
    assert fixed_support.num_reactions == 3


def test_fixed_support_is_a_support(fixed_support):
    """Check the fixed support is an instance of the Support class."""
    assert isinstance(fixed_support, Support)


def test_fixed_support_is_a_boundary_condition(fixed_support):
    """Check the fixed support is an instance of the BoundaryCondition class."""
    assert isinstance(fixed_support, BoundaryCondition)


def test_fixed_support_is_a_stimulus(fixed_support):
    """Check the fixed support is an instance of the Stimulus class."""
    assert isinstance(fixed_support, Stimulus)


def test_fixed_support_inp_deck():
    """Test the fixed support returns the correct inp deck."""
    Stimulus.reset_instance_count()
    support_0 = FixedSupport(pos_x=0.0, pos_y=0.0, pos_z=0.0)
    assert support_0.get_inp_deck() == '*BOUNDARY\nfixed_support_0, 1, 3\n'


def test_fixed_support_location(fixed_support):
    """Test the fixed support returns the correct location based on its position coordinates."""
    assert fixed_support.location == [0.0, 0.0, 0.0]


def test_if_point_is_at_fixed_support(fixed_support):
    """Test if a point is at same location as the fixed support."""
    assert fixed_support.is_point_at_support(Point(x=0.0, y=0.0, z=0.0)) == True
    assert fixed_support.is_point_at_support(Point(x=1.0, y=0.0, z=0.0)) == False


def test_if_fixed_support_constrains_coordinate_delta(fixed_support):
    """Test if fixed support will set the coordinate delta to be zero-valued."""
    coord_delta = np.array([1.0, 1.0, 1.0])
    updated_delta = fixed_support.constrain_coord_delta(coord_delta)
    assert np.array_equal(updated_delta, np.array([0.0, 0.0, 0.0]))

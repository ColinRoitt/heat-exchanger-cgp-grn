"""Tests for the PinnedSupport class."""

import pytest
import numpy as np

from src.environment.stimulus import Stimulus, BoundaryCondition, Support, PinnedSupport
from src.devo.geometry import Point


@pytest.fixture
def pinned_support():
    """Fixture for pinned support."""
    return PinnedSupport(pos_x=0.0, pos_y=0.0, pos_z=0.0)


def test_id_increments():
    """Test the pinned support id increments when a new stimulus is created."""
    Stimulus.reset_instance_count()
    support_0 = PinnedSupport(pos_x=0.0, pos_y=0.0, pos_z=0.0)
    support_1 = PinnedSupport(pos_x=0.0, pos_y=1.0, pos_z=1.0)
    assert support_0.id == 0
    assert support_1.id == 1


def test_num_reactions_of_pinned_support(pinned_support):
    """Test the pinned support object can return the correct number of reactions."""
    assert pinned_support.num_reactions == 2


def test_pinned_support_is_a_support(pinned_support):
    """Check the pinned support is an instance of the Support class."""
    assert isinstance(pinned_support, Support)


def test_pinned_support_is_a_boundary_condition(pinned_support):
    """Check the pinned support is an instance of the BoundaryCondition class."""
    assert isinstance(pinned_support, BoundaryCondition)


def test_pinned_support_is_a_stimulus(pinned_support):
    """Check the pinned support is an instance of the Stimulus class."""
    assert isinstance(pinned_support, Stimulus)


def test_pinned_support_inp_deck():
    """Test the pinned support returns the correct inp deck."""
    Stimulus.reset_instance_count()
    support_0 = PinnedSupport(pos_x=0.0, pos_y=0.0, pos_z=0.0)
    print(support_0.get_inp_deck())
    assert support_0.get_inp_deck() == '*BOUNDARY\npinned_support_0, 1, 2\n'


def test_pinned_support_location(pinned_support):
    """Test the pinned support returns the correct location based on its position coordinates."""
    assert pinned_support.location == [0.0, 0.0, 0.0]


def test_if_point_is_at_pinned_support(pinned_support):
    """Test if a point is at same location as the pinned support."""
    assert pinned_support.is_point_at_support(Point(x=0.0, y=0.0, z=0.0)) == True
    assert pinned_support.is_point_at_support(Point(x=1.0, y=0.0, z=0.0)) == False


def test_if_pinned_support_constrains_coordinate_delta(pinned_support):
    """Test if pinned support will set the coordinate delta to be zero-valued."""
    coord_delta = np.array([1.0, 1.0, 1.0])
    updated_delta = pinned_support.constrain_coord_delta(coord_delta)
    assert np.array_equal(updated_delta, np.array([0.0, 0.0, 0.0]))

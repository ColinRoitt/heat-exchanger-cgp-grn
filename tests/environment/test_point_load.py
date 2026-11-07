"""Tests for the PointLoad class."""

import pytest
import numpy as np

from src.environment.stimulus import Stimulus, Load, PointLoad
from src.devo.geometry import Point


@pytest.fixture
def point_load():
    """Fixture for point load."""
    return PointLoad(magnitude=10000, pos_x=0.0, pos_y=0.0, pos_z=0.0, dir_x=1.0, dir_y=0.0, dir_z=0.0)


def test_id_increments():
    """Test the point load id increments when a new stimulus is created."""
    Stimulus.reset_instance_count()
    point_load_0 = PointLoad(magnitude=10000, pos_x=0.0, pos_y=0.0, pos_z=0.0, dir_x=1.0, dir_y=0.0, dir_z=0.0)
    point_load_1 = PointLoad(magnitude=10000, pos_x=0.0, pos_y=0.0, pos_z=0.0, dir_x=1.0, dir_y=0.0, dir_z=0.0)
    assert point_load_0.id == 0
    assert point_load_1.id == 1


def test_point_load_is_a_load(point_load):
    """Check the point load is an instance of the Load class."""
    assert isinstance(point_load, Load)


def test_point_load_is_a_stimulus(point_load):
    """Check the point load is an instance of the Stimulus class."""
    assert isinstance(point_load, Stimulus)


def test_point_load_inp_deck():
    """Test the point load returns the correct inp deck."""
    Stimulus.reset_instance_count()
    point_load_0 = PointLoad(magnitude=10000, pos_x=0.0, pos_y=0.0, pos_z=0.0, dir_x=1.0, dir_y=0.0, dir_z=0.0)
    point_load_1 = PointLoad(magnitude=10000, pos_x=0.0, pos_y=0.0, pos_z=0.0, dir_x=0.0, dir_y=1.0, dir_z=0.0)
    point_load_2 = PointLoad(magnitude=10000, pos_x=0.0, pos_y=0.0, pos_z=0.0, dir_x=0.0, dir_y=0.0, dir_z=1.0)
    point_load_3 = PointLoad(magnitude=10000, pos_x=0.0, pos_y=0.0, pos_z=0.0, dir_x=-1.0, dir_y=0.0, dir_z=0.0)
    point_load_4 = PointLoad(magnitude=10000, pos_x=0.0, pos_y=0.0, pos_z=0.0, dir_x=0.0, dir_y=-1.0, dir_z=0.0)
    point_load_5 = PointLoad(magnitude=10000, pos_x=0.0, pos_y=0.0, pos_z=0.0, dir_x=0.0, dir_y=0.0, dir_z=-1.0)
    print(point_load_0.get_inp_deck())
    assert point_load_0.get_inp_deck() == '*CLOAD\npoint_load_0, 1, 10000.0\n'
    assert point_load_1.get_inp_deck() == '*CLOAD\npoint_load_1, 2, 10000.0\n'
    assert point_load_2.get_inp_deck() == '*CLOAD\npoint_load_2, 3, 10000.0\n'
    assert point_load_3.get_inp_deck() == '*CLOAD\npoint_load_3, 1, -10000.0\n'
    assert point_load_4.get_inp_deck() == '*CLOAD\npoint_load_4, 2, -10000.0\n'
    assert point_load_5.get_inp_deck() == '*CLOAD\npoint_load_5, 3, -10000.0\n'


def test_point_load_location(point_load):
    """Test the point load returns the correct location based on its position coordinates."""
    assert point_load.location == [0.0, 0.0, 0.0]


def test_if_point_is_at_point_load(point_load):
    """Test if a point is at same location as the point load."""
    assert point_load.is_point_at_load(Point(x=0.0, y=0.0, z=0.0)) == True
    assert point_load.is_point_at_load(Point(x=1.0, y=0.0, z=0.0)) == False


def test_if_point_load_constrains_coordinate_delta(point_load):
    """Test if point load will set the coordinate delta to be zero-valued."""
    coord_delta = np.array([1.0, 1.0, 1.0])
    updated_delta = point_load.constrain_coord_delta(coord_delta)
    assert np.array_equal(updated_delta, np.array([0.0, 0.0, 0.0]))

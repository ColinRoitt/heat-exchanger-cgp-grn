"""Tests for the Point geometric class."""

import pytest
import numpy as np

from OCC.Core.gp import gp_Pnt

from src.devo.geometry import Point


@pytest.fixture
def point():
    """Point object fixture."""
    Point.coord_delta_ratio = 1.0
    return Point(x=10.0, y=-50.0, z=3.0)


def test_point_id_increments():
    """Test the point id increments when a new point object is instantiated."""
    Point.reset_instance_count()
    assert Point(x=10.0, y=-50.0, z=3.0).id == 0
    assert Point(x=10.0, y=-50.0, z=3.0).id == 1


def test_point_id_resets():
    """Test point class instance count and id can be reset."""
    Point.reset_instance_count()
    assert Point(x=10.0, y=-50.0, z=3.0).id == 0
    assert Point(x=10.0, y=-50.0, z=3.0).id == 1
    Point.reset_instance_count()
    assert Point(x=10.0, y=-50.0, z=3.0).id == 0


def test_point_as_array(point):
    """Test the Point object returns the correct coordinates as array."""
    assert np.array_equal(point.get_point_array(), np.array([10.0, -50.0, 3.0]))


def test_point_coordinate_update(point):
    """Test the Point object can be updated with the coordinate delta."""
    delta = np.array([1.0, -1.0, -0.5])
    point.update_point_coord(delta)
    assert np.array_equal(point.get_point_array(), np.array([11.0, -51.0, 2.5]))


def test_point_as_opencascade_object(point):
    """Test that the point object can be represented as an OpenCascade gp_Pnt object."""
    point_occ = point.as_occ
    assert type(point_occ == gp_Pnt)
    assert point_occ.Coord() == (10.0, -50.0, 3.0)

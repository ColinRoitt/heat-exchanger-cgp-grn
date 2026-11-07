"""Tests for the Plane geometric class."""

import pytest
import numpy as np

from OCC.Core.Geom import Geom_Plane

from src.devo.geometry import Point, Plane


@pytest.fixture
def point_1():
    """Point object fixture."""
    return Point(x=5.0, y=5.0, z=10.0)


@pytest.fixture
def plane(point_1):
    """Plane object fixture"""
    return Plane(point=point_1, dir_x=0.0, dir_y=0.0, dir_z=-1.0)


def test_plane_id_increments():
    """Test the plane id increments when a new point object is instantiated."""
    Plane.reset_instance_count()
    Point.reset_instance_count()
    point_4 = Point(x=5.0, y=5.0, z=10.0)
    point_5 = Point(x=2.0, y=1.0, z=10.0)

    assert Plane(point=point_4, dir_x=0.0, dir_y=0.0, dir_z=-1.0).id == 0
    assert Plane(point=point_5, dir_x=0.0, dir_y=0.0, dir_z=-1.0).id == 1


def test_plane_id_resets(point_1):
    """Test plane class instance count and id can be reset."""
    Plane.reset_instance_count()
    assert Plane(point=point_1, dir_x=0.0, dir_y=0.0, dir_z=-1.0).id == 0
    assert Plane(point=point_1,  dir_x=0.0, dir_y=0.0, dir_z=-1.0).id == 1
    Plane.reset_instance_count()
    assert Plane(point=point_1,  dir_x=0.0, dir_y=0.0, dir_z=-1.0).id == 0


def test_direction(plane):
    """Test that the plane direction is created"""
    assert np.array_equal(plane.direction, np.array([0.0, 0.0, -1.0]))


def test_point_on_plane(plane):
    """Test that a point lies on the plane"""
    this_point = Point(x=3.0, y=6.0, z=10.0)
    assert plane.is_point_on_plane(this_point.as_array)


def test_not_point_on_plane(plane):
    """Test that a point does not lie on the plane"""
    this_point = Point(x=3.0, y=6.0, z=5.0)
    assert not plane.is_point_on_plane(this_point.as_array)


def test_plane_as_opencascade_object(plane):
    """Test that the plane object can be represented as an OpenCascade Geom_Plane object."""
    plane_occ = plane.as_occ
    assert type(plane_occ == Geom_Plane)

    

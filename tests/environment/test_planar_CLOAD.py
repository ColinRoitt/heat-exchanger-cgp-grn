"""Tests for the Planar Distributed Load class."""

import pytest
import numpy as np
from typing import Type

from src.environment.stimulus import Stimulus, Load, PlanarConcentratedLoad
from src.devo.geometry import Point, Plane


@pytest.fixture
def planar_conc_load():
    """Fixture for Planar Concentrated object."""
    return PlanarConcentratedLoad(Plane(point=Point(5.0, 5.0, 10.0), dir_x=0.0, dir_y=0.0, dir_z=-1.0), magnitude=1.0, application='face')

def test_id_increment():
    """test the id of stimulus increments with each initialisation"""
    Stimulus.reset_instance_count()
    dload_0 = PlanarConcentratedLoad(Plane(point=Point(5.0, 5.0, 10.0), dir_x=0.0, dir_y=0.0, dir_z=-1.0), magnitude=1.0, application='edge')
    dload_1 = PlanarConcentratedLoad(Plane(point=Point(5.0, 5.0, 0.0), dir_x=0.0, dir_y=0.0, dir_z=1.0), magnitude=1.0, application='edge')
    assert dload_0.id == 0
    assert dload_1.id == 1


def test_Concentrated_load_is_a_load(planar_conc_load):
    """Check the Concentrated load is an instance of the Load class."""
    assert isinstance(planar_conc_load, Load)


def test_distributed_load_is_a_stimulus(planar_conc_load):
    """Check the distributed load is an instance of the Stimulus class."""
    assert isinstance(planar_conc_load, Stimulus)


def test_direction(planar_conc_load):
    """Test that the plane direction is created"""
    assert np.array_equal(planar_conc_load.plane.direction, np.array([0.0, 0.0, -1.0]))    


def test_point_load_inp_deck():
    """Test the distributed load returns the correct inp deck."""
    Stimulus.reset_instance_count()
    cload_0 = PlanarConcentratedLoad(Plane(point=Point(5.0, 5.0, 10.0), dir_x=0.0, dir_y=0.0, dir_z=-1.0), magnitude=100.0, application='edge')
    cload_1 = PlanarConcentratedLoad(Plane(point=Point(5.0, 5.0, 0.0), dir_x=0.0, dir_y=1.0, dir_z=0.0), magnitude=120.0, application='edge')

    assert cload_0.get_inp_deck() == '*CLOAD\nplanar_edge_CLOAD_0, 3, -100.0\n'
    assert cload_1.get_inp_deck() == '*CLOAD\nplanar_edge_CLOAD_1, 2, 120.0\n'


def test_point_on_plane(planar_conc_load):
    """Test that a point lies on the plane"""
    this_point = Point(x=3.0, y=6.0, z=10.0)
    assert planar_conc_load.is_point_on_plane(this_point.as_array)


def test_not_point_on_plane(planar_conc_load):
    """Test that a point does not lie on the plane"""
    this_point = Point(x=3.0, y=6.0, z=5.0)
    assert not planar_conc_load.is_point_on_plane(this_point.as_array)


def test_if_point_load_constrains_coordinate_delta(planar_conc_load):
    """Test if point load will set the coordinate delta to be zero-valued."""
    coord_delta = np.array([1.0, 1.0, 1.0])
    updated_delta = planar_conc_load.constrain_coord_delta(coord_delta)
    assert np.array_equal(updated_delta, np.array([0.0, 0.0, 0.0]))

"""Tests for the Planar Distributed Load class."""

import pytest
import numpy as np

from src.environment.stimulus import Stimulus, Load, PlanarDistributedLoad
from src.devo.geometry import Point, Line, Plane
from src.devo.topology import Edge, Face
from src.environment.materials import Material


@pytest.fixture
def planar_dist_load():
    """Fixture for Planar Distributed object."""
    Stimulus.reset_instance_count()
    return PlanarDistributedLoad(Plane(point=Point(5.0, 5.0, 0.0), dir_x=0.0, dir_y=0.0, dir_z=-1.0), magnitude=1.0, application='face')


@pytest.fixture
def material():
    """Fixture for the material of the face."""
    return Material()


@pytest.fixture
def points():
    """Fixture for the points of the edge."""
    Point.reset_instance_count()
    point_0 = Point(x=0.0, y=0.0, z=0.0)
    point_1 = Point(x=0.0, y=10.0, z=0.0)
    point_2 = Point(x=10.0, y=10.0, z=0.0)
    return point_0, point_1, point_2


@pytest.fixture
def lines(points):
    """Fixture for the lines of the face."""
    Line.reset_instance_count()
    line_0 = Line(start_point=points[2], end_point=points[1])
    line_1 = Line(start_point=points[1], end_point=points[0])
    line_2 = Line(start_point=points[0], end_point=points[2])
    return [line_0, line_1, line_2]


@pytest.fixture
def edges(material, lines):
    """Fixture for the edges of face."""
    Edge.reset_instance_count()
    edge_0 = Edge(material=material, line=lines[0], cs_area=1.0)
    edge_1 = Edge(material=material, line=lines[1], cs_area=1.0)
    edge_2 = Edge(material=material, line=lines[2], cs_area=1.0)
    return [edge_0, edge_1, edge_2]


@pytest.fixture
def face(material, edges):
    """Fixture for instance of face"""  
    Face.reset_instance_count()
    face_0 = Face(material=material, edges=edges, thickness=1.0)
    return face_0    


def test_id_increment():
    """test the id of stimulus increments with each initialisation"""
    Stimulus.reset_instance_count()
    dload_0 = PlanarDistributedLoad(Plane(point=Point(5.0, 5.0, 10.0), dir_x=0.0, dir_y=0.0, dir_z=-1.0), magnitude=1.0, application='face')
    dload_1 = PlanarDistributedLoad(Plane(point=Point(5.0, 5.0, 0.0), dir_x=0.0, dir_y=0.0, dir_z=-1.0), magnitude=1.0, application='face')
    assert dload_0.id == 0
    assert dload_1.id == 1


def test_distributed_load_is_a_load(planar_dist_load):
    """Check the distributed load is an instance of the Load class."""
    assert isinstance(planar_dist_load, Load)


def test_distributed_load_is_a_stimulus(planar_dist_load):
    """Check the distributed load is an instance of the Stimulus class."""
    assert isinstance(planar_dist_load, Stimulus)


def test_direction(planar_dist_load):
    """Test that the plane normal is created"""
    assert np.array_equal(planar_dist_load.plane.direction, np.array([0.0, 0.0, -1.0]))    


def test_dist_load_inp_deck():
    """Test the distributed load returns the correct inp deck."""
    Stimulus.reset_instance_count()
    dload_0 = PlanarDistributedLoad((Plane(point=Point(5.0, 5.0, 10.0), dir_x=0.0, dir_y=0.0, dir_z=-1.0)), magnitude=120.0, application='face')
    dload_1 = PlanarDistributedLoad((Plane(point=Point(5.0, 5.0, 0.0), dir_x=0.0, dir_y=0.0, dir_z=-1.0)), magnitude=-50.0, application='face')

    assert dload_0.get_inp_deck() == '*DLOAD\nplanar_face_DLOAD_0, P, 120.0\n'
    assert dload_1.get_inp_deck() == '*DLOAD\nplanar_face_DLOAD_1, P, -50.0\n'

    Stimulus.reset_instance_count()

def test_point_on_plane(planar_dist_load):
    """Test that a point lies on the plane"""
    this_point = Point(x=20.0, y=20.0, z=0.0)
    assert planar_dist_load.is_point_on_plane(this_point.as_array)


def test_not_point_on_plane(planar_dist_load):
    """Test that a point does not lie on the plane"""
    this_point = Point(x=3.0, y=6.0, z=5.0)
    assert not planar_dist_load.is_point_on_plane(this_point.as_array)


def test_is_normal_coincident(planar_dist_load, face):
    """Test that the normal of a plane and a normal vector (of a face for e.g.) are coincident"""
    assert planar_dist_load.is_normal_coincident(face)


def test_if_load_constrains_coordinate_delta(planar_dist_load):
    """Test if point load will set the coordinate delta to be zero-valued."""
    coord_delta = np.array([1.0, 1.0, 1.0])
    updated_delta = planar_dist_load.constrain_coord_delta(coord_delta)
    assert np.array_equal(updated_delta, np.array([0.0, 0.0, 0.0]))

def test_if_load_acts_on_edge(planar_dist_load):
    """Test if the load is to act on an edge entity"""
    assert planar_dist_load.application_type == "planar_face_DLOAD"
    Stimulus.reset_instance_count()
    edge_load = PlanarDistributedLoad(Plane(point=Point(0.0, 5.0, 0.0), dir_x=1.0, dir_y=0.0, dir_z=0.0), magnitude=1.0, application='edge')
    assert edge_load.application_type == "planar_edge_DLOAD"

def test_if_edge_is_on_plane(edges):
    """Test if edge is located on plane"""
    Stimulus.reset_instance_count()
    edge_load = PlanarDistributedLoad(Plane(point=Point(0.0, 5.0, 0.0), dir_x=1.0, dir_y=0.0, dir_z=0.0), magnitude=1.0, application='edge')
    assert edge_load.is_edge_on_plane(edges[1])

def test_edge_load_inp_deck():
    """Test inp deck is correct for edge load"""
    Stimulus.reset_instance_count()
    edge_load = PlanarDistributedLoad(Plane(point=Point(0.0, 5.0, 0.0), dir_x=1.0, dir_y=0.0, dir_z=0.0), magnitude=1.0, application='edge')
    assert edge_load.get_inp_deck(ednor=1) == '*DLOAD\nplanar_edge_DLOAD_0, EDNOR1, 1.0\n'
"""Tests for the Face topology class."""

import pytest
import numpy as np

from src.devo.geometry import Line, Point
from src.devo.topology import Edge, Face
from src.environment.materials import Material


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

def test_id_of_face(face, material, edges):
    """Test face id increments when a new facee object is instantiated."""
    assert face.id == 0
    assert Face(material=material, edges=edges, thickness=1.0).id == 1

def test_face_id_resets(face):
    """Test face class instance count and id can be reset."""
    this_face = Face(material=material, edges=edges, thickness=1.0)
    assert this_face.id == 1
    Face.reset_instance_count()
    assert face.id == 0    

def test_get_curve_loop(face):
    assert np.array_equal(face.curve_loop, [face.edges[0].id, face.edges[1].id, face.edges[2].id])

def test_face_normal(face): 
    assert np.allclose(face.normal, np.array([0.0, 0.0, 1.0]))

def test_face_surface_area(face):
    assert face.surface_area == 50

def test_face_thickness_can_be_updated(face):
    """Basic test to see if the cs area of the edge can be updated with an area delta."""
    thickness_delta = 1.0
    assert face.thickness == 1.0
    face.update_thickness(thickness_delta)
    assert face.thickness == 2.0

def test_face_cs_area_can_not_be_set_below_min():
    """Test the face class can have its minimum thickness set."""
    Face.min_thickness = 0.4
    new_face = Face(material=material, edges=edges, thickness=0.5)
    assert new_face.thickness == 0.5
    new_face.update_thickness(-2.0)
    assert new_face.thickness == 0.4
    
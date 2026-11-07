"""Tests for the Face topology class."""

import pytest
import numpy as np

from OCC.Core.TopoDS import TopoDS_Face, TopoDS_Wire, TopoDS_Shape
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeWire,\
                                    BRepBuilderAPI_MakeShape, BRepBuilderAPI_Sewing

from src.devo.geometry import Line, Point, Plane
from src.devo.topology import Edge, Face
from src.environment.materials import Material


from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.BRep import BRep_Tool, BRep_Tool_Surface
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.GeomLProp import GeomLProp_SLProps
from OCC.Core.TopAbs import TopAbs_REVERSED
from OCC.Extend.TopologyUtils import TopologyExplorer
from OCC.Core.BRep import BRep_Builder, BRep_Tool
from OCC.Core.TopoDS import TopoDS_Compound, topods_Face


@pytest.fixture
def material():
    """Fixture for the material of the face."""
    return Material()


@pytest.fixture
def points():
    """Fixture for the points of the face."""
    point_0 = Point(x=0.0, y=0.0, z=0.0)
    point_1 = Point(x=50.0, y=0.0, z=0.0)
    point_2 = Point(x=50.0, y=50.0, z=0.0)
    point_3 = Point(x=0.0, y=50.0, z=0.0)
    return point_0, point_1, point_2, point_3


@pytest.fixture
def lines(points):
    """Fixture for the lines of the face."""
    line_0 = Line(start_point=points[0], end_point=points[1])
    line_1 = Line(start_point=points[1], end_point=points[2])
    line_2 = Line(start_point=points[2], end_point=points[3])
    line_3 = Line(start_point=points[3], end_point=points[0])
    return line_0, line_1, line_2, line_3


@pytest.fixture
def edges(material, lines):
    """Fixture for the edges of face."""
    edge_0 = Edge(material=material, line=lines[0], cs_area=1.0)
    edge_1 = Edge(material=material, line=lines[1], cs_area=1.0)
    edge_2 = Edge(material=material, line=lines[2], cs_area=1.0)
    edge_3 = Edge(material=material, line=lines[3], cs_area=1.0)
    return [edge_0, edge_1, edge_2, edge_3]


@pytest.fixture
def surface(points):
    return Plane(point_0=points[0], point_1=points[1], point_2=points[2])


@pytest.fixture
def face(edges, surface, material):
    return Face(material=material, edges=edges, surface=surface)


def test_id_of_face(material, edges):
    """Test face id increments when a new edge object is instantiated."""
    Face.reset_instance_count()
    assert Face(material=material, edges=edges).id == 0
    assert Face(material=material, edges=edges).id == 1


def test_edge_id_resets(material, edges):
    """Test Face class instance count and id can be reset."""
    Face.reset_instance_count()
    assert Face(material=material, edges=edges).id == 0
    assert Face(material=material, edges=edges).id == 1
    Face.reset_instance_count()
    assert Face(material=material, edges=edges).id == 0


def test_face_can_be_assigned_a_surface(material, edges, surface):
    """Test if the Face object can be assigned a surface."""
    test_face = Face(material=material, edges=edges, surface=surface)
    assert test_face.surface == surface


def test_generating_a_wire_opencascade_object(edges):
    """Test that a OCC wire object for bounding the OCC face object can be generated from the edges."""
    wire = BRepBuilderAPI_MakeWire(edges[0].as_occ).Wire()
    assert type(wire) == TopoDS_Wire
    wire = BRepBuilderAPI_MakeWire(wire, edges[1].as_occ).Wire()
    assert type(wire) == TopoDS_Wire


def test_face_as_opencascade_object(face):
    """Test that the Face object can be represented as an OpenCascade TopoDS_Face object."""
    face_occ = face.as_occ
    assert type(face_occ) == TopoDS_Face

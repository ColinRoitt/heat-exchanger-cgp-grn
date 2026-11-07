"""Topology Classes"""

import numpy as np

from typing import Type
from enum import Enum

from OCC.Core.gp import gp_Pnt, gp_Dir
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeWire
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.BRep import BRep_Builder, BRep_Tool, BRep_Tool_Surface
from OCC.Core.TopoDS import TopoDS_Compound, topods_Face
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopAbs import TopAbs_REVERSED
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface

from src.environment.materials import Material
from src.devo.geometry import Point, Line


class CSType(Enum):
    """Enum class to hold the different cross-sectional type options."""
    circle = 0
    rectangle = 1


class Edge:
    """Topology entity that bounds lines/curves.

    Class Attributes:
        __instance_count: An int which keeps track of the number of Edge class object that have been instantiated.
        default_cs_area: Default cross-sectional area of an edge if no cross-sectional area is given in instantiation.
        min_cs_area: Minimum cross-sectional area any edge can have.
        trigger_to_virtualise: Value at which the edge will turn virtual.
        trigger_to_realise: Value at which the edge will turn real.

    Attributes:
        id: Unique id of Edge object.
        material: Material of edge.
        line: Line which the edge bounds.
        virtual: Boolean that if true, the edge will be virtual and will not be represented geometrically or in an analysis
         and exists of to maintain the topology of the cell.
        cs_type: Cross-sectional type of edge.
    """
    __instance_count = 0
    default_cs_area = 0.1
    min_cs_area = 0.02
    max_cs_area = None
    min_dim1 = 0.1
    max_dim1 = None
    min_dim2 = 0.1
    max_dim2 = None
    trigger_to_virtualise = 1.0
    trigger_to_realise = -1.0

    @classmethod
    def increment_instance_count(cls):
        """Increments the instance count for the Edge class."""
        cls.__instance_count += 1

    @classmethod
    def reset_instance_count(cls):
        """Resets the instance count for the Edge class."""
        cls.__instance_count = 0

    @classmethod
    def set_instance_count(cls, count):
        """Sets the instance count for the Edge class."""
        cls.__instance_count = count

    @classmethod
    def remove_instance(cls):
        """Removes the instance count for the Edge class."""
        cls.__instance_count -= 1

    def __init__(self, material: Type[Material], line: Type[Line], cs_area: float=None, dim_1=None, dim_2=None, cs_type: Type[CSType]=CSType.circle, virtual: bool=False):
        """Inits the Edge object with a unique id."""
        self.id = self.__instance_count
        Edge.increment_instance_count()

        self.line = line
        self.material = material
        self.virtual = virtual
        self.cs_type = cs_type
        self.on_boundary = False
        self.b_id = None
        self._has_fin = False

        if self.cs_type is CSType.circle:
            if cs_area is None:
                self.cs_area = self.default_cs_area
            elif cs_area < self.min_cs_area:
                self.cs_area = self.min_cs_area
            elif self.max_cs_area != None:
                if cs_area > self.max_cs_area:
                    self.cs_area = self.max_cs_area
                else:
                    self.cs_area = cs_area
            else:
                self.cs_area = cs_area

        elif self.cs_type is CSType.rectangle:
            if dim_1 < self.min_dim1:
                self.dim_1 = 0.0
            elif self.max_dim1 != None:
                if dim_1 > self.max_dim1:
                    self.dim_1 = self.max_dim1
                else:
                    self.dim_1 = dim_1
            else:
                self.dim_1 = dim_1                    
            if dim_2 < self.min_dim2:
                self.dim_2 = 0.0
            elif self.max_dim2 != None:
                if dim_2 > self.max_dim2:
                    self.dim_2 = self.max_dim2
                else:
                    self.dim_2 = dim_2
            else:
                self.dim_2 = dim_2
            self.cs_area = self.dim_1 * self.dim_2
        self._moment_of_inertia = None
        self._length = None
        self._volume = None
        self._midpoint = None
        self._normal = None
        self.length_norm = None
        self.cs_area_norm = None
        self._el_id = None
        self.channel_temp = []

    def __repr__(self) -> str:
        """Defines the printable representation of the Edge object."""
        return repr(f"Edge id {self.id} (points [{self.line.start_pt.id}, {self.line.end_pt.id}])")

    @property
    def start_point(self):
        return self.line.start_pt

    @property
    def end_point(self):
        return self.line.end_pt

    @property
    def points(self):
        return [self.line.start_pt, self.line.end_pt]

    @property
    def length(self):
        """Get length of edge.

        Returns:
            Length as float.
        """
        self._length = self.line.length
        return self._length

    @property
    def volume(self):
        """Get volume of edge.

        Returns:
            Volume as float.
        """
        if self.cs_type is CSType.circle:
            self._volume = self.line.length * self.cs_area
        elif self.cs_type is CSType.rectangle:
            if self.dim_1 == 0 or self.dim_2 == 0:
                self._volume = 0
            else:
                self._volume = self.line.length * self.dim_1 * self.dim_2
        
        return self._volume

    @property
    def second_moment_of_area(self):
        """Get second moment of area of edge based on its cross-sectional type.

        Returns:
            Second moment of area as float.
        """
        if self.cs_type is CSType.circle:
            radius = self.cs_area * (1 / np.pi)**0.5
            self._moment_of_inertia = radius**4 * (np.pi / 4)
            return self._moment_of_inertia
        elif self.cs_type is CSType.rectangle:
            self._moment_of_inertia = (self.dim_2 * self.dim_1 ** 3) / 12
            return self._moment_of_inertia

    @property
    def midpoint(self):
        """Get midpoint of edge.

        Returns:
            Coordinate of midpoint as array.
        """
        self._midpoint = self.line.midpoint
        return self._midpoint

    @property
    def radius(self):
        """Get radius of edge

        Returns:
            Float of edge radius
        """
        self._radius = np.sqrt(self.cs_area / np.pi)
        return self._radius

    @property
    def surface_area(self):
        """ Get surface area of edge

        Returns:
            Float of surface area (mm2)
        """
        self._surface_area = np.pi * self.radius * 2 * self.length
        return self._surface_area

    @property
    def as_occ(self):
        start_point = self.start_point.as_array
        start_point_occ = gp_Pnt(start_point[0], start_point[1], start_point[2])
        end_point = self.end_point.as_array
        end_point_occ = gp_Pnt(end_point[0], end_point[1], end_point[2])
        edge = BRepBuilderAPI_MakeEdge(self.line.as_occ, start_point_occ, end_point_occ)
        return edge.Edge()
    
    @property
    def cs_ratio(self):
        """Returns the ratio of the depth to width dimensions of the rectangular edge"""
        if self.cs_type == CSType.rectangle:
            if self.dim_1 and self.dim_2 != 0.0:
                return self.dim_1 / self.dim_2
            else:
                return 0.0

    def update_channel_cs_area(self, delta):
        """Updates the cross-sectional area of the edge based on an area delta signal.

        - Method takes an delta value and adds it to the cross-sectional area of the edge.
        - If the updated cross-sectional area is below the minimum value, the cross-sectional area is set to this minimum.

        Args:
            delta: A signal from the GRN dictating how to change the cross-sectional area.

        Returns:
            None
        """

        self.cs_area += delta * 0.1

        if self.cs_area < self.min_cs_area:
            self.cs_area = self.default_cs_area
        elif self.max_cs_area is not None:
            if self.cs_area > self.max_cs_area:
                self.cs_area = self.max_cs_area

    def update_edge_depth(self, delta):
        """Updates the edge depth based on a delta signal.

        - Method takes an delta value and adds it to the dim_1 of edge.
        - If the updated depth is below the minimum value, the depth is set to this minimum.

        Args:
            delta: A signal from the GRN dictating how to change the edge depth

        Returns:
            None
        """

        self.dim_1 += delta

        if self.dim_1 < self.min_dim1:
            self.dim_1 = 0.0
            self.dim_2 = 0.0
     
    def update_edge_width(self, delta):
        """Updates the edge width based on a delta signal.

        - Method takes an delta value and adds it to the dim_2 of edge.
        - If the updated width is below the minimum value, the width is set to this minimum.

        Args:
            delta: A signal from the GRN dictating how to change the edge width

        Returns:
            None
        """

        self.dim_2 += delta

        if self.dim_2 < self.min_dim2:
            self.dim_2 = 0.0
            self.dim_1 = 0.0

class Face:
    """Face class to hold line objs, material and thickness data associate with the cell faces"""

    __instance_count = 0
    min_thickness = 0.0
    trigger_to_virtualise = 1.0
    trigger_to_realise = -1.0
    min_thickness = 0.1
    max_thickness = None

    @classmethod
    def increment_instance_count(cls):
        cls.__instance_count += 1

    @classmethod
    def reset_instance_count(cls):
        cls.__instance_count = 0

    @classmethod
    def set_instance_count(cls, count):
        cls.__instance_count = count

    @classmethod
    def remove_instance(cls):
        cls.__instance_count -= 1

    def __init__(self, material: Type[Material], edges: list, thickness: float=0.0, surface=None, virtual: bool=False):
        self.id = self.__instance_count
        Face.increment_instance_count()
        self.material = material
        self._edges = edges
        self._points = {}
        self._centroid = None
        self._surface = surface
        self.thickness = float(thickness)
        self.on_boundary = False
        self.virtual = virtual

    def __repr__(self) -> str:
        """Defines the printable representation of the Face object."""
        return repr(f"Face id {self.id} (Edges [{self.edges[0].id}, {self.edges[1].id}, {self.edges[2].id}])")

    @property
    def edges(self):
        return self._edges

    @property
    def points(self):
        self._points = {}
        for edge in self.edges:
            line = edge.line
            points = line.points
            for pt in points:
                self._points[pt.id] = pt
        return self._points

    @property
    def surface_area(self):
        """Use Heron's Formula to calc area from edge lengths"""
        edge_lengths = []
        for edge in self.edges:
            edge_lengths.append(edge.length)
        edge_lengths = np.asarray(edge_lengths)

        sp = np.sum(edge_lengths) / 2

        self._surface_area = np.sqrt(sp * (sp-edge_lengths[0]) * (sp-edge_lengths[1]) * (sp-edge_lengths[2]))

        return self._surface_area

    @property
    def centroid(self):
        centroid = np.array([0, 0, 0])
        for point in self.points.values():
            centroid = centroid + point.get_point_array()
        centroid /= 3
        self._centroid = centroid
        return self._centroid

    @property
    def normal(self):
        curve_loop = self.curve_loop
        v1 = self.edges[0].line.direction
        v2 = self.edges[1].line.direction
        face_normal = np.cross(v1, v2)
        self._normal = face_normal / np.linalg.norm(face_normal)
        return self._normal

    @property
    def surface(self):
        return self._surface

    @property
    def as_occ(self):
        edges = self.edges
        wire = BRepBuilderAPI_MakeWire(edges[0].as_occ).Wire()

        for i in range(1, len(edges)):
            wire = BRepBuilderAPI_MakeWire(wire, edges[i].as_occ).Wire()

        bounded_face = BRepBuilderAPI_MakeFace(self._surface.as_occ, wire)

        return bounded_face.Face()

    @property
    def curve_loop(self):
        """
        Loops through the edges of the face, returning the ordered index of the edges that makes a closed loop.
        Direction of the edge is indicated by negative/positive sign.
        Must start with Edge 0 if present in face, as 0 can't be negative.
        Checks for matching end and start points of consecutive edges, reversing if needed before moving to next edge.
        """
        edges = self.edges
        # start the loop with any edge with id = 0
        start_edge = [edge for edge in edges if edge.id == 0]
        if len(start_edge) == 0:
            start_edge = self.edges[0]
            curve_loop = [start_edge.id]
            loop_start_pt = start_edge.line.end_pt.id
        else:
            curve_loop = [start_edge[0].id]
            loop_start_pt = start_edge[0].line.end_pt.id
        next_edge = [edge for edge in edges if edge.line.start_pt.id == loop_start_pt]
        if len(next_edge) > 0:
            curve_loop.append(next_edge[0].id)
            last_edge = [edge for edge in edges if edge.line.id not in curve_loop]
            if last_edge[0].line.start_pt.id == next_edge[0].line.end_pt.id:
                curve_loop.append(last_edge[0].id)
            else:
                curve_loop.append(-last_edge[0].id)
        else:
            next_edge = [edge for edge in edges if edge.line.end_pt.id == loop_start_pt and edge.line.id not in curve_loop]
            if len(next_edge) > 0:
                curve_loop.append(-next_edge[0].id)
                last_edge = [edge for edge in edges if edge.line.id not in curve_loop and -edge.line.id not in curve_loop]
                if len(last_edge) > 0:
                    if last_edge[0].line.start_pt.id == next_edge[0].line.start_pt.id:
                        curve_loop.append(last_edge[0].id)
                    else:
                        curve_loop.append(-last_edge[0].id)

        return curve_loop

    def update_thickness(self, delta):
        """Updates the thickness of the face based on a delta signal.

        - Method takes an delta value and adds it to the thickness of the face.
        - If the updated thickness is below the minimum value, the thickness is set to this minimum.

        Args:
            delta: A signal from the GRN dictating how to change the thickness.

        Returns:
            None
        """
        if self.thickness + delta < self.min_thickness:
            self.thickness = self.min_thickness
        elif self.max_thickness is not None:
            if self.thickness + delta > self.max_thickness:
                self.thickness = self.max_thickness
        else:
            self.thickness += delta

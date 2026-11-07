"""Geometry Classes"""

from array import ArrayType
import numpy as np
from typing import Type
from abc import ABC, abstractmethod

from OCC.Core.gp import gp_Pnt, gp_Dir
from OCC.Core.Geom import Geom_Plane, Geom_Line


class Point:
    """Zero-dimensional geometric entity.

    Class Attributes:
        __instance_count: An int which keeps track of the number of Point class object that have been instantiated.

    Attributes:
        id: Unique id of Point object.
        x: X coordinate of the point.
        y: Y coordinate of the point.
        z: Z coordinate of the point.
        virtual: Boolean that if true, the point will be virtual and will not be represented geometrically or in an analysis
        and exists of to maintain the topology of the cell.
        virtual_edges: List of Booleans indicating if an edge of point is virtual or not.
        is_loaded: Boolean indicating if the point is loaded.
        _coords_norm: Array of normalised point coordinates.
    """
    __instance_count = 0
    coord_delta_ratio = 1.0

    @classmethod
    def increment_instance_count(cls):
        """Increments the instance count for the Point class."""
        cls.__instance_count += 1

    @classmethod
    def reset_instance_count(cls):
        """Resets the instance count for the Point class."""
        cls.__instance_count = 0

    @classmethod
    def set_instance_count(cls, count):
        """Sets the instance count for the Point class."""
        cls.__instance_count = count

    @classmethod
    def remove_instance(cls):
        """Removes the instance count for the Point class."""
        cls.__instance_count -= 1

    def __init__(self, x: float, y: float, z: float):
        """Inits the Point object with a unique id."""
        self.id = self.__instance_count
        Point.increment_instance_count()

        self.x = x
        self.y = y
        self.z = z
        self.virtual = False
        self.virtual_edges = None
        self.is_loaded = False
        self.loadcases = []
        self._coords_norm = None

    def __repr__(self):
        """Defines the printable representation of the Point object."""
        return repr(f"Point id {self.id} at location ({self.x}, {self.y}, {self.z})")

    @property
    def as_array(self):
        """Get point coordinates as numpy array.

        Returns:
            Numpy array of point coordinates.
        """
        return np.array([self.x, self.y, self.z])
    
    @property
    def as_string(self):
        """Get point coordinates as string.

        Returns:
            String of point coordinates.
        """
        return f"{self.x}, {self.y}, {self.z}"

    @property
    def as_occ(self):
        """Get point as an PythonOCC object.

        Returns:
            PythonOCC Geom_Pnt object.
        """
        return gp_Pnt(self.x, self.y, self.z)

    def get_point_array(self):
        """Get point coordinates as numpy array.

        Returns:
            Numpy array of point coordinates.
        """
        return np.array([self.x, self.y, self.z])

    def update_point_coord(self, coord_delta):
        """Add each delta value to each coord value

        Args:
            coord_delta: Three delta values for the point's coordinates.
        """
        self.x += coord_delta[0] * self.coord_delta_ratio
        self.y += coord_delta[1] * self.coord_delta_ratio
        self.z += coord_delta[2] * self.coord_delta_ratio


class Line:
    """One-dimensional geometric entity.

    Class Attributes:
        __instance_count: An int which keeps track of the number of Line class object that has been instantiated.

    Attributes:
        id: Unique id of Line object.
        start_point: The start point of the line.
        end_point: The end point of the line.
        _points: List of Point objects.
        _length: Float of length of edge.
        _direction: Numpy array of direction of edge.
        _midpoint: Numpy array of midpoint of edge.
        _normal: Numpy array of normal of edge.
    """
    __instance_count = 0

    @classmethod
    def increment_instance_count(cls):
        """Increments the instance count for the Line class."""
        cls.__instance_count += 1

    @classmethod
    def reset_instance_count(cls):
        """Resets the instance count for the Line class."""
        cls.__instance_count = 0

    @classmethod
    def set_instance_count(cls, count):
        """Sets the instance count for the Line class."""
        cls.__instance_count = count

    @classmethod
    def remove_instance(cls):
        """Removes the instance count for the Line class."""
        cls.__instance_count -= 1

    def __init__(self, start_point: Type[Point], end_point: Type[Point]):
        """Inits the Line object with a unique id."""
        self.id = self.__instance_count
        Line.increment_instance_count()

        self.start_pt = start_point
        self.end_pt = end_point
        self._points = []
        self._length = None
        self._direction = None
        self._midpoint = None
        self._normal = None

    def __repr__(self):
        """Defines the printable representation of the Line object."""
        return repr(f"Line id {self.id}")

    @property
    def points(self):
        """Get points of line.

        Returns:
            Two point objects of line.
        """
        self._points = [self.start_pt, self.end_pt]
        return self._points

    @property
    def direction(self):
        """Get direction of line.

        Returns:
            Direction of line.
        """
        self._direction = np.subtract(self.end_pt.get_point_array(), self.start_pt.get_point_array())
        return self._direction

    @property
    def length(self):
        """Get length of line.

        Returns:
            Length of line.
        """
        line_vector = np.abs(np.subtract(self.end_pt.get_point_array(), self.start_pt.get_point_array()))
        self._length = np.linalg.norm(line_vector)
        return self._length

    @property
    def midpoint(self):
        """Get midpoint of line.

        Returns:
            Midpoint of line as array.
        """
        self._midpoint = (self.end_pt.get_point_array() + self.start_pt.get_point_array()) / 2
        return self._midpoint

    @property
    def as_occ(self):
        """Get line as an PythonOCC object.

        Returns:
            PythonOCC Geom_Line object.
        """
        gp_pnt = gp_Pnt(self.start_pt.x, self.start_pt.y, self.start_pt.z)
        direction = self.end_pt.as_array - self.start_pt.as_array
        gp_dir = gp_Dir(direction[0], direction[1], direction[2])
        line = Geom_Line(gp_pnt, gp_dir)
        return line

    def get_indexed_line(self):
        return sorted((self.start_pt.id, self.end_pt.id))


class BezierCurve:
    """One-dimensional geometric entity.

    Source: https://github.com/torresjrjr/Bezier.py

    Class Attributes:
        __instance_count: An int which keeps track of the number of Curve class object that has been instantiated.

    Attributes:
        id: Unique id of Curve object.
        t_points: interpolation points along the curve
        control_points: The control points of the curve.
    """
    __instance_count = 0

    @classmethod
    def increment_instance_count(cls):
        """Increments the instance count for the Line class."""
        cls.__instance_count += 1

    @classmethod
    def reset_instance_count(cls):
        """Resets the instance count for the Line class."""
        cls.__instance_count = 0

    @classmethod
    def set_instance_count(cls, count):
        """Sets the instance count for the Line class."""
        cls.__instance_count = count

    @classmethod
    def remove_instance(cls):
        """Removes the instance count for the Line class."""
        cls.__instance_count -= 1

    def __init__(self, t_points: Type[ArrayType], control_points: Type[ArrayType]):
        """Inits the Curve object with a unique id."""
        self.id = self.__instance_count
        BezierCurve.increment_instance_count()

        self._t_points = t_points
        self._control_points = control_points
        self.curve_points = None
        self._create_curve()

    @property
    def t_points(self):
        return self._t_points

    @property
    def control_points(self):
        return self._control_points

    def two_points(self, t, p_1, p_2):
        """Returns a point between p_1 and p_2, parametised by t.

        Args:
            t: A parameterisation as a float/int.
            p_1: A point as a numpy array.
            p_2: A point as a numpy array.

        Returns:
            A point as a numpy array.
        """
        if not isinstance(p_1, np.ndarray) or not isinstance(p_2, np.ndarray):
            raise TypeError('Points must be an instance of the numpy.ndarray!')
        if not isinstance(t, (int, float)):
            raise TypeError('Parameter t must be an int or float!')

        q_1 = (1 - t) * p_1 + t * p_2
        return q_1

    def points(self, t, points):
        """Returns a list of points interpolated by the Bezier process.

        Args:
            t: A parameterisation as a float/int.
            points: Points as a list of numpy arrays.

        Returns:
            Points as a list of numpy arrays.
        """
        newpoints = []
        for i1 in range(0, len(points) - 1):
            newpoints += [self.two_points(t, points[i1], points[i1 + 1])]
        return newpoints

    def point(self, t, points):
        """Returns a point interpolated by the Bezier process.

        Args:
            t: A parameterisation as a float/int.
            points: Points as a list of numpy arrays.

        Returns:
            A point as numpy array.
        """
        newpoints = points
        while len(newpoints) > 1:
            newpoints = self.points(t, newpoints)
        return newpoints[0]

    def _create_curve(self):
        """Returns a list of points defining a Bezier curve.

        Args:
            t_values: A parameterisation as a list of floats/ints.
            points: Points as a list of numpy arrays.

        Returns:
           None
        """
        if not hasattr(self.t_points, '__iter__'):
            raise TypeError("`t_values` Must be an iterable of integers or floats, of length greater than 0 .")
        if len(self.t_points) < 1:
            raise TypeError("`t_values` Must be an iterable of integers or floats, of length greater than 0 .")
        if not isinstance(self.t_points[0], (int, float)):
            raise TypeError("`t_values` Must be an iterable of integers or floats, of length greater than 0 .")

        curve = np.array([[0.0] * len(self.control_points[0])])
        for t in self.t_points:
            curve = np.append(curve, [self.point(t, self.control_points)], axis=0)
        curve = np.delete(curve, 0, 0)

        self.curve_points = curve


class Surface:
    """Abstract class for surfaces."""
    @property
    @abstractmethod
    def as_occ(self):
        pass


class Plane(Surface):
    """Two-dimensional geometric entity.

    Class Attributes:
        __instance_count: An int which keeps track of the number of Plane class object that has been instantiated.

    Attributes:
        id: Unique id of Plane object.
        normal: normal of the plane
    """
    __instance_count = 0

    @classmethod
    def increment_instance_count(cls):
        """Increments the instance count for the Plane class."""
        cls.__instance_count += 1

    @classmethod
    def reset_instance_count(cls):
        """Resets the instance count for the Plane class."""
        cls.__instance_count = 0

    @classmethod
    def set_instance_count(cls, count):
        """Sets the instance count for the Plane class."""
        cls.__instance_count = count

    @classmethod
    def remove_instance(cls):
        """Removes the instance count for the Plane class."""
        cls.__instance_count -= 1

    def __init__(self, **kwargs):
        """Inits the Plane object with a unique id.

        Kwargs:
            - Option 1 => (point: Type[Point], dir_x: Type[float], dir_y: Type[float], dir_z: Type[float])
            - Option 2 => (point_0: Type[Point], point_1: Type[Point], point_2: Type[Point])
        """
        self.id = self.__instance_count

        if "point" in kwargs and "dir_x" in kwargs and "dir_y" in kwargs and "dir_z" in kwargs:
            self.point = kwargs.get("point")
            self.direction = np.array([float(kwargs.get("dir_x")), float(kwargs.get("dir_y")), float(kwargs.get("dir_z"))])
        elif "point_0" in kwargs and "point_1" in kwargs and "point_2" in kwargs:
            self.point = kwargs.get("point_0")
            self.point_1 = kwargs.get("point_1")
            self.point_2 = kwargs.get("point_2")
            self.direction = None
            self.__set_direction()
        else:
            raise Exception("Incorrect args given to Plane object.")

        Plane.increment_instance_count()

    def __repr__(self):
        """Defines the printable representation of the Planar object."""
        return repr(f"Plane id {self.id}.")

    @property
    def as_occ(self):
        point = self.point.as_array
        gp_pnt = gp_Pnt(point[0], point[1], point[2])
        gp_dir = gp_Dir(self.direction[0], self.direction[1], self.direction[2])
        plane = Geom_Plane(gp_pnt, gp_dir)
        return plane

    def is_point_on_plane(self, point=Type[ArrayType]):
        """Checks if point lies on the plane.

        Args:
            point: array of point location

        Returns:
            Boolean dictating if the point is on the plane.
        """
        a = self.direction[0] * (point[0] - self.point.x)
        b = self.direction[1] * (point[1] - self.point.y)
        c = self.direction[2] * (point[2] - self.point.z)

        if (a + b + c) == 0:
            return True

    def __set_direction(self):
        """Set the direction by calculating the cross product between two point vectors"""
        vec_01 = np.subtract(self.point_1.get_point_array(), self.point.get_point_array())
        vec_02 = np.subtract(self.point_2.get_point_array(), self.point.get_point_array())

        self.direction = np.cross(vec_01, vec_02)

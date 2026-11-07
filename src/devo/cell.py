"""Module for different cell classes."""


import numpy as np

from typing import OrderedDict, Type, List

from src.devo.topology import Edge, Face
from src.devo.geometry import Point, Line
from src.environment.materials import Material


class Cell:
    """Class for all cells.

    - The cell object is the interface between the organism and the GRN.

    Attributes:
        material: A Material object which holds the material properties of the cell.
        physical_state: A PhysicalState object which holds certain properties of the cell such as the forces acting upon it.
    """
    def __init__(self, material: Type[Material]):
        self.material = material


class Triangle(Cell):
    """Cell class with the topology of a triangle.

    Class Attributes:
        __instance_count: An int which keeps track of the number of Triangle class object that have been instantiated.

    Attributes:
        id: Unique id of Triangle cell object.
        material: A Material object which holds the material properties of the cell.
        physical_state: A PhysicalState object which holds certain properties of the cell such as the forces acting upon it.
        edge_0: An edge object of the first edge of the triangle cell.
        edge_1: An edge object of the second edge of the triangle cell.
        edge_2: An edge object of the third edge of the triangle cell.
        depth: Number of cell divisions before the creation of the cell in its ancestry.
    """
    __instance_count = 0
    division_depth = 3
    trigger_for_addition = 1.0
    trigger_for_fin_addition = 0.5
    trigger_for_division = 1.0
    fin_growth_delta = 1.0
    fin_height = 0.0

    @classmethod
    def increment_instance_count(cls):
        """Increments the instance count for the Triangle cell class."""
        cls.__instance_count += 1

    @classmethod
    def reset_instance_count(cls):
        """Resets the instance count for the Triangle cell class."""
        cls.__instance_count = 0

    @classmethod
    def set_instance_count(cls, count):
        """Sets the instance count for the Triangle cell class."""
        cls.__instance_count = count

    @classmethod
    def remove_instance(cls):
        """Removes the instance count for the Triangle cell class."""
        cls.__instance_count -= 1

    def __init__(self, edge_0: Type[Edge], edge_1: Type[Edge], edge_2: Type[Edge], material: Type[Material], depth=0, thickness=None, cell_points=None, is_fin=False):
        super().__init__(material)
        """Inits the Triangle cell object with a unique id."""
        self.id = self.__instance_count
        Triangle.increment_instance_count()

        self.edge_0 = edge_0
        self.edge_1 = edge_1
        self.edge_2 = edge_2
        self.depth = int(depth)
        self._points = {}
        self._lines = {}
        self._edges = {}
        self._face = []
        self.initialisation_checks()
        self._edge_normals = {}
        self._unit_normal = None
        self._is_fin = is_fin
        self._cell_points = cell_points

        # build faces of triangle cell if thickness value is present
        self._face_0 = None
        if thickness is not None:
            self._face_0 = self._build_face(thickness)
            self.normal = self.unit_normal

    def __repr__(self) -> str:
        """Defines the printable representation of the Triangle cell object."""
        return repr(f"Triangle cell {self.id} with edges {self.edge_0.id}, {self.edge_1.id} & {self.edge_2.id}")
    
    @property
    def pos_hash(self):
        # return a hash made from the value of points of line obkectgs of the edges
        return hash(str(self.edge_0.line.points[0].as_string) + str(self.edge_0.line.points[1].as_string))

    @property
    def edge_strings(self):
        return str(self.edge_0.line.points[0].as_string) + str(self.edge_0.line.points[1].as_string)

    @property
    def faces(self):
        if self._face_0 is not None:
            self._face = [self._face_0]
            return self._face
        else:
            return self._face

    @property
    def edges(self):
        """Get edges of triangle cell.

        Returns:
            Three edge objects of triangle cell.
        """
        self._edges = [self.edge_0, self.edge_1, self.edge_2]
        return self._edges


    @property
    def lines(self):
        """Get lines of triangle cell.

        Returns:
            Three line objects of triangle cell.
        """
        self._lines = [self.edge_0.line, self.edge_1.line, self.edge_2.line]
        return self._lines

    @property
    def points(self):
        """Get points of triangle cell.

        Returns:
            Three point objects of triangle cell.
        """
        self._points = []
        for line in self.lines:
            for pt in line.points:
                if pt not in self._points:
                    self._points.append(pt)
        return self._points

    @property
    def indexed_faces(self):
        return [self.faces[0].id]

    @property
    def unit_normal(self):
        """Get unit face normal of triangle cell.

        Returns:
            Unit face normal of triangle cell as array.
        """
        if self._cell_points is not None:
            A = np.subtract(self._cell_points[1].as_array, self._cell_points[0].as_array)
            B = np.subtract(self._cell_points[2].as_array, self._cell_points[0].as_array)
            norm = np.cross(A, B)
        else:
            d_0 = self.lines[0].direction
            d_1 = self.lines[1].direction
            norm = np.cross(d_0, d_1)
        self._unit_normal = norm / np.linalg.norm(norm)

        return self._unit_normal
    
    @property
    def surface_area(self):
        """Get the surface area of a triangle cell from its edge lengths
        using Heron's Formula
        
        Returns:
            float of cell's surface area
        """
        a = self.edge_0.length
        b = self.edge_1.length
        c = self.edge_2.length
        s = (a + b + c) / 2
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))

        return area
    
    @property
    def edge_normals(self):
        """Get normals of triangle cell.

        Returns:
            Three point objects of triangle cell.
        """
        d_0 = self.edges[0].line.direction
        d_1 = self.edges[1].line.direction
        d_2 = self.edges[2].line.direction
        cell_norm = self.unit_normal

        n_0 = np.cross(d_0, cell_norm)
        n_0_norm = n_0 / np.linalg.norm(n_0)

        n_1 = np.cross(d_1, cell_norm)
        n_1_norm = n_1 / np.linalg.norm(n_1)

        n_2 = np.cross(d_2, cell_norm)
        n_2_norm = n_2 / np.linalg.norm(n_2)

        self.edges[0]._normal = n_0_norm
        self.edges[1]._normal = n_1_norm
        self.edges[2]._normal = n_2_norm
        self._edge_normals = {self.edges[0]: n_0_norm, self.edges[1]: n_1_norm, self.edges[2]: n_2_norm}
        return self._edge_normals

    def initialisation_checks(self):
        """Checks for initialisation of cell to ensure correct number of geometric and topology entities were given."""
        assert len(self.points) == 3
        assert len(self.lines) == 3
        assert len(self.edges) == 3

    def _build_face(self, thickness):
        return Face(self.material, self.edges, thickness)

    def get_centroid(self, cell_points):
        """Get centroid of cell.

        Args:
            cell_points: Points of triangle cell.

        Returns:
            Point object of centroid.
        """
        centroid = np.array([0, 0, 0])
        for point in cell_points:
            centroid = centroid + point.get_point_array()

        centroid = centroid / 3
        return Point(x=centroid[0], y=centroid[1], z=centroid[2])

    def divide(self):
        """Mechanism for cell to divide.

        - Generates a new point at the centroid of the cell. Three new edges are created from the centroid and the
          points of the triangle.
        - Three new cells are created using the new edges and the original edges of the cell.

        Returns:
            Three new cell objects
        """
        new_cells = []
        new_edges = []
        points = self.points

        if self.depth < self.division_depth:
            centroid = self.get_centroid(points)

            for point in points:
                new_line = Line(point, centroid)
                if point.virtual:
                    new_edges.append(Edge(self.material, new_line, virtual=True))
                else:
                    new_edges.append(Edge(self.material, new_line))

            for edge in self.edges:
                adj_edges = []
                line_points = edge.line.points
                for new_edge in new_edges:
                    for point in new_edge.line.points:
                        if point in line_points:
                            adj_edges.append(new_edge)

                new_cells.append(Triangle(edge, adj_edges[0], adj_edges[1], self.material, depth=self.depth + 1))

        return new_cells

    def addition(self, signal):
        """Mechanism to add a new cell to the boundary of the cell.

        - Given a signal to add a new cell for each line of the cell.
        - Check which lines want to add a new cell on them.
        - Check if the lines are on the boundary of the organism.
        - Add a new point normal to the line outside the cell.
        - This new point will be places at a distance half of the length of the line.
        - Generate two new lines/edges connected to the new cell and the parent cell.

        Returns:
            New cells if edge is on boundary.
        """
        new_cells = []
        points = self.points
        for i, edge in enumerate(self.edges):
            if signal[i] >= self.trigger_for_addition and edge.on_boundary:
                midpoint = edge.midpoint
                edge_points = edge.points

                opposite_point = None
                for point in points:
                    if point not in edge_points:
                        opposite_point = point

                vec = midpoint - opposite_point.as_array
                point_pos = vec + midpoint

                new_point = Point(x=point_pos[0], y=point_pos[1], z=point_pos[2])
                line_0 = Line(edge.start_point, new_point)
                line_1 = Line(new_point, edge.end_point)

                edge_0 = Edge(self.material, line_0)
                edge_1 = Edge(self.material, line_1)

                edge_0.on_boundary = True
                edge_1.on_boundary = True
                edge.on_boundary = False

                new_cells.append(Triangle(edge, edge_0, edge_1, self.material))

        return new_cells

    def fin_addition(self, signal):
        """Mechanism to add a new fin cell to the edge of the cell.

        - Given a signal to add a new cell for each line of the cell.
        - Check which lines want to add a new cell on them.
        - Add a new point, normal to the cell.
        - This new point will be placed at a distance half of the length of the line.
        - Generate two new lines/edges connected to the new cell and the parent cell.

        Returns:
            New cells if edge has been triggered.
        """
        new_cells = []

        if self._is_fin:
            # fins can't have fins added to themselves
            pass
        else:

            points = self.points
            # normal = None
            # if self._face_0 is not None:
            #     #normal = self._face_0.normal
            #     normal = np.array([0.0, 0.0, 1.0])
            #     thickness = self._face_0.thickness
            # else:
            #     edge_normals = self.edge_normals
            #     thickness = 0.5
            thickness = self._face_0.thickness
            for i, edge in enumerate(self.edges):
                # if i == 0:
                if signal[i] >= self.trigger_for_fin_addition:
                # force fin addition
                # if True:
                    # if not edge._has_fin:
                    # midpoint = edge.midpoint
                    # manually calc midpoint 
                    midpoint = edge.start_point.as_array + (edge.end_point.as_array - edge.start_point.as_array) / 2
                    value_over_thresh = (signal[i] - self.trigger_for_fin_addition) / (1.0 - self.trigger_for_fin_addition)
                    # value_over_thresh = 1.0

                    # fixed fin height
                    # self.fin_height = edge.line.length / 2
                    
                    # variable fin height with a minimum
                    min_fin_height = 0.2
                    addition = value_over_thresh * self.fin_growth_delta
                    self.fin_height += addition
                    self.fin_height = max(self.fin_height, min_fin_height)

                    # if normal is None:
                    #     normal = edge_normals[edge]
                    try:
                        normal = self._unit_normal
                    except AttributeError:
                        normal = self._face_0.normal

                    point_pos = (-normal * self.fin_height) + midpoint
                    # point_pos = (normal * self.fin_height) + midpoint

                    new_point = Point(x=point_pos[0], y=point_pos[1], z=point_pos[2])
                    line_0 = Line(edge.start_point, new_point)
                    line_1 = Line(new_point, edge.end_point)

                    edge_0 = Edge(self.material, line_0)
                    edge_1 = Edge(self.material, line_1)

                    edge_0.on_boundary = True
                    edge_1.on_boundary = True
                    edge.on_boundary = False
                    edge._has_fin = True
                    edge_0._has_fin = False
                    edge_1._has_fin = False

                    fin_cell = Triangle(edge, edge_0, edge_1, self.material, thickness=thickness, is_fin = True)
                    fin_cell._is_fin = True
                    new_cells.append(fin_cell)
        # print('new cells: ', new_cells)
        return new_cells


class Tet(Cell):
    """Cell class with the topology of a tetrahedral (tet).

    Class Attributes:
        instance_count: An int which keeps track of the number of Tet class object that have been instantiated.

    Attributes:
        id: Unique id of Tet cell object.
        material: A Material object which holds the material properties of the cell.
        face_0: An face object of the first face of the tet cell.
        face_1: An face object of the second face of the tet cell.
        face_2: An face object of the third face of the tet cell.
        face_3: An face object of the fourth face of the tet cell.
    """
    instance_count = 0
    division_depth = 3
    trigger_for_addition = 1.0
    trigger_for_division = 1.0

    @classmethod
    def increment_instance_count(cls):
        """Increments the instance count for the Tet cell class."""
        cls.instance_count += 1

    @classmethod
    def reset_instance_count(cls):
        """Resets the instance count for the Tet cell class."""
        cls.instance_count = 0

    @classmethod
    def set_instance_count(cls, count):
        """Sets the instance count for the Tet cell class."""
        cls.instance_count = count

    @classmethod
    def remove_instance(cls):
        """Removes the instance count for the Tet cell class."""
        cls.__instance_count -= 1

    def __init__(self, material: Type[Material], face_0,
                 face_1, face_2, face_3, depth=0):
        super().__init__(material)
        """Inits the Tet cell object with a unique id."""
        self.id = self.instance_count
        Tet.increment_instance_count()

        self.face_0 = face_0
        self.face_1 = face_1
        self.face_2 = face_2
        self.face_3 = face_3
        self.depth = int(depth)
        self._points = []
        self._lines = []
        self._edges = []
        self._faces = []

        self.nb_0 = None
        self.nb_1 = None
        self.nb_2 = None
        self.nb_3 = None

    def __repr__(self) -> str:
        """Defines the printable representation of the Tet cell object."""
        return repr(f"Tet cell {self.id}")

    @property
    def faces(self):
        """Get faces of tet cell.

        Returns:
            Four face objects of tet cell.
        """
        self._faces = [self.face_0, self.face_1, self.face_2, self.face_3]
        return self._faces

    @property
    def edges(self):
        """Get edges of tet cell.

        Returns:
            Six edge objects of tet cell.
        """
        self._edges = []
        these_edges = {}
        for face in self.faces:
            edges = face.edges
            for edge in edges:
                these_edges[edge.id] = edge
        for k in sorted(these_edges):
            self._edges.append(these_edges[k])
        return self._edges

    @property
    def lines(self):
        """Get lines of tet cell.

        Returns:
            Six line objects of tet cell.
        """
        self._lines = []
        [self._lines.append(edge.line) for edge in self.edges]
        return self._lines

    @property
    def points(self):
        """Get points of tet cell.

        Returns:
            Four point objects of tet cell.
        """
        self._points = []
        for line in self.lines:
            for pt in line.points:
                if pt not in self._points:
                    self._points.append(pt)
        return self._points

    @property
    def indexed_faces(self):
        return [self.face_0.id, self.face_1.id, self.face_2.id, self.face_3.id]

    def get_centroid(self, cell_points):
        """Get centroid of cell.

        Args:
            cell_points: Points of triangle cell.

        Returns:
            Point object of centroid.
        """
        centroid = np.array([0, 0, 0])
        for point in cell_points:
            centroid = centroid + point.get_point_array()

        centroid = centroid / 4
        return Point(x=centroid[0], y=centroid[1], z=centroid[2])

    def divide(self):
        """Mechanism for cell to divide.

        - First checks if the cell depth has been reached.
        - Generates a new point at the centroid of the cell. Four new edges are created from the centroid and the
          points of the tet.
        - Loop over the faces of the parent cell and generate a new cell for each parent face.
        - To do this, new internal faces must be generated using the four new edges and the edges of the parent faces.
        - Loop over the edge of the parent cell faces and check if the edge has been used by another parent face to
          create a new face, if so, use the previously created face.
        - If the edge has not been used for generating a new face, loop over the newly created internal edges and find
          the edges whose point match the parent face edge.
        - Once two matching new edges have been found, use the parent face edge and the matching new edges to generate a
          new face.

        Returns:
            Four new cell objects
        """
        new_cells = []
        new_edges = []
        shared_face_edges = {}

        points = self.points

        if self.depth < self.division_depth:
            centroid = self.get_centroid(points)

            for point in points:
                new_line = Line(point, centroid)
                new_edges.append(Edge(self.material, new_line))

            for face in self.faces:
                new_cell_faces = []

                for face_edge in face.edges:
                    if face_edge in shared_face_edges:
                        new_cell_faces.append(shared_face_edges[face_edge])
                        continue
                    else:
                        matching_edges = []
                        for edge in new_edges:
                            for face_edge_point in face_edge.points:
                                if face_edge_point in edge.points:
                                    matching_edges.append(edge)
                                    break

                            if len(matching_edges) == 2:
                                break

                        new_face = Face(self.material, [face_edge, matching_edges[0], matching_edges[1]], thickness=0.0)
                        shared_face_edges[face_edge] = new_face
                        new_cell_faces.append(new_face)

                new_cells.append(Tet(self.material, face, new_cell_faces[0], new_cell_faces[1], new_cell_faces[2], depth=self.depth + 1))

        return new_cells

    def addition(self, signal):
        """Mechanism to add a new cell to the boundary of the cell.

        - Given a signal to add a new cell for each line of the cell.
        - Check which faces want to add a new cell on them.
        - Check if the faces are on the boundary of the organism.
        - Add a new point normal to the face outside the cell.
        - This new point will be places at a distance half of the length of the line.
        - Generate two new lines/edges connected to the new cell and the parent cell.

        Returns:
            New cells if face is on boundary.
        """
        new_cells = []
        cell_points = self.points

        for i, face in enumerate(self.faces):
            if signal[i] >= self.trigger_for_addition and face.on_boundary:

                centroid = face.centroid
                face_points = list(face.points.values())

                opposite_point = None
                for cell_point in cell_points:
                    if cell_point not in face_points:
                        opposite_point = cell_point

                vec = centroid - opposite_point.as_array
                point_pos = vec + centroid

                new_point = Point(x=point_pos[0], y=point_pos[1], z=point_pos[2])

                line_0 = Line(face_points[0], new_point)
                line_1 = Line(face_points[1], new_point)
                line_2 = Line(face_points[2], new_point)

                new_edges = []
                new_edges.append(Edge(self.material, line_0))
                new_edges.append(Edge(self.material, line_1))
                new_edges.append(Edge(self.material, line_2))

                new_faces = []

                for face_edge in face.edges:
                    matched_edges = []
                    for new_edge in new_edges:
                        for face_point in face_edge.points:
                            if face_point in new_edge.points:
                                matched_edges.append(new_edge)

                    new_face = Face(self.material, [face_edge, matched_edges[0], matched_edges[1]], thickness=0.0)
                    new_face.on_boundary = True
                    face.on_boundary = False
                    new_faces.append(new_face)

                new_cells.append(Tet(self.material, face, new_faces[0], new_faces[1], new_faces[2], depth=self.depth + 1))

        return new_cells

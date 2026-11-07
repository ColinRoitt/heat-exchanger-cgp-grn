"""Module for holding all stimulus classes."""
from typing import Type
from abc import ABC, abstractmethod
import numpy as np
from src.devo.geometry import Plane, Point
from src.devo.topology import Edge, Face
from src.utils.errors import SupportPointError

class Stimulus(ABC):
    """Abstract class for all stimuli.

    Class Attributes:
        __instance_count: An int which keeps track of the number of stimulus objects or child
        objects that have been instantiated.

    Attributes:
        id: Unique id of Stimulus object or its children.
        application_type: String of the type of stimulus.
        group_name: Unique identifer in the inp deck.
    """

    __instance_count = 0

    @classmethod
    def increment_instance_count(cls):
        """Increments the instance count for the Stimulus class and child classes."""
        cls.__instance_count += 1

    @classmethod
    def reset_instance_count(cls):
        """Resets the instance count for the Stimulus class and child classes."""
        cls.__instance_count = 0

    def __init__(self):
        """Inits the Stimulus object."""
        self.id = self.__instance_count
        Stimulus.increment_instance_count()
        self.application_type = None
        self._group_name = None

    @property
    def group_name(self):
        """Gets the group name for the inp deck.

        Returns:
            String holding the group name.
        """
        self._group_name = f"{self.application_type}_{self.id}"
        return self._group_name


class BoundaryCondition(Stimulus):
    """Abstract class for boundary condition objects."""
    @abstractmethod
    def get_inp_deck(self):
        """Get section for inp deck.

        Returns:
            String for inp deck section.
        """
        pass

    @abstractmethod
    def constrain_coord_delta(self, coord_delta):
        """Constrain GRN coordinate delta signal based on the stimulus type.

        Arg:
            coord_delta: Three element array with a delta value for each coordinate.

        Returns:
            Constrained coordinate delta.
        """
        pass


class InitialConditions(Stimulus):
    """Abstract class for initial condition objects."""
    @abstractmethod
    def get_inp_deck(self):
        """Get section for inp deck

        Returns:
            string for inp deck section
        """


class PlanarDisplacement(BoundaryCondition):
    """Displacement boundary condition class.

    Attributes:
        plane: plane geometry through which the displacement acts
        magnitude: magnitude of the displacement
    """
    def __init__(self, plane: Type[Plane], magnitude):
        super().__init__()
        self.plane = plane
        self.magnitude = float(magnitude)
        self.application_type = 'planar_displacement'

    def __repr__(self):
        """Defines the printable representation of the Support object."""
        return repr(f"Displacement boundary condition, id {self.id} at plane {self.plane}\
                    with direction {self.plane.direction}")

    def is_point_on_plane(self, point):
        """Checks if point is on the displacement plane.

        Args:
            point: Point object.

        Returns:
            Boolean dictating if the point is on the displacement plane.
        """
        return self.plane.is_point_on_plane(point)

    def get_inp_deck(self):
        """Get section for inp deck.

        Returns:
            String for inp deck section.
        """
        return f'*BOUNDARY\n{self.group_name}, 1, 1, {self.plane.direction[0]}\n\
            {self.group_name}, 2, 2, {self.plane.direction[1]}\n\
            {self.group_name}, 3, 3, {self.plane.direction[2]}\n'

    def constrain_coord_delta(self, coord_delta):
        """Constrain GRN coordinate delta signal based on the stimulus type.

        Arg:
            coord_delta: Three element array with a delta value for each coordinate.

        Returns:
            Constrained coordinate delta.
        """
        return np.array([0.0, 0.0, 0.0])


class Support(BoundaryCondition):
    """Parent class for support objects."""
    def __init__(self, pos_x, pos_y, pos_z):
        super().__init__()
        self.location = [float(pos_x), float(pos_y), float(pos_z)]
        self.application_type = None
        self.num_reactions = None

    def __repr__(self):
        """Defines the printable representation of the Support object."""
        return repr(f"{self.application_type} support id {self.id} at location {self.location}.")


    def get_point_at_support(self, points, as_array=False):
        """Checks if point is at the support stimulus.

        Args:
            point: Points dictionary.

        Returns:
            Point object at the support stimulus.
        """
        support_point = []
        for id, point in points.items():
            if as_array:
                if np.array_equal(np.array(self.location), point):
                    support_point.append(point)
            else:
                if np.array_equal(np.array(self.location), point.as_array):
                    support_point.append(point)

        if len(support_point) == 0:
            raise SupportPointError("Error in support point location: supports must be located on a known point in the organism seedling - please change pos_x, pos_y and pos_z accordingly.")
        else:
            return support_point
        
    def is_point_at_support(self, point, as_array=False):
            """Checks if point is at the support stimulus.

            Args:
                point: Point object.

            Returns:
                Boolean dictating if the point is at the support stimulus.
            """
            if as_array:
                return np.array_equal(np.array(self.location), point)
            else:
                return np.array_equal(np.array(self.location), point.as_array)

    def constrain_coord_delta(self, coord_delta):
        """Constrain GRN coordinate delta signal based on the stimulus type.

        Arg:
            coord_delta: Three element array with a delta value for each coordinate.

        Returns:
            Constrained coordinate delta.
        """
        return np.array([0.0, 0.0, 0.0])


class TemperaturePlanar(BoundaryCondition):
    """Temperature boundary condition class.

    Attributes:
    plane: plane geometry where the temperature acts
    magnitude: temperature magnitude
    application_type: Application type of stimulus: point or planar.

    """
    def __init__(self, plane: Type[Plane], magnitude: Type[float]):
        super().__init__()
        self.plane = plane
        self.magnitude = float(magnitude)
        self.application_type = "temperature_planar"

    def get_inp_deck(self):
        """Get section for inp deck.

        Returns:
            String for inp deck section.
        """
        return f'*Boundary\n{self.group_name}, 11, 11, {self.magnitude}\n'

    def constrain_coord_delta(self, coord_delta):
        """Constrain GRN coordinate delta signal based on the stimulus type.

        Arg:
            coord_delta: Three element array with a delta value for each coordinate.

        Returns:
            Constrained coordinate delta.
        """
        return np.array([0.0, 0.0, 0.0])

    def is_edge_on_plane(self, edges):
        """Checks if edge on plane

        Args:
            edges: Edge dictionary object

        Returns:
            Boolean describing edge on the plane
        """
        entities = []
        for id, edge in edges.items():
            line_vec = edge.line.direction/np.linalg.norm(edge.line.direction)
            dot_prod = np.dot(self.plane.direction, line_vec)
            if dot_prod == 0.0 and self.is_point_on_plane(edge.midpoint):
                entities.append(edge)

        return entities

    def is_point_on_plane(self, point):
        """Checks if point is on the loading plane.

        Args:
            point: Point object.

        Returns:
            Boolean dictating if the point is on the loading plane.
        """
        return self.plane.is_point_on_plane(point)

    def is_normal_coincident(self, face):
        """
        Checks if face normal and plane normal are coincident
        """
        face_norm = face.normal
        plane_norm = self.plane.direction / np.linalg.norm(self.plane.direction)
        dot = np.dot(face_norm, plane_norm)
        if abs(dot) == 1:
            return True
        return False

    def is_face_on_plane(self, face):
        """Checks if face is on the plane.

        Args:
            face: Face object.

        Returns:
            Boolean dictating if the face is on the plane.
        """
        if self.is_point_on_plane(face.centroid) and self.is_normal_coincident(face):
            return True


class TemperaturePoint(BoundaryCondition):
    """Temperature boundary condition class.

    Attributes:
    point: point geometry where the temperature acts
    magnitude: temperature magnitude
    application_type: Application type of stimulus: point or planar.

    """
    def __init__(self, point: Type[Point], magnitude: Type[float]):
        super().__init__()
        self.point = point
        self.magnitude = float(magnitude)
        self.application_type = "temperature_point"

    def get_inp_deck(self):
        """Get section for inp deck.

        Returns:
            String for inp deck section.
        """
        return f'*Boundary\n{self.group_name}, 11, 11, {self.magnitude}\n'

    def constrain_coord_delta(self, coord_delta):
        """Constrain GRN coordinate delta signal based on the stimulus type.

        Arg:
            coord_delta: Three element array with a delta value for each coordinate.

        Returns:
            Constrained coordinate delta.
        """
        return np.array([0.0, 0.0, 0.0])


    def get_nearest_point_on_bc(self, points):
        """Checks if point is on the stimulus point.

        Args:
            points: Points dictionary.

        Returns:
            Nearest point object

        """
        nearest_pt = None
        this_distance = None
        for pt in points.values():
            distance = np.linalg.norm(np.subtract(self.point.as_array, pt.as_array))
            if this_distance is None:
                this_distance = distance
                nearest_pt = pt
            elif distance < this_distance:
                this_distance = distance
                nearest_pt = pt
                
        return nearest_pt


class FixedSupportPlanar(Support):
    """Fixed support class with three reaction.

    Attributes:
        plane: plane geometry where the support acts
        application_type: Application type of stimulus: point or planar.

    """
    def __init__(self, plane: Type[Plane]):
        super().__init__(plane.point.x, plane.point.y, plane.point.z)
        self.plane = plane
        self.application_type = "planar_support_fixed"

    def is_point_on_plane(self, point):
        """Checks if point is on the fixed plane.

        Args:
            point: Point object.

        Returns:
            Boolean dictating if the point is on the displacement plane.
        """
        return self.plane.is_point_on_plane(point)

    def is_edge_on_plane(self, edge):
        """Checks if edge on plane

        Args:
            edge: Edge object

        Returns:
            Boolean describing edge on the plane
        """
        line_vec = edge.line.direction/np.linalg.norm(edge.line.direction)
        dot_prod = np.dot(self.plane.direction, line_vec)
        if dot_prod == 0.0 and self.is_point_on_plane(edge.midpoint):
            return True

    def get_inp_deck(self, entity=Type[str]):
        """Get section for inp deck.

        Returns:
            String for inp deck section.
        """
        if entity == 'edge':
            return f'*BOUNDARY\n{self.group_name}, 1, 3\n'
        elif entity == 'face':
            return f'*BOUNDARY\n{self.group_name}, 1, 6\n'


class FixedSupport(Support):
    """Fixed support class with three reaction.

    Attributes:
        pos_x: X coordinate of the position of the support.
        pos_y: Y coordinate of the position of the support.
        pos_z: Z coordinate of the position of the support.
        application_type: str of support type
        num_reactions: Number of reactions induced by the support.
    """
    def __init__(self, pos_x, pos_y, pos_z):
        super().__init__(pos_x, pos_y, pos_z)
        self.application_type = "fixed_support"
        self.num_reactions = 3

    def get_inp_deck(self):
        """Get section for inp deck.

        Returns:
            String for inp deck section.
        """
        return f'*BOUNDARY\n{self.group_name}, 1, 3\n'


class PinnedSupport(Support):
    """Pinned support class with two reactions.

    Attributes:
        pos_x: X coordinate of the position of the support.
        pos_y: Y coordinate of the position of the support.
        pos_z: Z coordinate of the position of the support.
        application_type: Type of stimulus.
        num_reactions: Number of reactions induced by the support.
    """
    def __init__(self, pos_x, pos_y, pos_z):
        super().__init__(pos_x, pos_y, pos_z)
        self.application_type = "pinned_support"
        self.num_reactions = 2

    def get_inp_deck(self):
        return f'*BOUNDARY\n{self.group_name}, 1, 2\n'


class RollerSupport(Support):
    """Roller support class with one reaction.

    Attributes:
        pos_x: X coordinate of the position of the support.
        pos_y: Y coordinate of the position of the support.
        pos_z: Z coordinate of the position of the support.
        application_type: Type of stimulus.
        num_reactions: Number of reactions induced by the support.
    """
    def __init__(self, pos_x, pos_y, pos_z):
        super().__init__(pos_x, pos_y, pos_z)
        self.application_type = "roller_support"
        self.num_reactions = 1

    def get_inp_deck(self):
        """Get section for inp deck.

        Returns:
            String for inp deck section.
        """
        return f'*BOUNDARY\n{self.group_name}, 2\n'


class SlidingSupport(Support):
    """Sliding support class with two reactions.

    Attributes:
        pos_x: X coordinate of the position of the support.
        pos_y: Y coordinate of the position of the support.
        pos_z: Z coordinate of the position of the support.
        application_type: Type of stimulus.
        num_reactions: Number of reactions induced by the support.
    """
    def __init__(self, pos_x, pos_y, pos_z):
        super().__init__(pos_x, pos_y, pos_z)
        self.application_type = "sliding_support"
        self.num_reactions = 2

    def get_inp_deck(self):
        """Get section for inp deck.

        Returns:
            String for inp deck section.
        """
        return f'*BOUNDARY\n{self.group_name}, 1\n{self.group_name}, 3\n'


class TranslationBC(Support):
    """Support class that restricts translation specified degree of freedom

    Attributes:
        plane: plane on which the edge/face lies
        dof_x: x degree of freedom, default 1 is open
        dof_y: y degree of freedom, default 1 is open
        dof_z: z degree of freedom default 1 is open
        application_type: Type of stimulus.
    """
    def __init__(self, plane, application: str, dof_x=None, dof_y=None, dof_z=None, boundary=False):
        self.plane = plane
        pos_x = self.plane.point.x
        pos_y = self.plane.point.y
        pos_z = self.plane.point.z
        super().__init__(pos_x, pos_y, pos_z)
        self.U1 = dof_x
        self.U2 = dof_y
        self.U3 = dof_z
        self.application_type = f'planar_{application}_translation_bc'
        self.boundary = boundary

    def __repr__(self):
        """Defines the printable representation of the Support object."""
        return repr(f"{self.application_type} support id {self.id} with {self.U1, self.U2, self.U3} translations constrained.")        

    def get_inp_deck(self):
        """Get section for inp deck.

        Returns:
            String for inp deck section.
        """
        inp_deck = f'*BOUNDARY\n'
        if self.U1 == 0:
            inp_deck += f'{self.group_name}, 1\n'
        if self.U2 == 0:
            inp_deck += f'{self.group_name}, 2\n'
        if self.U3 == 0:
            inp_deck += f'{self.group_name}, 3\n'

        return inp_deck

    def is_point_on_plane(self, point):
        """Checks if point is on the loading plane.

        Args:
            point: Point object.

        Returns:
            Boolean dictating if the point is on the loading plane.
        """
        return self.plane.is_point_on_plane(point)
    
    def is_edge_on_plane(self, edge):
        """Checks if edge on plane

        Args:
            edge: Edge object

        Returns:
            Boolean describing edge on the plane
        """
        line_vec = edge.line.direction / np.linalg.norm(edge.line.direction)
        dot_prod = np.dot(self.plane.direction, line_vec)
        if dot_prod == 0.0 and self.is_point_on_plane(edge.midpoint):
            if self.boundary:
                if edge.on_boundary:
                    return True
                else:
                    return False
            else:
                return True    


class RotationBC(Support):
    """Support class that restricts translation specified degree of freedom

    Attributes:
        plane: plane on which the edge/face lies
        dof_x: x degree of freedom, default 1 is open
        dof_y: y degree of freedom, default 1 is open
        dof_z: z degree of freedom default 1 is open
        application_type: Type of stimulus.
    """
    def __init__(self, plane, application: str, dof_x=None, dof_y=None, dof_z=None, boundary=False):
        self.plane = plane
        pos_x = self.plane.point.x
        pos_y = self.plane.point.y
        pos_z = self.plane.point.z
        super().__init__(pos_x, pos_y, pos_z)
        self.UR1 = dof_x
        self.UR2 = dof_y
        self.UR3 = dof_z
        self.application_type = f'planar_{application}_rotation_bc'
        self.boundary = boundary

    def __repr__(self):
        """Defines the printable representation of the Support object."""
        return repr(f"{self.application_type} support id {self.id} with {self.UR1, self.UR2, self.UR3} rotations constrained.")        

    def get_inp_deck(self):
        """Get section for inp deck.

        Returns:
            String for inp deck section.
        """
        inp_deck = f'*BOUNDARY\n'
        if self.UR1 == 0:
            inp_deck += f'{self.group_name}, 4\n'
        if self.UR2 == 0:
            inp_deck += f'{self.group_name}, 5\n'
        if self.UR3 == 0:
            inp_deck += f'{self.group_name}, 6\n'

        return inp_deck

    def is_point_on_plane(self, point):
        """Checks if point is on the loading plane.

        Args:
            point: Point object.

        Returns:
            Boolean dictating if the point is on the loading plane.
        """
        return self.plane.is_point_on_plane(point)

    def is_edge_on_plane(self, edge):
        """Checks if edge on plane

        Args:
            edge: Edge object

        Returns:
            Boolean describing edge on the plane
        """
        line_vec = edge.line.direction / np.linalg.norm(edge.line.direction)
        dot_prod = np.dot(self.plane.direction, line_vec)
        if dot_prod == 0.0 and self.is_point_on_plane(edge.midpoint):
            if self.boundary:
                if edge.on_boundary:
                    return True
                else:
                    return False
            else:
                return True               


class EncastreBC(Support):
    """Support class that restricts translation & rotations in all degree of freedom

    Attributes:
        point: point at which the stimulus acts
        application_type: Type of stimulus.
    """
    def __init__(self, point, application: str, boundary=False):
        self.point = point
        pos_x = self.point.x
        pos_y = self.point.y
        pos_z = self.point.z
        super().__init__(pos_x, pos_y, pos_z)
        self.application_type = f'{application}_encastre_bc'
        self.boundary = boundary
    
    def __repr__(self):
        """Defines the printable representation of the Support object."""
        return repr(f"{self.application_type} support id {self.id}.")        

    def get_inp_deck(self):
        """Get section for inp deck.

        Returns:
            String for inp deck section.
        """
        inp_deck = f'*BOUNDARY\n'
        inp_deck += f'{self.group_name}, 1, 6\n'

        return inp_deck


class Constraint(Stimulus):
    """Abstract class for constraint objects.

    - This class should be used to inherit from when you are looking to create a stimuli that
    constrains an entity in the organism.
    - An example would be to prevent any point in the organism moving in the Z-axis.
    """
    @abstractmethod
    def constrain_coord_delta(self, coord_delta):
        """Constrain GRN coordinate delta signal based on the stimulus type.

        Arg:
            coord_delta: Three element array with a delta value for each coordinate.

        Returns:
            Constrained coordinate delta.
        """


class TwoDimConstraint(Constraint):
    """Stimulus for constraining the environmental element to 2D."""
    def __init__(self):
        super().__init__()
        self.application_type = "2D Constraint"

    def __repr__(self):
        """Defines the printable representation of the TwoDimConstraint object."""
        return repr(f"2D constraint id {self.id}.")

    def get_inp_deck(self):
        """Get section for inp deck.

        Returns:
            String for inp deck section.
        """
        return '*BOUNDARY\nALLNODES, 3\n'

    def constrain_coord_delta(self, coord_delta):
        """Constrain GRN coordinate delta signal based on the stimulus type.

        Arg:
            coord_delta: Three element array with a delta value for each coordinate.

        Returns:
            Constrained coordinate delta.
        """
        coord_delta[2] = 0.0
        return coord_delta


class PointConstraint(Constraint):
    """Stimulus for constraining a point in a given location.

    Attributes:
        pos_x: X coordinate of the position of the constraint.
        pos_y: Y coordinate of the position of the constraint.
        pos_z: Z coordinate of the position of the constraint.
        location: Location of constraint.
    """
    def __init__(self, pos_x, pos_y, pos_z):
        super().__init__()
        self.location = [float(pos_x), float(pos_y), float(pos_z)]
        self.application_type = "point_constraint"

    def __repr__(self):
        """Defines the printable representation of the PointConstraint object."""
        return repr(f"Point constraint id {self.id}.")

    def constrain_coord_delta(self, coord_delta):
        """Constrain GRN coordinate delta signal based on the stimulus type.

        Arg:
            coord_delta: Three element array with a delta value for each coordinate.

        Returns:
            Constrained coordinate delta.
        """
        return np.array([0.0, 0.0, 0.0])

    def is_point_at_constraint(self, point, as_array=False):
        """Checks if point is at the support stimulus.

        Args:
            point: Point object.

        Returns:
            Boolean dictating if the point is at the constraint stimulus.
        """
        if as_array:
            return np.array_equal(np.array(self.location), point)
        else:
            return np.array_equal(np.array(self.location), point.get_point_array())


class PlanarConstraint(Constraint):
    """Stimulus for constraining a point on a given plane.

    Attributes:
        plane: plane geomtry on which the constraint acts
        normal: numpy array of normal vector of plane
    """

    def __init__(self, plane):
        super().__init__()
        self.plane = plane
        self.application_type = "planar_constraint"

    def __repr__(self):
        """Defines the printable representation of the PointConstraint object."""
        return repr(f"Planar constraint id {self.id}.")

    def constrain_coord_delta(self, coord_delta):
        """Constrain GRN coordinate delta signal based on the stimulus type.

        Arg:
            coord_delta: Three element array with a delta value for each coordinate.

        Returns:
            Constrained coordinate delta.
        """
        return np.array([0.0, 0.0, 0.0])

    def is_point_at_constraint(self, point):
        """Checks if point is at the constraint stimulus.

        Args:
            point: Point object.

        Returns:
            Boolean dictating if the point is at the constraint stimulus.
        """
        return self.plane.is_point_on_plane(point.as_array)


class Load(Stimulus):
    """Abstract class for all load objects.

    Attributes:
        magnitude: Magnitude of the load (absolute value with no direction).
        pos_x: X coordinate of the position of the load.
        pos_y: Y coordinate of the position of the load.
        pos_z: Z coordinate of the position of the load.
        dir_x: Direction of the load in the X axis.
        dir_y: Direction of the load in the Y axis.
        dir_z: Direction of the load in the Z axis.
        direction: Coordinate of the direction of the load.
    """
    def __init__(self, magnitude, dir_x, dir_y, dir_z):
        self.magnitude = float(magnitude)
        self.dir_x = float(dir_x)
        self.dir_y = float(dir_y)
        self.dir_z = float(dir_z)
        self.direction = [dir_x, dir_y, dir_z]
        super().__init__()

    @abstractmethod
    def get_inp_deck(self):
        """Get section for inp deck.

        Returns:
            String for inp deck section.
        """

    @abstractmethod
    def constrain_coord_delta(self, coord_delta):
        """Constrain GRN coordinate delta signal based on the stimulus type.

        Arg:
            coord_delta: Three element array with a delta value for each coordinate.

        Returns:
            Constrained coordinate delta.
        """


class PointLoad(Load):
    """Class for a load acting through a single point in space.

    Attributes:
        magnitude: Magnitude of the load (absolute value with no direction).
        pos_x: X coordinate of the position of the load.
        pos_y: Y coordinate of the position of the load.
        pos_z: Z coordinate of the position of the load.
        dir_x: Direction of the load in the X axis.
        dir_y: Direction of the load in the Y axis.
        dir_z: Direction of the load in the Z axis.
        location: Coordinate of where the load is located.
        application_type: The type of the stimulus.
    """
    def __init__(self, magnitude, pos_x, pos_y, pos_z, dir_x, dir_y, dir_z):
        super().__init__(magnitude, dir_x, dir_y, dir_z)
        self.location = [float(pos_x), float(pos_y), float(pos_z)]
        self.application_type = "point_load"

    def __repr__(self):
        """Defines the printable representation of the PointLoad object."""
        return repr(f"Point load id {self.id} of magnitude {self.magnitude}N at location\
                    {self.location} in direction {self.direction}")

    def is_point_at_load(self, point, as_array=False):
        """Checks if point is at the point load stimulus.

        Args:
            point: Point object.

        Returns:
            Boolean dictating if the point is at the point load stimulus.
        """
        if as_array:
            return np.array_equal(np.array(self.location), point)
        else:
            return np.array_equal(np.array(self.location), point.as_array)

    def get_inp_deck(self):
        """Get section for inp deck.

        Returns:
            String for inp deck section.
        """
        sign = None
        inp_deck = ''
    
        if self.dir_x != 0 or self.dir_y != 0.0 or self.dir_z != 0.0:
            mag_comp = self.magnitude / (abs(self.dir_x) + abs(self.dir_y) + abs(self.dir_z))
            if self.dir_x != 0.0:
                inp_deck += f'*CLOAD\n{self.group_name}, '
                inp_deck += f'1, {self.dir_x * mag_comp}\n'
            if self.dir_y != 0.0:
                inp_deck += f'*CLOAD\n{self.group_name}, '
                inp_deck += f'2, {self.dir_y * mag_comp}\n'
            if self.dir_z != 0.0:          
                inp_deck += f'*CLOAD\n{self.group_name}, '
                inp_deck += f'3, {self.dir_z * mag_comp}\n'
        else:
            inp_deck = f'*CLOAD\n{self.group_name}, '
            if self.dir_x != 0:
                inp_deck += '1, '
                sign = np.sign(self.dir_x)
            elif self.dir_y != 0:
                inp_deck += '2, '
                sign = np.sign(self.dir_y)
            elif self.dir_z != 0:
                inp_deck += '3, '
                sign = np.sign(self.dir_z)
            else:
                raise ValueError(f"Direction in {self} must not be [0, 0, 0]")

            if sign == 1:
                inp_deck += f'{self.magnitude}\n'
            elif sign == -1:
                inp_deck += f'-{self.magnitude}\n'
            else:
                raise ValueError

        return inp_deck

    def constrain_coord_delta(self, coord_delta):
        """Constrain GRN coordinate delta signal based on the stimulus type.

        Arg:
            coord_delta: Three element array with a delta value for each coordinate.

        Returns:
            Constrained coordinate delta.
        """
        return np.array([0.0, 0.0, 0.0])


class PlanarDistributedLoad(Load):
    """Class for a distributed load acting through a planar face in space.

    Attributes:
        magnitude: Magnitude of the load (+ve or -ve to indicate direction).
        plane: plane through which the load acts
        application_type: The type of the stimulus.
    """

    def __init__(self, plane: Type[Plane], magnitude=Type[float], application=Type[str], all_elements=False, boundary=False):
        super().__init__(magnitude, plane.direction[0], plane.direction[1], plane.direction[2])
        self.plane = plane
        if application == "face":
            self.application_type = "planar_face_DLOAD"
        elif application == "edge":
            self.application_type = "planar_edge_DLOAD"
        self.magnitude = float(magnitude)
        self.all_elements = all_elements
        self.boundary = boundary

    def __repr__(self):
        """Defines the printable representation of the DistributedLoad object."""
        return repr(f"Distributed load id {self.id} of magnitude {self.magnitude} at location {self.plane}")

    def get_inp_deck(self, ednor=None):
        """Get section for inp deck.

        Returns:
            String for inp deck section.
        """
        if self.all_elements:
            group_name = "eplate"
            inp_deck =  f"*DLOAD\n{group_name}, P, {self.magnitude}\n"
        elif self.application_type == "planar_edge_DLOAD":
            magnitude = self.magnitude
            group_name = self.group_name
            if ednor is False:
                inp_deck = f'*DLOAD\n{group_name}, EDNOR1, {magnitude}\n'
            else:
                inp_deck = f'*DLOAD\n{group_name}, EDNOR{ednor}, {magnitude}\n'
        else:
            group_name = self.group_name
            
            inp_deck =  f"*DLOAD\n{group_name}, P, {self.magnitude}\n"
        
        return inp_deck
    
    def constrain_coord_delta(self, coord_delta):
        """Constrain GRN coordinate delta signal based on the stimulus type.

        Arg:
            coord_delta: Three element array with a delta value for each coordinate.

        Returns:
            Constrained coordinate delta.
        """
        return np.array([0.0, 0.0, 0.0])

    def is_point_on_plane(self, point):
        """Checks if point is on the loading plane.

        Args:
            point: Point object.

        Returns:
            Boolean dictating if the point is on the loading plane.
        """
        return self.plane.is_point_on_plane(point)

    def is_normal_coincident(self, face):
        """
        Checks if face normal and plane normal are coincident
        """
        face_norm = face.normal
        plane_norm = self.plane.direction / np.linalg.norm(self.plane.direction)
        dot = np.dot(face_norm, plane_norm)
        if abs(dot) == 1:
            return True
        return False

    def is_face_on_plane(self, face):
        """Checks if face is on the plane.

        Args:
            face: Face object.

        Returns:
            Boolean dictating if the face is on the plane.
        """
        if self.is_point_on_plane(face.centroid) and self.is_normal_coincident(face):
            return True

    def is_edge_on_plane(self, edge):
        """Checks if edge on plane

        Args:
            edge: Edge object

        Returns:
            Boolean describing edge on the plane
        """
        line_vec = edge.line.direction / np.linalg.norm(edge.line.direction)
        dot_prod = np.dot(self.plane.direction, line_vec)
        if dot_prod == 0.0 and self.is_point_on_plane(edge.midpoint):
            if self.boundary:
                if edge.on_boundary:
                    return True
                else:
                    return False
            else:
                return True


class PlanarConcentratedLoad(Load):
    """Class for a distributed load acting through a planar face in space.

    Attributes:
        magnitude: Magnitude of the load (+ve or -ve to indicate direction).
        plane: plane through which the load acts
        application_type: The type of the stimulus.
    """

    def __init__(self, plane: Type[Plane], magnitude, application):
        super().__init__(magnitude, plane.direction[0], plane.direction[1], plane.direction[2])
        self.plane = plane
        if application == "face":
            self.application_type = "planar_face_CLOAD"
        elif application == "edge":
            self.application_type = "planar_edge_CLOAD"

    def __repr__(self):
        """Defines the printable representation of the ConcentratedLoad object."""
        return repr(f"Concentrated load id {self.id} of magnitude {self.magnitude}\
                    at location {self.plane}")

    def get_inp_deck(self):
        """Get section for inp deck.

        Returns:
            String for inp deck section.
        """
        sign = None
        inp_deck = f'*CLOAD\n{self.group_name}, '

        if self.plane.direction[0] != 0.0:
            inp_deck += '1, '
            sign = np.sign(self.plane.direction[0])
        elif self.plane.direction[1] != 0.0:
            inp_deck += '2, '
            sign = np.sign(self.plane.direction[1])
        elif self.plane.direction[2] != 0.0:
            inp_deck += '3, '
            sign = np.sign(self.plane.direction[2])
        else:
            raise ValueError(f"Direction in {self} must not be [0, 0, 0]")

        if sign == 1:
            inp_deck += f'{self.magnitude}\n'
        elif sign == -1:
            inp_deck += f'-{self.magnitude}\n'
        else:
            raise ValueError

        return inp_deck

    def is_edge_on_plane(self, edge):
        """Checks if edge on plane

        Args:
            edge: Edge object

        Returns:
            Boolean describing edge on the plane
        """
        line_vec = edge.line.direction / np.linalg.norm(edge.line.direction)
        dot_prod = np.dot(self.plane.direction, line_vec)
        if dot_prod == 0.0 and self.is_point_on_plane(edge.midpoint):
            return True

    def is_face_on_plane(self, face):
        """Checks if face is on the plane.

        Args:
            face: Face object.

        Returns:
            Boolean dictating if the face is on the plane.
        """
        if self.is_point_on_plane(face.centroid) and self.is_normal_coincident(face):
            return True

    def constrain_coord_delta(self, coord_delta):
        """Constrain GRN coordinate delta signal based on the stimulus type.

        Arg:
            coord_delta: Three element array with a delta value for each coordinate.

        Returns:
            Constrained coordinate delta.
        """
        return np.array([0.0, 0.0, 0.0])

    def is_point_on_plane(self, point):
        """Checks if point is on the loading plane.

        Args:
            point: Point object.

        Returns:
            Boolean dictating if the point is on the loading plane.
        """
        return self.plane.is_point_on_plane(point)

    def is_normal_coincident(self, face):
        """
        Checks if face normal and plane normal are coincident
        """
        face_norm = face.normal
        plane_norm = self.plane.direction / np.linalg.norm(self.plane.direction)
        dot = np.dot(face_norm, plane_norm)
        if abs(dot) == 1:
            return True
        else:
            return False


class Initialisation(Stimulus):
    """Abstract class for initialisation objects.

    - Inherited classes should be used when you want influence the initialisation of the seedling
    but not the environment in regards to analysis.
    """
    def constrain_coord_delta(self, coord_delta):
        """Constrain GRN coordinate delta signal based on the stimulus type.

        Arg:
            coord_delta: Three element array with a delta value for each coordinate.

        Returns:
            Constrained coordinate delta.
        """
        return coord_delta


class PointInit(Initialisation):
    """Initialisation point used to influenence the generation of the seedling.

     Attributes:
        pos_x: X coordinate of the position of the point.
        pos_y: Y coordinate of the position of the point.
        pos_z: Z coordinate of the position of the point.
    """
    def __init__(self, pos_x, pos_y, pos_z):
        super().__init__()
        self.location = [float(pos_x), float(pos_y), float(pos_z)]
        self.application_type = "initialisation_point"

    def __repr__(self):
        """Defines the printable representation of the PointInit object."""
        return repr(f"Initialisation point id {self.id} at location {self.location}")


class InitialTemperature(InitialConditions):
    """Class for initial conditions during analysis

    Attributes:
        magnitude: magnitude of temperature
    """

    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        self.application_type = "initial_conditions_temp"

    def get_inp_deck(self):
        """Get section for inp deck.

        Returns:
            String for inp deck section.
        """
        inp_deck = f'*INITIAL CONDITIONS, TYPE=TEMPERATURE\nALLNODES, {self.temperature}\n'

        return inp_deck


class Temperature(Stimulus):
    """Class for temperature object for thermal analysis

    Attributes:
        temperature: magnitude of temperature
    """
    def __init__(self, temperature, all_nodes=False, location=None):
        super().__init__()
        self.temperature = temperature
        self.application_type = "temperature"
        self._file_name = None
        self._on_all_nodes = all_nodes
        self.location = location

    def get_inp_deck(self):
        """Get section for inp deck.

        Returns:
            String for inp deck section.
        """
        if self._on_all_nodes:
            inp_deck = f'*TEMPERATURE\nALLNODES, {self.temperature}\n'
        elif self._file_name is not None:
            inp_deck = f'*TEMPERATURE, FILE={self._file_name}.frd\n'
        else:
            inp_deck = f'*TEMPERATURE\n{self.group_name}, {self.temperature}\n'

        return inp_deck

    def get_nearest_entity(self, entities):
        """Return the entity nearest to the location of the stimulus

        Args:
            entities: list of entities - points, edges, faces of the organism

        Return:
            nearest_entity: instance of the nearest entity
        """
        distance = 0
        nearest_entity = None

        for ind, entity in enumerate(entities):
            if isinstance(entity, Point):
                #Loop through and get distance between point and location
                this_distance = np.linalg.norm(np.subtract(entity.get_point_array(),
                                                           self.location.get_point_array()))
                if ind == 0:
                    distance = this_distance
                    nearest_entity = entity
                else:
                    if this_distance < distance:
                        nearest_entity = entity
                        distance = this_distance
            elif isinstance(entity, Edge):
                #Loop through and get distance between midpoint and location
                this_distance = np.linalg.norm(np.subtract(entity.midpoint,
                                                           self.location.get_point_array()))
                if ind == 0:
                    distance = this_distance
                    nearest_entity = entity
                else:
                    if this_distance < distance:
                        nearest_entity = entity
                        distance = this_distance
            elif isinstance(entity, Face):
                #Loop through and get distance between centroid and location
                this_distance = np.linalg.norm(np.subtract(entity.centroid,
                                                           self.location.get_point_array()))
                if ind == 0:
                    distance = this_distance
                    nearest_entity = entity
                else:
                    if this_distance < distance:
                        distance = this_distance
                        nearest_entity = entity

        return nearest_entity


class FilmCondition(Stimulus):
    """Class to defin film condition in thermal analysis simulations
    """

    def __init__(self, film_coefficient, ambient_temperature,
                 pos_x=None, pos_y=None, pos_z=None, all_elements=False):
        super().__init__()
        self._on_all_elements = all_elements
        self.coefficient = film_coefficient
        self.ambient_temp = ambient_temperature
        self.application_type = 'film_condition'

        if all_elements is False:
            self._location = np.array([pos_x, pos_y, pos_z])

    def get_inp_deck(self, fins=False):
        """Get section for inp deck.

        Returns:
            String for inp deck section.
        """
        inp_deck = ''
        if self._on_all_elements:
            inp_deck += f'*FILM\nALLELEMENTS, F2, {self.ambient_temp}, {self.coefficient}\n'
        if fins:
            inp_deck += f'*FILM\n{self.group_name}, F3, {self.ambient_temp}, {self.coefficient}\n'

        return inp_deck
    
    def get_nearest_point_on_bc(self, points):
        """Checks if point is on the stimulus point.

        Args:
            points: Points dictionary.

        Returns:
            Nearest point object

        """
        nearest_pt = None
        this_distance = None
        for pt in points.values():
            distance = np.linalg.norm(np.subtract(self._location, pt.as_array))
            if this_distance is None:
                this_distance = distance
                nearest_pt = pt
            elif distance < this_distance:
                this_distance = distance
                nearest_pt = pt
                
        return nearest_pt  
    

class FilmConditionEdge(Stimulus):
    """Class to define film condition in thermal analysis simulations
    on organism edge
    """

    def __init__(self, plane, film_coefficient, ambient_temperature):
        super().__init__()
        self.coefficient = film_coefficient
        self.ambient_temp = ambient_temperature
        self.plane = plane
        self.application_type = 'film_condition'

    def get_inp_deck(self):
        """Get section for inp deck.

        Returns:
            String for inp deck section.
        """

        sign = None
        inp_deck = '*FILM\n'
        inp_deck += f'{self.group_name}, '

        if self.plane.direction[0] != 0:
            sign = np.sign(self.plane.direction[0])
        elif self.plane.direction[1] != 0:
            sign = np.sign(self.plane.direction[1])
            # if applying to shell
            #inp_deck += 'F4, '
            # if applying to beam
            inp_deck += 'F3, '
        elif self.plane.direction[2] != 0:
            sign = np.sign(self.plane.direction[2])
            if sign == 1:
                inp_deck += 'F1, '
            else:
                inp_deck += 'F2, '
        else:
            raise ValueError(f"Direction in {self} must not be [0, 0, 0]")

        inp_deck += f'{self.ambient_temp}, {self.coefficient}\n'

        return inp_deck

    def is_point_on_plane(self, point):
        """Checks if point is on the loading plane.

        Args:
            point: Point object.

        Returns:
            Boolean dictating if the point is on the loading plane.
        """
        return self.plane.is_point_on_plane(point)
    
    def is_edge_on_plane(self, edges):
        """Checks if edge on plane

        Args:
            edges: Edges dictionary

        Returns:
            Boolean describing edge on the plane
        """
        entities = []
        for id, edge in edges.items():
            line_vec = edge.line.direction/np.linalg.norm(edge.line.direction)
            dot_prod = np.dot(self.plane.direction, line_vec)
            if dot_prod == 0.0 and self.is_point_on_plane(edge.midpoint):
                entities.append(edge)

        return entities

    def get_nearest_edge(self, edges):
        for ind, edge in enumerate(edges):    
            midpoint = edge.line.midpoint
            this_distance = np.linalg.norm(np.subtract(midpoint, self.plane.point.as_array))
            if ind == 0:
                distance = this_distance
                closest_edge = edge
            else:
                if this_distance < distance:
                    closest_edge = edge
                    distance = this_distance

        return closest_edge

    
class DistributedHeatFlux(Stimulus):
    """Class for a distributed heat flux load acting through a planar face in space.

    Attributes:
        magnitude: Magnitude of the heat flux
    """

    def __init__(self, all_elements, plane, magnitude):
        super().__init__()
        self.magnitude = magnitude
        self._on_all_elements = all_elements
        self.plane = plane
        self.application_type = 'dist_heat_flux'

    def __repr__(self):
        """Defines the printable representation of the DistHeatFlux object."""
        return repr(f"Heat flux id {self.id} of magnitude {self.magnitude} at on all elements\
                    in direction {self.plane.direction}")


    def get_inp_deck(self):
        """Get section for inp deck.

        Returns:
            String for inp deck section.
        """
        sign = None
        inp_deck = f'*DFLUX\n'

        if self._on_all_elements:
            inp_deck += "eplate, "
        else:
            inp_deck += f'{self.group_name}, '

        if self.plane.direction[0] != 0:
            sign = np.sign(self.plane.direction[0])
        elif self.plane.direction[1] != 0:
            sign = np.sign(self.plane.direction[1])
            inp_deck += 'S4, '
        elif self.plane.direction[2] != 0:
            sign = np.sign(self.plane.direction[2])
            if sign == 1:
                inp_deck += 'S1, '
            else:
                inp_deck += 'S2, '
        else:
            raise ValueError(f"Direction in {self} must not be [0, 0, 0]")

        if sign == -1:
            inp_deck += f'-{self.magnitude}\n'
        else:
            inp_deck += f'{self.magnitude}\n'
    
        return inp_deck                            

    def is_point_on_plane(self, point):
        """Checks if point is on the loading plane.

        Args:
            point: Point object.

        Returns:
            Boolean dictating if the point is on the loading plane.
        """
        return self.plane.is_point_on_plane(point)

    def is_normal_coincident(self, face):
        """
        Checks if face normal and plane normal are coincident
        """
        face_norm = face.normal
        plane_norm = self.plane.direction / np.linalg.norm(self.plane.direction)
        dot = np.dot(face_norm, plane_norm)
        if abs(dot) == 1:
            return True
        return False

    def is_face_on_plane(self, face):
        """Checks if face is on the plane.

        Args:
            face: Face object.

        Returns:
            Boolean dictating if the face is on the plane.
        """
        if self.is_point_on_plane(face.centroid) and self.is_normal_coincident(face):
            return True

    def is_edge_on_plane(self, edges):
        """Checks if edge on plane

        Args:
            edges: Edges dictionary

        Returns:
            Boolean describing edge on the plane
        """
        entities = []
        for id, edge in edges.items():
            line_vec = edge.line.direction/np.linalg.norm(edge.line.direction)
            dot_prod = np.dot(self.plane.direction, line_vec)
            if dot_prod == 0.0 and self.is_point_on_plane(edge.midpoint):
                entities.append(edge)

        return entities


class ConcentratedHeatFlux(Stimulus):
    """Class for a concentrated heat flux load acting through a point on organism.

    Attributes:
        point: Point object as the source of the heat flux
        magnitude: Magnitude of the heat flux
    """

    def __init__(self, point, magnitude):
        super().__init__()
        self.point = point
        self.magnitude = magnitude
        self.application_type = 'conc_heat_flux'

    def get_closest_point(self, org_points):

        for ind, (_, point) in enumerate(org_points.items()):
            this_distance = np.linalg.norm(np.subtract(point.as_array, self.point.as_array))
            if ind == 0:
                distance = this_distance
                closest_point = point
            else:
                if this_distance < distance:
                    closest_point = point
                    distance = this_distance

        return closest_point

    def get_inp_deck(self):
        """Get section for inp deck.

        Returns:
            String for inp deck section.
        """

        inp_deck = f'*CFLUX\n{self.group_name}, 11, {self.magnitude}\n'

        return inp_deck
        
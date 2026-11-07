"""Class for initialising a curved plate seedling type."""

import gmsh
import numpy as np

from typing import Type

from src.initialisation.initialisation_parent import InitialisationParent
from src.devo.topology import CSType, Edge
from src.devo.organisms.flat_plate import FlatPlate
from src.environment.structural.plate_structural_element import PlateStructuralElement
from src.environment.thermal.plate_thermal_element import PlateThermalElement


class CurvedPlate(InitialisationParent):
    """
    Initialisation class for a curved plate seedling
    inheriting from the Initialisation Parent class

    Args:
        environment: environment object containing stimuli
        material: material object
        pt_1: first point of circle arc
        pt_2: centre of circle arc
        pt_3: third point of circle arc
        length: length of plate surface
        thickness: thickness of faces in seedling
        save_image: boolean to save an image of the seedling, default True
     """
    def __init__(self, environment, material_face, pt_1, pt_2, pt_3, length: float, thickness: float,
                 cs_type: Type[CSType]= CSType.rectangle, 
                 dim_1: float = None, 
                 dim_2: float = None,
                 cs_area: float = None,
                 material_edge = None,
                 save_image=False):
        super().__init__(environment, material_face, save_image)
        self.points = [pt_1, pt_2, pt_3]
        self.thickness = thickness
        self.length = length
        if material_edge is None:
            self.material_edge = material_face
        else:
            self.material_edge = material_edge
        self.cs_type= cs_type
        self.dim_1= dim_1
        self.dim_2= dim_2
        self.cs_area = cs_area


    def build_seedling_gmsh(self, mesh_size):
        """Method to build the seedling organism from the attributes

        - This method uses gmsh to build and mesh a plate based on the
        circle arc points and length attributes
        - The mesh data is extracted and the neighbour data is built which
        is then used to build the geometry and topology of the seedling by
        recursively stepping through element neighbours

        Returns:
            seedling: organism instance of seedling
        """
        Edge.default_cs_area = 0.0
        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", 1)
        gmsh.model.add("Model1")

        for ind in range(len(self.points)*2):
            if ind < len(self.points):
                gmsh.model.geo.addPoint(self.points[ind][0], self.points[ind][1], self.points[ind][2], 1, ind)
            else:
                new_ind = ind - len(self.points)
                gmsh.model.geo.addPoint(self.points[new_ind][0], self.points[new_ind][1], self.points[new_ind][2] + self.length, 1, ind)

        gmsh.model.geo.addCircleArc(0, 1, 2, ind + 1)   #6 
        gmsh.model.geo.addCircleArc(3, 4, 5, ind + 2)   #7
        gmsh.model.geo.addLine(0, 3, ind +  3)          #8
        gmsh.model.geo.addLine(2, 5, ind + 4)           #9
        gmsh.model.geo.addCurveLoop([6, 9, -7, -8], ind + 5)
        gmsh.model.geo.addSurfaceFilling([10], ind + 6)
        gmsh.model.geo.synchronize()
        #gmsh.fltk.run()
        
        gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size*2)
        gmsh.option.setNumber("Mesh.Algorithm", 5)
        gmsh.model.mesh.generate(2)
        #gmsh.fltk.run()

        _, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(2)
        node_tags = elem_node_tags[0]

        node_dict = {}
        all_node_tags, coord, _ = gmsh.model.mesh.getNodes(-1)
        for node_tag in all_node_tags:
            node_coords = coord[:3]
            node_dict[node_tag] = [node_coords]
            coord = np.delete(coord, [0, 1, 2])

        element_dict = {}
        for ind, _ in enumerate(elem_tags[0]):
            el_nodes = node_tags[:3]
            element_dict[ind] = list(el_nodes)
            node_tags = np.delete(node_tags, [0, 1, 2])
        
        gmsh.finalize()

        return element_dict, node_dict
    
    def build_seedling(self, mesh_size=3):

        element_dict, node_dict = self.build_seedling_gmsh(mesh_size)

        element_neighbours = self._find_tri_neighbours(element_dict) 

        all_points_dict = {}
        all_lines_dict = {}
        all_edges = []
        all_faces = []
        elements_visited = []
        seedling_dict = {}
        cell_edges = {}        

        el_id = 0
        self._build_tri_cells(el_id, element_neighbours, seedling_dict,
                                self.material, self.thickness,
                                node_dict, element_dict, all_points_dict,
                                all_lines_dict, all_faces, all_edges,
                                elements_visited, cell_edges,
                                self.material_edge, self.cs_area,
                                self.cs_type, self.dim_1, self.dim_2)

        return FlatPlate(generation_id=0, population_id=0, cells=seedling_dict)


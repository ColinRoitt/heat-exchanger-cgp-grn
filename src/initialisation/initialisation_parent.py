"""
The initialisation parent class for seedling initialisations prior to EvoDevo algoritm
"""

import os
import numpy as np
import itertools

from typing import Type
from abc import ABC, abstractmethod
from collections import defaultdict, OrderedDict

import src.environment.stimulus as stimulus
from src.environment.materials import Material
from src.devo.geometry import Point, Line
from src.devo.topology import Edge, Face
from src.devo.cell import Triangle, Tet


class InitialisationParent(ABC):
    """
    Parent class for initialisation of the seedling which will be
    inherited by child initialisation classes.

    Geometry, topology and cell classes are reset upon initialisation.
    Any image of seedlings in directory are removed.

    Args:
        environment: environment instance containining stimuli
        material: material instance
        save: saves image of seedling to root directory
    """
    def __init__(self, environment, material: Type[Material], save: bool):
        self.environment = environment
        self.material = material
        self.save = save

        Triangle.reset_instance_count()
        Tet.reset_instance_count()
        Point.reset_instance_count()
        Line.reset_instance_count()
        Edge.reset_instance_count()
        Face.reset_instance_count()

        if os.path.exists(os.path.join(os.getcwd(), "seedling.png")):
            os.remove(os.path.join(os.getcwd(), "seedling.png"))

    @abstractmethod
    def build_seedling(self):
        """
        Build the seedling geometry and topology to create an organism

        Returns:
            seedling: organism instance
        """
        pass

    def _find_tri_neighbours(self, element_dict):
        """
        Uses the element dictionary from the mesh data to find the neighbours
        of each element by its shared edge nodes

        Args:
            element_dict: dictinary of element index (key) and its node tags (value)

        Returns:
            Ordered dictionary of element id (key) and list of its element neighbours id (value)
        """
        all_elements = element_dict.values()
        elem_neighbours = defaultdict(list)
        for elem_id, elem_nodes in element_dict.items():
            this_el_edges = itertools.combinations(elem_nodes, 2)
            for this_edge in list(this_el_edges):
                neighbour_found = False
                other_el_id = 0
                while (neighbour_found is False) and (other_el_id < len(all_elements)):
                    other_elem_nodes = element_dict[other_el_id]
                    if not elem_id == other_el_id:
                        other_el_edges = itertools.combinations(other_elem_nodes, 2)
                        for other_edge in list(other_el_edges):
                            if set(other_edge) == set(this_edge):
                                if len(elem_neighbours) == 0:
                                    neighbour_found = True
                                    elem_neighbours[elem_id].append(other_el_id)
                                    elem_neighbours[other_el_id].append(elem_id)
                                elif not elem_id in elem_neighbours[other_el_id]:
                                    neighbour_found = True
                                    elem_neighbours[elem_id].append(other_el_id)
                                    elem_neighbours[other_el_id].append(elem_id)
                                else:
                                    neighbour_found = False
                    other_el_id += 1

                if (neighbour_found is False) and (len(elem_neighbours[elem_id]) < 3):
                    elem_neighbours[elem_id].append(-1)

        return OrderedDict(elem_neighbours)
    
    def _build_tri_cells(self, el_id, element_neighbours, seedling_dict,
                            face_material, thickness, 
                            node_dict, element_dict, all_points_dict,
                            all_lines_dict, all_faces, all_edges,
                            elements_visited, cell_edges,
                            edge_material=None, max_cs_area=None,
                            cs_type=None, dim_1=None, dim_2=None):

        """
        Traverses the neighbour data by element id and builds the geometry, topology and
        triangle cells of the seedling

        Args:
            el_id: integer element id from the mesh data
            element_neighbours: ordered dictionary of element neighbour by ids
            seedling_dict: dict of seedling cell ids and objects
            material: material instance for cells
            cs_area: cross section area of edges
            thickness: thickness of faces
            node_dict: dictionary of node tags and coordinates from mesh
            element_dict: dictionary of element index and node tags from mesh
            all_points_dict: dictionary of point objects and list of location
            all_lines_dict: dictionary of line object and indexed line (by point id)
            all_edges: list of all edge instances created
            all_faces: list of line indexed faces
            elements_visited: list of elements traversed by id number
            cell_edges: dict of cell id and its edge ids
        """
        elements_visited.append(el_id)
        neighbours = element_neighbours[el_id]

        el_node_coords = []
        el_node_ids = element_dict[el_id]
        cell_point_dict = {}
        for node in el_node_ids:
            coord = node_dict[node][0]
            el_node_coords.append(coord)

            for this_node in el_node_coords:
                this_point = Point(round(this_node[0], 10), round(this_node[1], 10), round(this_node[2],
                                                                                        10))
                if this_point.get_point_array().tolist() not in all_points_dict.values():
                    cell_point_dict[this_point.id] = this_point
                    all_points_dict[this_point] = this_point.get_point_array().tolist()
                else:
                    shared_point = [k for k, v in all_points_dict.items() if
                                    v == this_point.get_point_array().tolist()]
                    cell_point_dict[shared_point[0].id] = shared_point[0]
                    Point.remove_instance()

        cell_points = list(cell_point_dict.values())
        this_face_edges = [0]*3
        i = 0
        ind=-2
        while i != len(cell_points):
            if i == len(cell_points)-1:
                this_cell_line = Line(cell_points[i], cell_points[0])
            else:
                this_cell_line = Line(cell_points[i], cell_points[i+1])
            this_line_ind = this_cell_line.get_indexed_line()
            if this_line_ind not in all_lines_dict.values():
                if self.cs_type is None:
                    this_edge = Edge(edge_material, this_cell_line)
                elif self.cs_type.value == 0:
                    this_edge = Edge(edge_material, this_cell_line, cs_area=max_cs_area)
                elif self.cs_type.value == 1:
                    this_edge = Edge(edge_material, this_cell_line, max_cs_area, dim_1, dim_2, cs_type=cs_type)
                this_edge.on_boundary = True
                this_edge.shared = False
                if this_face_edges[ind] == 0:
                    this_face_edges[ind] = this_edge
                else:
                    this_face_edges[-1] = this_edge
                all_lines_dict[this_cell_line] = this_line_ind
                all_edges.append(this_edge)
                ind+=1
            else:
                shared_line = [k for k, v in all_lines_dict.items() if this_line_ind == v]
                shared_line[0].shared = True
                shared_line[0].on_boundary = False
                all_edges[shared_line[0].id].shared = True
                all_edges[shared_line[0].id].on_boundary = False
                # get cell of shared line as edge neighbour cell
                for c, e in cell_edges.items():
                    if shared_line[0].id in e:
                        all_edges[shared_line[0].id].neighbour = seedling_dict[c]
                # check when a cell has two or more shared edges
                if this_face_edges[0] == 0:
                    this_face_edges[0] = all_edges[shared_line[0].id]
                elif this_face_edges[1] == 0:
                    this_face_edges[1] = all_edges[shared_line[0].id]
                else:
                    this_face_edges[2] = all_edges[shared_line[0].id]
                this_cell_line.remove_instance()
            i+=1

        this_cell = Triangle(this_face_edges[0], this_face_edges[1], this_face_edges[2],
                            face_material, depth=0, thickness=thickness, cell_points=cell_points)
        cell_edges[this_cell.id] = [this_face_edges[0].id, this_face_edges[1].id, this_face_edges[2].id]

        # set the edge normal as the average of the normals of its cell and its neighbour cells normals
        for edge in this_face_edges:
            if edge.shared:
                edge._normal = (this_cell.normal + edge.neighbour.normal) / 2
            else:
                edge._normal = this_cell.normal

        if this_cell.indexed_faces not in all_faces:
            seedling_dict[this_cell.id] = this_cell
            all_faces.append(this_cell.indexed_faces)
        else:
            Triangle.remove_instance()

        for neighbour in neighbours:
            if neighbour != -1:
                if neighbour not in elements_visited:
                    self._build_tri_cells(neighbour, element_neighbours, seedling_dict,
                                        face_material, thickness, node_dict, element_dict,
                                        all_points_dict, all_lines_dict, all_faces, all_edges,
                                        elements_visited, cell_edges, edge_material, max_cs_area,
                                        cs_type, dim_1, dim_2)    


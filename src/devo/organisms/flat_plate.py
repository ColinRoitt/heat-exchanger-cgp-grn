"""Child class of plate parent to create a flat plate"""
import os
import numpy as np
import gmsh

from collections import defaultdict
from dataclasses import dataclass

import src.environment.fitnesses as fit
import src.environment.stimulus as stim
import src.environment.structural.complete_inp as inp_write
import src.environment.thermal .complete_inp as thermal_inp_write
import src.environment.structural.calculix as structural_calculix
import src.environment.thermal.calculix as thermal_calculix
import src.environment.constraints as grnc

from src.devo.organisms.plate_parent import Plate, State
from src.devo.geometry import Point
from src.devo.topology import Edge, Face
from src.devo.cell import Triangle
from src.environment.thermal.plate_thermal_element import PlateThermalElement
from src.environment.structural.plate_structural_element import PlateStructuralElement
import src.environment.stimulus as stim
from src.utils.errors import CcxElemNumError, InvalidDesignError, CcxElemSizeError, CcxElemJacobianError


@dataclass
class FlatPlateState(State):
    """Class for state information of flat plate organisms.

    Attributes:
        num_neurons: A dictionary that provides the number of neurons for each state variable.
    """
    num_neurons = {
                    "coord_x": 3,
                    "coord_y": 3,
                    "coord_z": 3,
                    "stress": 1,
                    "strain_energy": 1,
                    "heat_flux": 1,
                    "face_temp": 3,
                    "point_disps_x": 3,
                    "point_disps_y": 3,
                    "point_disps_z": 3,
                }


@dataclass
class ThicknessDelta:
    """A dataclass for the face thickness growth mechanism for plate objects.

    Attributes:
        num_neurons: Number of neurons for growth mechanism.
        csv_name: Name used to save csv file of GRN output.
    """
    num_neurons = 1
    csv_name = "thickness_delta"


@dataclass
class CoordDelta:
    """A dataclass for the point coordinate delta growth mechanism for plate objects.

    Attributes:
        num_neurons: Number of neurons for growth mechanism.
        csv_name: Name used to save csv file of GRN output.
    """
    num_neurons = 9
    csv_name = "coord_delta"


@dataclass
class FinAddition:
    """A dataclass for the fin addition growth mechanism for plate objects.

    Attributes:
        num_neurons: Number of neurons for growth mechanism.
        csv_name: Name used to save csv file of GRN output.
    """
    num_neurons = 3
    csv_name = "fin_addition"


class FlatPlate(Plate):
    """Class for a flate plate object.

    Attributes:
        generation_id: Id of the evo generation in which the organism exists.
        population_id: Id of the organism in the current population.
        cells: A dictionary of cell objects.
    """
    def __init__(self, generation_id, population_id, cells):
        super().__init__(generation_id, population_id, cells)
        self.state = FlatPlateState()
        self._points = {}
        self._lines = {}
        self._edges = {}
        self._faces = {}
        self.devo_steps = 0

    @property
    def points(self):
        """Get points of organism.

        Returns:
            Dictionary of point ids as keys and point objects as values.
        """
        self._points = {}
        for _ , line in sorted(self.lines.items()):
            for point in line.points:
                self._points[point.id] = point

        return self._points

    @property
    def lines(self):
        """Get lines of organism.

        Returns:
            Dictionary of line ids as keys and line objects as values.
        """
        self._lines = {}
        for _ , cell in sorted(self.cells.items()):
            for line in cell.lines:
                self._lines[line.id] = line

        return self._lines

    @property
    def edges(self):
        """Get edges of organism.

        Returns:
            Dictionary of edge ids as keys and edge objects as values.
        """
        self._edges = {}
        for _ , cell in sorted(self.cells.items()):
            for edge in cell.edges:
                self._edges[edge.id] = edge

        return self._edges

    @property
    def faces(self):
        """Get faces of organism.

        Returns:
            Dictionary of face ids as keys and face objects as values.
        """

        self._faces = {}
        for _ , cell in sorted(self.cells.items()):
            for face in cell.faces:
                self._faces[face.id] = face
        return self._faces

    @property
    def point_cell_map(self):
        """Maps state information of points to cells.

        Returns:
            Numpy array.
        """
        points = self.points
        cells = self.cells
        entity_map = defaultdict(list)

        for point_idx, (_, point) in enumerate(sorted(points.items())):
            for cell_idx, (_, cell) in enumerate(sorted(cells.items())):
                for cell_point in cell.points:
                    if point is cell_point:
                        entity_map[point_idx].append(cell_idx)

        return entity_map

    @property
    def face_cell_map(self):
        """Maps state information of faces to cells.

        Returns:
            Numpy array.
        """
        faces = self.faces
        cells = self.cells
        entity_map = defaultdict(list)

        for face_idx, (_, face) in enumerate(sorted(faces.items())):
            for cell_idx, (_, cell) in enumerate(sorted(cells.items())):
                for cell_face in cell.faces:
                    if face is cell_face:
                        entity_map[face_idx].append(cell_idx)

        return entity_map

    @property
    def network_outputs(self):
        """Get GRN outputs determined by the user.

        Return:
            List of network output dataclass objects.
        """
        return self._network_outputs

    def add_fin_addition_as_network_output(self):
        """Adds adding a fin cell as a GRN output.

        This adds a fin cell in the z direction from the edge of the cell face.

        Returns:
            None
        """
        self._network_outputs.append(FinAddition())

    def _generate_inp(self, iteration_path, stimuli_entities, env_element,node_temp_file_path=None):

        element_data, node_data = self._write_inp_mesh(iteration_path)
        stimuli_entities = self._get_stimulus_nodes(node_data, element_data, env_element)
        self._complete_inp(iteration_path, env_element, node_data, element_data, stimuli_entities,
                           node_temp_file_path)

    def _write_inp_mesh(self, iteration_path):

        node_data = {}
        element_data = {}

        node_script = "*NODE\n"
        for _, point in self.points.items():
            node_script += f"{point.id+1}, {point.x}, {point.y}, {point.z}\n"
            node_data[point.id+1] = [point.x, point.y, point.z]
        
        ele_script = "******* E L E M E N T S *************\n"
        for _, cell in self.cells.items():
            ele_id = cell.id + len(self.points) + 1
            ele_script += f"*ELEMENT, type=S3, ELSET=Surface{cell.id}\n"
            ele_script += f"{ele_id}, {cell.points[0].id+1}, {cell.points[1].id+1}, {cell.points[2].id+1}\n"
            element_data[ele_id] = [cell.points[0].id+1, cell.points[1].id+1, cell.points[2].id+1]

        mesh_script = node_script + ele_script

        f = open(f"{iteration_path}.inp", "w")
        f.write(mesh_script)
        f.close()

        return element_data, node_data

    def _get_stimulus_nodes(self, node_data, ele_data, environment_element):
        """
        Using the node data and environment, match the stimuli to a 'point' object
        created from the node data.

        This is to support writing the loads and boundary conditions to a beam mesh
        specifically.

        Args:
            node_data: dictionary with node ids (keys) and node coord list (values)
            environment_element: environment object

        Returns:
            Dictionary with stimulus object (keys) and a list of the point ids (values)
            the stimulus acts on.
        """
        stimulus_entities = defaultdict(list)
        edges = self.edges.values()
        faces = self.faces.values()
        points = {}
        for ind, coord in node_data.items():
            point = Point(coord[0], coord[1], coord[2])
            point.id = ind
            points[point.id] = point

        Point.set_instance_count(len(self.points))

        for stimulus in environment_element.stimuli:
            if isinstance(stimulus, stim.TwoDimConstraint):
                stimulus_entities[stimulus].append(-1)

            elif isinstance(stimulus, stim.Load):
                if stimulus.application_type.split("_")[0] == 'planar':
                    if stimulus.application_type.split("_")[1] == 'face':
                        for face in faces:
                            if stimulus.is_face_on_plane(face):
                                stimulus_entities[stimulus].append(face.id + len(points)+1)
                    elif stimulus.application_type.split("_")[1] == 'edge':
                        for edge in edges:
                            if stimulus.is_edge_on_plane(edge):
                                start_pt = edge.line.start_pt.id
                                end_pt = edge.line.end_pt.id
                                if start_pt not in stimulus_entities[stimulus]:
                                    stimulus_entities[stimulus].append(start_pt+1)
                                if end_pt not in stimulus_entities[stimulus]:
                                    stimulus_entities[stimulus].append(end_pt+1)                
            
            elif isinstance(stimulus, stim.Support):
                if stimulus.application_type.split("_")[0] == 'planar':
                    for _, point in points.items():
                        if stimulus.is_point_on_plane(point.as_array):
                            stimulus_entities[stimulus].append(point.id)
                else:                    
                    support_point = stimulus.get_point_at_support(points)
                    for point in support_point:
                        stimulus_entities[stimulus].append(point.id)
            elif isinstance(stimulus, stim.FilmCondition):
                stimulus_entities[stimulus].append("ALLELEMENTS")
            elif isinstance(stimulus, stim.ConcentratedHeatFlux):
                stim_point = stimulus.get_closest_point(points)
                stimulus_entities[stimulus].append(stim_point.id)
            elif isinstance(stimulus, stim.TemperaturePlanar):
                for _, point in points.items():
                    if stimulus.is_point_on_plane(point.as_array):
                        stimulus_entities[stimulus].append(point.id)
            elif isinstance(stimulus, stim.Temperature) and stimulus._on_all_nodes:
                stimulus_entities[stimulus].append("ALLNODES")
            elif isinstance(stimulus, stim.TemperaturePlanar):
                for _, point in points.items():
                    if stimulus.is_point_on_plane(point.as_array):
                        stimulus_entities[stimulus].append(point.id)             
   
        return stimulus_entities     

    def _complete_inp(self, file_path, env_element, node_labels, element_labels, stimuli_entities,
                      temp_file):
        """Completes an inp deck outputted from GMSH.

        Args:
            file_path: File path of inp file.
            env_element: Environment Element object
            node_labels: Node labels from GMSH.
            element_labels: Element labels from GMSH.
            stimuli_entities: Dictionary associating environmental stimuli with entities in organism
            Key is stimulus object, value is list of associated entity ids.
            temp_file: results file to be written in to inp, e.g. nodal temperatures

        Returns:
            None
        """
        with open(f"{file_path}.inp", 'r+') as file:

            inp_write._write_shell_elements(file)
            file.read()
            inp_write._write_generate_node_set(file, node_labels)
            inp_write._write_generate_element_set(file, element_labels)
            if isinstance(env_element, PlateThermalElement):
                thermal_inp_write._write_stimuli_groups(file, stimuli_entities)
            elif isinstance(env_element, PlateStructuralElement):
                inp_write._write_stimuli_groups(file, stimuli_entities)
            materials = inp_write._write_plate_shell_section(file, self)
            if isinstance(env_element, PlateThermalElement):
                thermal_inp_write._write_physical_constants(file)
            inp_write._write_materials(file, materials)
            inp_write._write_initial_conditions(file, stimuli_entities)
            if isinstance(env_element, PlateStructuralElement):
                inp_write._write_step(file)
            elif isinstance(env_element, PlateThermalElement):                
                thermal_inp_write._write_step(file, stimuli_entities)
            if isinstance(env_element, PlateStructuralElement) and temp_file is not None:
                inp_write._write_temp_from_csv_file(file, temp_file)
            # if isinstance(env_element, PlateThermalElement):
            #     thermal_inp_write._write_radiate(file, materials, 'all')
            if isinstance(env_element, PlateStructuralElement):
                inp_write._write_stimuli(file, stimuli_entities, 2)
            if isinstance(env_element, PlateThermalElement):
                thermal_inp_write._write_stimuli(file, stimuli_entities, 2)
            # inp_write._write_all_node_output(file)
            inp_write._write_all_node_output(file, output="temp")
            if isinstance(env_element, PlateThermalElement):
                 thermal_inp_write._write_element_output(file)
            else:
                inp_write._write_element_output(file)
            file.write('*END STEP')

    def _get_stimuli_entities(self, environment_element):
        """Gets the entities in the organism that are associated with environmental stimuli.

        Args:
            environment_element: environment element object

        Returns:
            Dictionary associating environmental stimuli with entities in organism. Key is stimulus
            object, value is list of entity ids
        """
        stimulus_entities = defaultdict(list)
        points = self.points
        edges = self.edges.values()
        faces = self.faces.values()

        for stimulus in environment_element.stimuli:
            if isinstance(stimulus, stim.InitialTemperature):
                stimulus_entities[stimulus].append("ALLNODES")

            elif isinstance(stimulus, stim.Temperature) and stimulus._on_all_nodes:
                stimulus_entities[stimulus].append("ALLNODES")

            elif isinstance(stimulus, stim.PointInit):
                pass

            elif isinstance(stimulus, stim.FilmCondition):
                if stimulus._on_all_elements:
                    stimulus_entities[stimulus].append("ALLELEMENTS")
                else:
                    for line in self.lines.items():
                        if np.allclose(line.start_pt.as_array, stimulus._location)\
                        or np.allclose(line.end_pt.as_array, stimulus._location):
                            stimulus_entities[stimulus].append(line.id)
                    for id, cell in self.cells.items():
                        if cell._is_fin:
                            stimulus_entities[stimulus].append(cell)
                

            elif isinstance(stimulus, stim.DistributedHeatFlux) and stimulus._on_all_elements:
                stimulus_entities[stimulus].append("ALLELEMENTS")

            elif isinstance(stimulus, stim.ConcentratedHeatFlux):
                nearest_point = stimulus.get_closest_point(self.points)
                stimulus_entities[stimulus].append(nearest_point)

            elif isinstance(stimulus, stim.Load):
                if stimulus.application_type.split("_")[0] == 'planar':
                    if stimulus.application_type.split("_")[1] == 'face':
                        for face in faces:
                            if stimulus.is_face_on_plane(face):
                                stimulus_entities[stimulus].append(face.id)
                    elif stimulus.application_type.split("_")[1] == 'edge':
                        for edge in edges:
                            if stimulus.is_edge_on_plane(edge):
                                start_pt = edge.line.start_pt.id
                                end_pt = edge.line.end_pt.id
                                if start_pt not in stimulus_entities[stimulus]:
                                    stimulus_entities[stimulus].append(start_pt)
                                if end_pt not in stimulus_entities[stimulus]:
                                    stimulus_entities[stimulus].append(end_pt)

            elif isinstance(stimulus, stim.Support):
                if stimulus.application_type.split("_")[0] == 'planar':
                    for edge in edges:
                        if stimulus.is_edge_on_plane(edge):
                            stimulus_entities[stimulus].append(edge.id)
                else:
                    support_point = stimulus.get_point_at_support(points)
                    for point in support_point:
                        stimulus_entities[stimulus].append(point.id)

            elif isinstance(stimulus, stim.TemperaturePlanar):
                entities =  stimulus.is_edge_on_plane(self.edges)
                for edge in entities:
                    stimulus_entities[stimulus].append(edge.id)

            elif isinstance(stimulus, stim.TemperaturePoint):
                nearest_point = stimulus.get_nearest_point_on_bc(points)
                stimulus_entities[stimulus].append(nearest_point)

            elif isinstance(stimulus, stim.Temperature):
                if stimulus.location is not None:
                    stimulus_entities[stimulus].append(stimulus.get_nearest_entity(edges))

        for _, point in sorted(points.items()):
            if point.is_loaded:
                for point_loadcase in point.loadcases:
                    stimulus_entities[point_loadcase].append(point.id)
            for stimulus in environment_element.stimuli:
                if isinstance(stimulus, stim.PointInit):
                    pass
                elif isinstance(stimulus, stim.Support):
                    if isinstance(stimulus, stim.PointConstraint):
                        if stimulus.is_point_at_constraint(point.as_array):
                            stimulus_entities[stimulus.group_name].append(point)
                elif isinstance(stimulus, stim.BoundaryCondition):
                    if isinstance(stimulus, stim.PlanarDisplacement):
                        if stimulus.is_point_on_plane(point.as_array):
                            stimulus_entities[stimulus].append(point.id)

        return stimulus_entities

    def get_network_inputs(self, environment):
        """Retrieve the network inputs for GRN.

        Args:
            environment: Environment object.

        Returns:
            Cell states as multi-dimensional Numpy array.
        """
        cell_states = [[] for _ in range(len(self.cells))]
        face_map = self.face_cell_map
        point_map = self.point_cell_map

        for element in environment:
            for network_input in element.network_inputs:
                state, entity_type = self.state.get_input(network_input)

                if entity_type == "face":
                    cell_states = self._get_state_from_map(face_map, state, cell_states)
                elif entity_type == "point":
                    cell_states = self._get_state_from_map(point_map, state, cell_states)

        return np.array(cell_states)
    
    def _get_state_from_map(self, entity_map, state, cell_states):
        """Get states from map between an entity and cell.

        Args:
            entity_map: Dictionary that maps entity to cells.
            state: Numpy array of organism state.
            cell_states: Cell states as multi-dimensional Numpy array.

        Returns:
            Cell states as multi-dimensional Numpy array.
        """
        for entity_idx in range(len(state)):
            cell_idxs = entity_map[entity_idx]
            for cell_idx in cell_idxs:
                cell_states[cell_idx].append(state[entity_idx])

        return cell_states    


    def update_with_network_outputs(self, output_signal, environment):
        """Takes the GRN outputs for each cell and updates the organism.

        Args:
            output_signal: The signal outputted from the GRN.
            environment: Environment object.

        Returns:
            None
        """
        self.devo_steps += 1
        idx = 0
        self.output_signal = np.array(output_signal)
        new_cells = []
        for output in self.network_outputs:
            signal = self.output_signal[:, idx:idx + output.num_neurons]
            idx += output.num_neurons

            fin_cells = []
            for cell in self.cells.items():
                if cell[1]._is_fin:
                    fin_cells.append(cell)

            if isinstance(output, ThicknessDelta):
                self._update_face_thickness(signal)
            elif isinstance(output, CoordDelta):
                self._update_point_coords(signal, environment)
            elif isinstance(output, FinAddition):
                new_cells.extend(self._update_fin_addition(signal))
        
        # this_cell_active_fins = []
        # for edge in cell.edges:
        #     for fin_cell in fin_cells:
        #         if fin_cell.hash() == edge.hash():
        #             this_cell_active_fins.append(fin_cell)



        # print(self.cells)

        self.cell_hashes = [cell[1].pos_hash for cell in self.cells.items()]

        count_new_cells = 0

        for cell in new_cells:
            # if the cell has the one edge we want to update then we shallll update it if not ignore it
            # if cell.edge_strings == "3.5647224768, 3.5061023464, 6.07733586581.9482820627, 4.6048015163, 8.1015915548":
            # print(cell.edge_strings)
            # if cell shares a pos_hash update its id to be the exists cell in self.cells
            id_to_set = cell.id
            for self_cell in self.cells.items():
                if self_cell[1]._is_fin:
                    # print(self_cell[1].pos_hash, cell.pos_hash)
                    if self_cell[1].pos_hash == cell.pos_hash:
                        count_new_cells += 1
                        id_to_remove = self_cell[1].id
                        # remove old cell
                        self.cells.pop(id_to_remove)
                        break
            self.cells[id_to_set] = cell

        # print("total cells", len(self.cells))
        # print("cells updates", len(new_cells))
        # print("new cells", count_new_cells)

        # count self.cells fins
        num_fins = 0
        for cell in self.cells.items():
            if cell[1]._is_fin:
                num_fins += 1

        # print(num_fins)
        # print(self.cells)
        # n = 1

    def _update_face_thickness(self, thickness_deltas):
        """Update triangle cell with thickness delta signals.

        - Associates the thickness deltas with a given face in the organism.
        - The thickness deltas are passed to the face to update the face thickness.

        Args:
            thickness_deltas: thickness delta signal for each face.

        Returns:
            None
        """
        thickness_deltas = np.array(thickness_deltas)
        faces = self.faces

        for idx, (face_id, _) in enumerate(sorted(self.cells.items())):
            faces[face_id].update_thickness(thickness_deltas[idx][0])

    def _update_point_coords(self, coord_deltas, environment):
        """Update triangle cell with point coordinate delta signals.

        - Associates the point coordinate delta with a given point in the organism.
        - Then takes an average of the delta where a given point is shared between triangle cells.
        - For certain stimuli, the delta values can be constrained.
        - The deltas are then passed to the points for updating the coordinates.

        Args:
            coord_deltas: Coordinate delta signal for each point.
            stimuli: Stimuli in the current environment.

        Returns:
            None
        """
        point_outputs = defaultdict(list)
        coord_deltas = np.array(coord_deltas).reshape([-1, 3, 3]).round(decimals=5)
        points = self.points

        for i, ( _ , cell) in enumerate(sorted(self.cells.items())):
            for j, point in enumerate(cell.points):
                point_outputs[point.id].append(coord_deltas[i][j])

        for point_id, output in sorted(point_outputs.items()):
            output_mean = np.mean(np.array(output), axis=0)
            for element in environment:
                for stimulus in element.stimuli:
                    if isinstance(stimulus, stim.Support):
                        if stimulus.is_point_at_support(points[point_id]):
                            output_mean = stimulus.constrain_coord_delta(output_mean)
                    if isinstance(stimulus, stim.PointConstraint):
                        if stimulus.is_point_at_constraint(points[point_id]):
                            output_mean = stimulus.constrain_coord_delta(output_mean)
                    if isinstance(stimulus, stim.PlanarConstraint):
                        if stimulus.is_point_at_constraint(points[point_id]):
                            output_mean = stimulus.constrain_coord_delta(output_mean)
            points[point_id].update_point_coord(output_mean)

    def _update_fin_addition(self, signal_to_add_fin):
        """Update triangle cell with according to fin addition signals.

        - Check if signal for addition is equal or above the trigger for addition.
        - If so a new cell is added onto the edge.
        - The cell is then deleted and the new cells are added to the organism cells dictionary.

        Args:
            signal_to_add_fin: Signals for addition for each cell in organism.

        Returns:
            None
        """
        new_cells = []
        fin_signals = np.array(signal_to_add_fin)
        edge_outputs = defaultdict(list)

        for idx, ( _ , cell) in enumerate(sorted(self.cells.items())):
            edge_outputs[cell.edge_0.id].append(fin_signals[idx][0])
            edge_outputs[cell.edge_1.id].append(fin_signals[idx][1])
            edge_outputs[cell.edge_2.id].append(fin_signals[idx][2])
        for idx, ( _ , cell) in enumerate(sorted(self.cells.items())):
            this_cell_signals = []
            for edge in cell.edges:
                this_cell_signals.append(np.mean(np.array(edge_outputs[edge.id])))
            new_cells.extend(cell.fin_addition(this_cell_signals))

        return new_cells

    def get_state(self, environment, save_dir, node_temp_file=None):
        """Retrieve fitness for organism from environmental element.

        Args:
            environment: list of Environment element objects
            save_dir: The run directory that all data is saved to, during a given evodevo run.
            temp_file_path: csv file of nodal temperatures produced from a heat transfer analysis to
              be included in structural analysis

        Returns:
            A list of different fitnesses and constraints.
        """

        fitnesses = []
        constraints = []

        calculix_path = os.path.join(save_dir, "calculix")
        iteration_path = os.path.join(calculix_path, self.iteration_name)

        for element in environment:

            stimuli_entities = self._get_stimuli_entities(element)

            if isinstance(element, PlateThermalElement):
                self._generate_inp(iteration_path, stimuli_entities, element)
                structural_calculix.run_ccx(iteration_path)
                fitness, constraint = self._update_physical_state(iteration_path, element)

                node_temp_file_path = f"{iteration_path}.csv"
                if os.path.isfile(node_temp_file_path) and\
                    os.stat(node_temp_file_path).st_size != 0:
                    node_temp_file = node_temp_file_path
                else:
                    node_temp_file = None

                fitnesses.extend(fitness)
                constraints.extend(constraint)

            if isinstance(element, PlateStructuralElement):
                element.element_checks()
                self._generate_inp(iteration_path, stimuli_entities, element, node_temp_file)
                structural_calculix.run_ccx(iteration_path)
                fitness, constraint = self._update_physical_state(iteration_path, element)

                fitnesses.extend(fitness)
                constraints.extend(constraint)

        return fitnesses, constraints

    def _update_physical_state(self, file_path, env_element):
        """Updates the physical state of each organism cell based on its interaction with the
        environmental element.

        - Gets certain variables from the organism.
        - Loads the dat file.
        - Uses the information in the dat file to calculate the forces on the edges.
        - Uses the forces on the edges to calculate other physical states.
        - Based on the fitness criteria, calculate the fitness of the organism.

        Args:
            file_path: File path of dat file produced by Calculix ccx.
            env_element: Environment element object

        Returns:
            Fitness of organism from its interaction with the environment.
        """
        constraint_criteria = env_element.constraint_criteria
        fitness_criteria = env_element.fitness_criteria

        fitness = []
        constraints = []

        point_coords = []
        points = self.points
        for _, point in sorted(points.items()):
            point_coords.append(point.as_array)
        point_coords = np.asarray(point_coords)

        self.state.coord_x = point_coords[:, 0]
        self.state.coord_y = point_coords[:, 1]
        self.state.coord_z = point_coords[:, 2]

        faces = self.faces
        face_areas = []
        thickness = []
        for _, face in sorted(faces.items()):
            face_areas.append(face.surface_area)
            thickness.append(face.thickness)
        face_areas = np.asarray(face_areas)
        thickness = np.asarray(thickness)

        try:
            lines = structural_calculix._load_dat_file(file_path)

            if isinstance(env_element, PlateStructuralElement):
                point_displacements = structural_calculix._get_point_displacements(lines, file_path,
                                                                                len(points))
                face_strain_energies = structural_calculix._get_face_strain_energies(lines,
                                                                                     len(faces))
                face_stresses = structural_calculix._get_face_stress(lines, len(faces))

                self.state.point_disps_x = point_displacements[:, 0]
                self.state.point_disps_y = point_displacements[:, 1]
                self.state.point_disps_z = point_displacements[:, 2]
                self.state.strain_energy = face_strain_energies
                self.state.stress = face_stresses

            if isinstance(env_element, PlateThermalElement):
																		 

                point_displacements = structural_calculix._get_point_displacements(lines, file_path, len(points))
                face_stresses = structural_calculix._get_face_stress(lines, len(faces))
                face_heat_flux = thermal_calculix._get_heat_flux(lines, len(faces))
                face_strain_energies = structural_calculix._get_face_strain_energies(lines,
                                                                                     len(faces))
                node_temperatures = thermal_calculix._get_node_temperatures(lines, file_path)

                self.state.point_disps_x = point_displacements[:, 0]
                self.state.point_disps_y = point_displacements[:, 1]
                self.state.point_disps_z = point_displacements[:, 2]
                self.state.strain_energy = face_strain_energies
                self.state.stress = face_stresses
                self.state.heat_flux = face_heat_flux
                self.state.strain_energy = face_strain_energies
                self.state.face_temp = np.array(list(node_temperatures.values()))

        except CcxElemNumError:
            print(f"Error: Ccx too many elements connected to one node for {self}.")

            for i, criteria in enumerate(constraint_criteria):
                if isinstance(criteria, grnc.FailedDesignConstraint):
                    constraints[i] = 1.0e10

            return [1.0e10] * len(fitness_criteria), constraints
        
        except CcxElemJacobianError:
            print(f"Error: Ccx element jacobian error for {self._iteration_name}.")

            for i, criteria in enumerate(constraint_criteria):
                if isinstance(criteria, grnc.FailedDesignConstraint):
                    constraints[i] = 1.0e10

            return [1.0e10] * len(fitness_criteria), [1.0e10] * len(constraints)

        except CcxElemSizeError:
            print(f"Error: Ccx element size error for {self._iteration_name}.")

            for i, criteria in enumerate(constraint_criteria):
                if isinstance(criteria, grnc.FailedDesignConstraint):
                    constraints[i] = 1.0e10

            return [1.0e10] * len(fitness_criteria), [1.0e10] * len(constraints)            

        except InvalidDesignError:
            print(f"Error: The network outputs have generated an invalid design for {self}.")

            for i, criteria in enumerate(constraint_criteria):
                if isinstance(criteria, grnc.FailedDesignConstraint):
                    constraints[i] = 1.0e10

            return [1.0e10] * len(fitness_criteria), constraints

        for this_fitness in fitness_criteria:
            if isinstance(this_fitness, fit.TotalStrainEnergyFit):
                fitness.append(structural_calculix.get_total_strain_energy(face_strain_energies))
            elif isinstance(this_fitness, fit.TotalVolumeFit):
                fitness.append(structural_calculix.get_total_volume_dim_two(organism=self))
            elif isinstance(this_fitness, fit.MaxDeflectionFit):
                fitness.append(structural_calculix.get_max_deflection(
                    np.array(point_displacements)))
            elif isinstance(this_fitness, fit.TotalStressFit):
                fitness.append(structural_calculix.get_total_stress(face_stresses))
            elif isinstance(this_fitness, fit.MaxStressFit):
                fitness.append(structural_calculix.get_max_stress(face_stresses))                
            elif isinstance(this_fitness, fit.TotalDeflectionFit):
                fitness.append(structural_calculix.get_total_deflection(point_displacements))
            elif isinstance(this_fitness, fit.TotalHeatFluxFit):
                fitness.append(thermal_calculix.get_total_heat_flux(face_heat_flux))
            elif isinstance(this_fitness, fit.TotalHeatTransferFit):
                fitness.append(thermal_calculix.get_total_heat_transfer(face_heat_flux,
                                                                        self, env_element.stimuli))
            elif isinstance(this_fitness, fit.NodeTempFit):
                fitness.append(thermal_calculix.get_node_temp(points, this_fitness, node_temperatures))
            elif isinstance(this_fitness, fit.AverageNodeTempFit):
                fitness.append(thermal_calculix.get_average_node_temp(points, this_fitness, node_temperatures))
            elif isinstance(this_fitness, fit.DevoStepsFit):
                fitness.append(self.devo_steps)


        for criteria in constraint_criteria:
            if isinstance(criteria, grnc.FailedDesignConstraint):
                constraints.append(0.0)
            if isinstance(criteria, grnc.TotalStrainEnergyConstraint):
                constraints.append(criteria.get_violation(
                    structural_calculix.get_total_strain_energy(face_strain_energies)))
            if isinstance(criteria, grnc.TotalVolumeConstraint):
                constraints.append(criteria.get_violation(
                    structural_calculix.get_total_volume_dim_two(organism=self)))
            if isinstance(criteria, grnc.TotalDeflectionConstraint):
                constraints.append(criteria.get_violation(
                    structural_calculix.get_total_deflection(point_displacements)))
            if isinstance(criteria, grnc.MaxDeflectionConstraint):
                constraints.append(criteria.get_violation(
                    structural_calculix.get_max_deflection(point_displacements,
                                                        direction=criteria.direction)))
            if isinstance(criteria, grnc.MaxStressConstraint):
                constraints.append(criteria.get_violation(face_stresses))

        if len(constraints) == 0:
            constraints = [-1]

        return fitness, constraints

    def write_stl(self, save_dir, save_step=False):
        """Not functional in plate"""
        pass

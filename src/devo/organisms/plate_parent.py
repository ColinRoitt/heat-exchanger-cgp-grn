""" Parent class for plate like organisms"""

import os
import numpy as np

from dataclasses import dataclass, fields
from typing import Optional, Any

import src.utils.errors as errors
from src.devo.organisms.organism_parent import Product
from src.utils.save import write_csv
from src.environment.thermal.calculix import get_volume_fraction


EPSILON = 1e-9


@dataclass
class State:
    """Class for state information of plate organisms.

    This state class should be inherited from in plate organism child classes.

    Attributes:
        thickness: thickness of each face in the organism.
        coord_x: X coordinates of points in the organism.
        coord_y: Y coordinates of points in the organism.
        coord_z: Z coordinates of points in the organism.
        stress: Stress induced by the loads on each face in the organism.
        strain_energy: Strain energy for each face in the organism.
        heat_flux: heat flux of each face in the organism.
        face_temp: node temperature of each cell in the organism
        point_disps_x: Displacement of points in the x direction.
        point_disps_y: Displacement of points in the y direction.
        point_disps_z: Displacement of points in the z direction.
        entity_dict: Dictionary mapping each state variable to a given entity e.g.'edge' or 'point'.
    """
    thickness: Optional[Any] = None
    coord_x: Optional[Any] = None
    coord_y: Optional[Any] = None
    coord_z: Optional[Any] = None
    stress: Optional[Any] = None
    strain_energy: Optional[Any] = None
    heat_flux: Optional[Any] = None
    face_temp: Optional[Any] = None
    point_disps_x: Optional[Any] = None
    point_disps_y: Optional[Any] = None
    point_disps_z: Optional[Any] = None
    cs_area: Optional[Any] = None
    volume: Optional[Any] = None
    channel_temp: Optional[Any] = None
    face_disp: Optional[Any] = None
    beam_disp: Optional[Any] = None
    beam_stress: Optional[Any] = None
    dim_1: Optional[Any] = None
    dim_2: Optional[Any] = None 

    entity_dict = {"thickness": "face", "coord_x": "point", "coord_y": "point", "coord_z": "point",
                   "stress": "face", "strain_energy": "face",  "heat_flux": "face",
                   "face_temp": "face", "point_disps_x": "point", "point_disps_y": "point",
                   "point_disps_z": "point", "cs_area" : "edge", "volume" : "edge", "channel_temp" : "edge",
                   "face_disp": "face", "beam_disp": "edge", "beam_stress": "edge", "dim_1": "edge",
                   "dim_2": "edge"}

    def get_input(self, name, normalise=True):
        """Retrieve network input from State class.

        Args:
            name: String of state variable name e.g. "thickness".

        Returns:
            Normalised state for given network input as Numpy array.
        """

        for field in fields(self):
            if name == field.name:
                state = getattr(self, field.name)
                if normalise:
                    return self.normalise(state), self.entity_dict[field.name]
                else:
                    return state, self.entity_dict[field.name]

        raise errors.StateVariableDoesNotExist

    def get_number_of_neurons(self, name):
        """Get the number of neurons for a given state variable.

        Args:
            name: String of state variable name e.g. "cs_areas".

        Returns:
            Int of the number of neurons for given state variable.
        """
        for field in fields(self):
            if name == field.name:
                return self.num_neurons[field.name]

    def normalise(self, state):
        """Normalises the state array on an Organism level using Z norm.

        Args:
            state: Numpy array of state variable.

        Returns:
            A Z normalised Numpy array of state variable.
        """
        # state = (state - np.mean(state, axis=0)) / (np.std(state, axis=0) + EPSILON)
        # return state
        # mean = 2.209494886820584
        # mean = 2.076738269699539
        # # std = 3.3824370429160977
        # std = 3.1920132397135594
        # state = (state - mean) / (std + EPSILON)
        return self.norm_log(state)

    def norm_log(self, state):
        """Normalises the state array on an Organism level using log.

        Args:
            state: Numpy array of state variable.

        Returns:
            A log normalised Numpy array of state variable.
        """
        # turn NaNs to 0
        # state = np.nan_to_num(state)
        state = np.log(state + EPSILON)
        return state

class Plate(Product):
    """Class for a plate object.

    Attributes:
        generation_id: Id of the evo generation in which the organism exists.
        population_id: Id of the organism in the current population.
        cells: A dictionary of cell objects.
    """
    def __init__(self, generation_id, population_id, cells):
        super().__init__(generation_id, population_id)
        self.cells = cells
        self._iteration_name = None
        self.state = State()

    @property
    def iteration_name(self):
        """Get name of the given organism iteration.

        Returns:
            Name of iteration.
        """
        self._iteration_name = f"{self.generation_id}-{self.population_id}-{self.clock}"
        return self._iteration_name

    def _save_csv(self, run_dir, environment, seedling=False):
        """Used to save the data of the organism relevant to the run.

        Args:
            run_dir: String of path to run directory.

        Returns:
            None
        """
        # Write the geometric entities of truss structure to csv files.
        geometry_dir = os.path.join(run_dir, "geometry")
        if not os.path.isdir(geometry_dir):
            os.mkdir(geometry_dir)

        write_csv(os.path.join(geometry_dir, "points.csv"), self.generation_id, self.population_id, self.clock,
                  np.asarray([pt.get_point_array() for _, pt in sorted(self.points.items())], dtype=object))

        write_csv(os.path.join(geometry_dir, "lines.csv"), self.generation_id, self.population_id, self.clock,
                  np.asarray([edge.line.get_indexed_line() for _, edge in sorted(self.edges.items())], dtype=object))
     
        write_csv(os.path.join(geometry_dir, "cs_areas.csv"), self.generation_id, self.population_id, self.clock,
                  np.asarray([edge.cs_area for _, edge in sorted(self.edges.items())], dtype=object))

        write_csv(os.path.join(geometry_dir, "vol_fraction.csv"), self.generation_id, self.population_id, self.clock,
                  np.asarray(get_volume_fraction(self), dtype=object))


        write_csv(os.path.join(geometry_dir, "cellxlines.csv"), self.generation_id, self.population_id, self.clock,
                  np.asarray([line.id for _, cell in sorted(self.cells.items()) for line in cell.lines], dtype=object))

        write_csv(os.path.join(geometry_dir, "fins.csv"), self.generation_id, self.population_id, self.clock,
                  np.asarray([edge._has_fin for _, edge in sorted(self.edges.items())], dtype=object))

   
        if not seedling:
            # Write the network inputs to csv files.
            network_input_dir = os.path.join(run_dir, "network_inputs")
            if not os.path.isdir(network_input_dir):
                os.mkdir(network_input_dir)

            for element in environment:
                for network_input in element.network_inputs:
                    if network_input == 'channel_temp':
                        state = self.channel_temp_csv
                    else:
                        state, _ = self.state.get_input(network_input, False)
                    write_csv(os.path.join(network_input_dir, f"{network_input}.csv"),
                            self.generation_id, self.population_id, self.clock, np.asarray(state))  
                            
            # Write the network outputs to csv files.
            network_output_dir = os.path.join(run_dir, "network_outputs")
            if not os.path.isdir(network_output_dir):
                os.mkdir(network_output_dir)

            idx = 0
            for output in self.network_outputs:
                signal = self.output_signal[:, idx:idx + output.num_neurons]
                idx += output.num_neurons
                file_path = os.path.join(network_output_dir, f"{output.csv_name}.csv")
                write_csv(file_path, self.generation_id, self.population_id, self.clock, signal)

        else:
            # Write the network inputs to csv files.
            network_input_dir = os.path.join(run_dir, "network_inputs")
            if not os.path.isdir(network_input_dir):
                os.mkdir(network_input_dir)

            for element in environment:
                for network_input in element.network_inputs:
                    if network_input == 'channel_temp':
                        state = self.channel_temp_csv
                    else:
                        state, _ = self.state.get_input(network_input, False)
                    write_csv(os.path.join(network_input_dir, f"{network_input}.csv"),
                            self.generation_id, self.population_id, self.clock, np.asarray(state))        

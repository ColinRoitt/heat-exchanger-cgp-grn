import os
import pickle

from abc import ABC, abstractmethod

import src.environment.constraints as grnc


class Organism(ABC):
    """Class for an Organism object.

    - The organism is initialised from a seedling (group of cells),
        it is the object that grows through the development (devo) cycle
          and interacts with the environment.
    - The organism consists of a group of cells which form its shape.
    - Each cell has a physical state which is updated when the organism
    interacts with environment.
    - This interact will also return a metric on the fitness of the organism
    for the given environment.
    - The organism also contain a GRN which is shared between each cell.
    - This GRN is the genome of the organism that makes decisions on growth
    based on the organism's current state.

    Attributes:
        generation_id: Id of the evo generation in which the organism exists.
        population_id: Id of the organism in the current population.
        clock: Internal counter which keeps track of the devo iteration the organism is on.
    """
    def __init__(self, generation_id, population_id):
        """Inits the Organism object with a given seedling."""
        self.generation_id = generation_id
        self.population_id = population_id
        self.clock = 0
        self.state = None
        self._network_outputs = []
        self._iteration_name = None
        self.save_as_csv = False
        self.save_as_pickle = False

    @property
    def iteration_name(self):
        """Get name of the given organism iteration.

        Returns:
            Name of iteration.
        """
        self._iteration_name = f"{self.generation_id}-{self.population_id}-{self.clock}"
        return self._iteration_name

    @abstractmethod
    def get_network_inputs(self):
        """Retrieve the network inputs for GRN.

        Args:
            environment: Environment object.

        Returns:
            Cell states as multi-dimensional Numpy array.
        """
        pass

    @abstractmethod
    def update_with_network_outputs(self, grn_output):
        """Takes the GRN outputs for each cell and updates the organism.

        Args:
            output_signal: The signal outputted from the GRN.
            environment: Environment object.

        Returns:
            None
        """
        pass

    @abstractmethod
    def get_state(self, environment):
        """Get state of organism from interaction with the environment.

        Args:
            environment: Environment object.

        Returns:
            Fitnesses and constraints as lists.
        """
        pass

    @abstractmethod
    def _save_csv(self, run_dir, environment):
        """Save organism data in csv file format.

        Args:
            run_dir: File path of run directory as string.
            environment: Environment object.

        Returns:
            None
        """
        pass

    def get_number_of_network_inputs(self, environment):
        """Retrieves the number of input neurons for the GRN.

        Returns:
            Number of neurons in GRN input layer as int.
        """
        num_neurons = 0
        for element in environment:
            for network_input in element.network_inputs:
                num_neurons += self.state.get_number_of_neurons(network_input)

        return num_neurons

    def get_number_of_network_outputs(self):
        """Retrieves the number of output neurons for the GRN.

        Returns:
            Number of neurons in GRN output layer as int.
        """
        num_outputs = 0
        for output in self.network_outputs:
            num_outputs += output.num_neurons

        return num_outputs

    def get_number_of_fitness_criteria(self, environment):
        """Get number of fitness criteria in environment.

        Args:
            environment: Environment object.

        Returns:
            Number of fitnesses as int.
        """
        num_fitness_criteria = 0
        for element in environment:
            num_fitness_criteria += len(element.fitness_criteria)
        return num_fitness_criteria

    def get_number_of_constraint_criteria(self, environment):
        """Get number of constraint criteria in environment.

        Args:
            environment: Environment object.

        Returns:
            Number of constraints as int.
        """
        num_constraint_criteria = 0
        for element in environment:
            constraints = element.constraint_criteria
            num_constraint_criteria += len(element.constraint_criteria)

            for criteria in constraints:
                if isinstance(criteria, grnc.SeedlingConstraint):
                    num_constraint_criteria -= 1
                    num_constraint_criteria += len(element.fitness_criteria)

        return num_constraint_criteria

    def save_organism(self, run_dir, environment, seedling=False):
        """Used to save organism data in various file formats.

        Returns:
            None
        """
        if self.save_as_csv:
            self._save_csv(run_dir, environment, seedling)
        if self.save_as_pickle:
            self._save_pickle(run_dir)

    def _save_pickle(self, run_dir):
        """Saves Organism object as Pickle file.

        run_dir: File path of run directory as string.

        Returns:
            None
        """
        org_dir = os.path.join(run_dir, "organisms")
        if not os.path.isdir(org_dir):
            os.mkdir(org_dir)

        org_path = os.path.join(org_dir, f"{self.iteration_name}.pkl")
        with open(org_path, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def save_seedling(self, run_dir):
        """Saves seedling as Pickle file.

        Returns:
            None
        """
        seedling_path = os.path.join(run_dir, "seedling.pkl")
        with open(seedling_path, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)


class Product(Organism):
    """Product Organism consiting of geometry"""
    def __init__(self, generation_id, population_id):
        super().__init__(generation_id, population_id)
        self.make_stl = False

    def _save_csv(self, run_dir, environment):
        """Save organism data in csv file format.

        Args:
            run_dir: File path of run directory as string.
            environment: Environment object.

        Returns:
            None
        """
        pass

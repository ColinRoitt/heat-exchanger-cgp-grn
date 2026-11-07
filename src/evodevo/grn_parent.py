"""Contains the GRNParent abstract class.

A gene regulatory network (GRN) is the growth controller in a cell, which
responses to state of the cell induced by an environment to adapt it.
In RIED, this GRN is often represented as a neural network. In this module,
the GRNParent abstract class is held which can be inherited from when
writing a new GRN class object.

Typical usage example:

  grn = GRN()
  growth_output = grn.activate(cell_state)
"""

import os
import dill

from abc import abstractmethod, ABC


class GRNParent(ABC):
    """Inherited from writing a new GRN class object.

    A developer should inherit from this class when creating a new GRN
    class object. This ensures every GRN has an activate and save_grn
    method.

    Class Attributes:
        grn_id: A unique id for the GRN object.

    Attributes:
        grn_id: A string denoting an individual id for the GRN in the run.
    """
    def __init__(self, gen_id, pop_id):
        """Initialises the GRN class object.

        Args:
            gen_id: A reference of the generation in which the GRN was created.
            pop_id: An id uniquely identifying the GRN in its population.
        """
        self.gen_id = gen_id
        self.pop_id = pop_id
        self.grn_id = f"{gen_id}-{pop_id}"
        self.grn = None

        self.fitness = None
        self.constraints = None
        self.running_saved_grn = False

    def set_grn(self, grn):
        self.grn = grn

    @abstractmethod
    def activate(self, cell_state):
        """Passes the cell state into the GRN outputting some growth mechanism.

        Args:
            cell_state: An array describing the cell state based on interactions with the environment.
        """
        pass

    def save_grn(self, run_dir, fitness, constraints):
        """Save the GRN class object.

        Used to save the GRN and its specific parameters as a Pickle file, so that
        it can be re-used in a seed for another experiment.

        Args:
            run_dir: A string dictating the current run directory.
            fitness: the fitness of the grn ==> match the fitness when this grn is loaded to run devo only
            constraint: the constraint of the grn ==> match the constraint when this grn is loaded to run devo only

        Returns:
            None
        """
        self.fitness = fitness
        self.constraints = constraints

        grn_dir = os.path.join(run_dir, "grns")
        grn_file_path = os.path.join(grn_dir, f"grn_{self.grn_id}.pkl")

        with open(grn_file_path, 'wb') as handle:
            dill.dump(self, handle, protocol=dill.HIGHEST_PROTOCOL)

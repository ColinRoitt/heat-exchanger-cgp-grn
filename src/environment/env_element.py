import src.environment.stimulus as stim


class EnvironmentalElement:
    """Abstract class for environmental elements.

    - The environment can be made up of different environmental elements, like the weather can be made up of the rain, sunshine and wind.
    - For us, this will be the different analyses like structural or thermal analysis.
    - These environmental elements are bespoke in their initialisation and combination for each environment, and will cause the emergence of different species.

    Attributes:
        _stimuli: The stimulus that generate a bespoke environmental element.
        _fitness_criteria: The fitness criteria associated with the environmental element.
        _constraints: The constraints that are provided to the GRN associated with the environmental element.
    """
    def __init__(self):
        self._stimulus_locations = []
        self._fitness_criteria = []
        self._constraint_criteria = []
        self._network_inputs = []
        self._stimuli = []
        self.save_analysis = False

    @property
    def stimuli(self):
        """Get the stimuli of the environment

        Returns:
            List of stimuli."""
        return self._stimuli

    @property
    def stimulus_locations(self):
        """Get the positions of stimuli.

        Returns:
            List of stimuli positions.
        """
        stim_locs = []
        for stimulus in self.stimuli:
            if not isinstance(stimulus, stim.Constraint):
                if stimulus.location not in stim_locs:
                    stim_locs.append(stimulus.location)
        self._stimulus_locations = stim_locs
        return self._stimulus_locations

    @property
    def fitness_criteria(self):
        """Get the fitness criteria of the environment

        Returns:
            List of fitness criteria.
        """
        return self._fitness_criteria

    @property
    def network_inputs(self):
        """Get network inputs of environmental element.

        Returns:
            List of network inputs.
        """
        return self._network_inputs

    @property
    def constraint_criteria(self):
        """Get the constraint criteria of the environment element.

        Returns:
            List of constraints.
        """
        return self._constraint_criteria

    @property
    def fitness_labels(self):
        """Get labels for each fitness criteria in the environment element.

        Returns:
            List of strings.
        """
        fitness_labels = []
        for criteria in self.fitness_criteria:
            fitness_labels.append(criteria.label)
        return fitness_labels

"""Class for plate structural element"""
import os

import src.environment.stimulus as stim
import src.environment.fitnesses as fit
import src.environment.constraints as grnc
from src.utils.errors import NoFitnessCriteriaInEnvElement, NoSupportsInEnvElement,\
    NoLoadDisporTempInEnvElement
from src.environment.env_element import EnvironmentalElement


class PlateStructuralElement(EnvironmentalElement):
    """An environmental element class for structural analysis of plate-like organisms.

    Attributes:
        _stimuli: The stimulus that generate a bespoke environmental element.
        _fitness_criteria: The fitness criteria associated with the environmental element.
        _constraint_criteria: The constraints that are provided to the GRN associated with the
        environmental element.
    """
    def __init__(self):
        """Inits the StructuralElement object with given stimuli."""
        super().__init__()

    def __repr__(self):
        """Defines the printable representation of the StructuralElement object."""
        return repr(f"Structural simulation with stimuli {self.stimuli}.")

    def add_initial_temperature(self, temperature):
        """Add temperature intitial conditions to the structural element

        Args:
            temperature: magnitude of temperature

        Returns:
            None
        """
        self._stimuli.append(stim.InitialTemperature(temperature))

    def add_total_volume_fit_crit(self):
        """Add total volume as a fitness criteria to the structural element.

        Returns:
            None
        """
        if not any(isinstance(x, fit.TotalVolumeFit) for x in self.fitness_criteria):
            self._fitness_criteria.append(fit.TotalVolumeFit())

    def element_checks(self):
        """Checks if the environmental element has been initialised correctly and is valid for
        analysis.

        Returns:
            None
        """
        num_loads = 0
        num_disp = 0
        num_supports = 0
        num_temp = 0

        for stimulus in self._stimuli:
            if isinstance(stimulus, stim.Load):
                num_loads += 1
            elif isinstance(stimulus, stim.Support):
                num_supports += 1
            elif isinstance(stimulus, stim.Temperature):
                num_temp += 1

        if num_loads == 0 and num_disp == 0 and num_temp == 0:
            raise NoLoadDisporTempInEnvElement("Error: No loads or displacements or\
                                               temperatures have been provided to structural\
                                                element.")

        if num_supports == 0:
            raise NoSupportsInEnvElement("Error: No supports have been provided to structural\
                                        element.")

        if len(self._fitness_criteria) == 0:
            raise NoFitnessCriteriaInEnvElement("Error: No fitness criteria have been provided\
                                                to structural element.")

    def remove_analysis_files(self, run_dir):
        """Remove analysis files from the Calculix directory.

        Arg:
            run_dir: The base run directory path.

        Returns:
            None
        """
        if not self.save_analysis:
            calculix_dir = os.path.join(run_dir, "calculix")
            for file in os.listdir(calculix_dir):
                if file.split(".")[0].split("-")[-1] == "lba":
                    os.remove(os.path.join(calculix_dir, file))                
                elif file.split(".")[-1] != "inp":
                    os.remove(os.path.join(calculix_dir, file))



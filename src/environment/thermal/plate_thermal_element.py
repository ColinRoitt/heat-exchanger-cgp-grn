"""Class for plate thermal element"""
import os
import src.environment.stimulus as stim
import src.environment.fitnesses as fit
from src.utils.errors import NoFitnessCriteriaInEnvElement, NoSupportsInEnvElement,\
    NoLoadDisporTempInEnvElement
from src.environment.env_element import EnvironmentalElement
import src.environment.constraints as grnc


class PlateThermalElement(EnvironmentalElement):
    """An environmental element class for thermal analysis of plate-like organisms.

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
        """Defines the printable representation of the ThermalElement object."""
        return repr(f"Thermal simulation with stimuli {self._stimuli}.")
    
    def add_film_condition(self,  film_coefficient, ambient_temperature,
                           pos_x=None, pos_y=None, pos_z=None,
                           all_elements=False):
        """Add a film condition interaction to the thermal element
        
        Args:
            all_elements: Boolean to add to all elements of organism
            film_coefficient: film coefficient value 
            ambient_temperature: ambient temperature value
        
        Returns:
            None
        """
        self._stimuli.append(stim.FilmCondition(film_coefficient,
                                                ambient_temperature,
                                                pos_x=pos_x,
                                                pos_y=pos_y,
                                                pos_z=pos_z,
                                                all_elements=all_elements))

    def add_initial_temperature(self, temperature):
        """Add temperature intitial conditions to the thermal element

        Args:
            temperature: magnitude of temperature

        Returns:
            None
        """
        self._stimuli.append(stim.InitialTemperature(temperature))

    def add_conc_heat_flux(self, point, magnitude):
        """Add concentrated heat flux to the thermal element
        
        Args:
            point: Point object as the source of the heat flux
            magnitude: Magnitude of the heat flux
            
        Returns:
            None
        """  
        self._stimuli.append(stim.ConcentratedHeatFlux(point, magnitude))

    def add_average_node_temp_fit_crit(self, target=None):
        """Add average node temperature fitness criteria to the thermal element
        
        Args:
            target: target for fitness value

        """
        if not any(isinstance(x, fit.AverageNodeTempFit) for x in self.fitness_criteria):
            self._fitness_criteria.append(fit.AverageNodeTempFit(target=target))
    
    def add_devo_steps_fit_crit(self):
        """Add devo steps as a fitness criteria to the thermal element.

        Returns:
            None
        """
        if not any(isinstance(x, fit.DevoStepsFit) for x in self.fitness_criteria):
            self._fitness_criteria.append(fit.DevoStepsFit())

    def add_total_volume_fit_crit(self):
        """Add total volume as a fitness criteria to the thermal element.

        Returns:
            None
        """
        if not any(isinstance(x, fit.TotalVolumeFit) for x in self.fitness_criteria):
            self._fitness_criteria.append(fit.TotalVolumeFit())

    def add_heat_flux_as_network_input(self):
        """Add heat flux as a network input to the thermal element.

        Returns:
            None
        """
        if "heat_flux" not in self.network_inputs:
            self.network_inputs.append("heat_flux")

    def element_checks(self):
        """Checks if the environmental element has been initialised correctly and is valid for
        analysis.

        Returns:
            None
        """
        num_supports = 0
        num_temp = 0

        for stimulus in self._stimuli:
            if isinstance(stimulus, stim.Support):
                num_supports += 1
            elif isinstance(stimulus, stim.Temperature):
                num_temp += 1

        if  num_temp == 0:
            raise NoLoadDisporTempInEnvElement("Error: No temperatures have been provided to this\
                                                thermal element.")

        if num_supports == 0:
            raise NoSupportsInEnvElement("Error: No supports have been provided to thermal\
                                        element.")

        if len(self._fitness_criteria) == 0:
            raise NoFitnessCriteriaInEnvElement("Error: No fitness criteria have been provided\
                                                to thermal element.")

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
                os.remove(os.path.join(calculix_dir, file))
        else:
            # save only frd files
            calculix_dir = os.path.join(run_dir, "calculix")
            for file in os.listdir(calculix_dir):
                # do not remove frd files
                if not file.split(".")[-1] == "frd" and not file.split(".")[-1] == "inp":# and not file.split(".")[-1] == "dat" and not file.split(".")[-1] == "log":
                # if not file.split(".")[-1] == "frd" and not file.split(".")[-1] == "inp" :# and not file.split(".")[-1] == "dat" and not file.split(".")[-1] == "log":
                    os.remove(os.path.join(calculix_dir, file))

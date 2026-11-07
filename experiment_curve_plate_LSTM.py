"""
Evolve fin growth on a heated curved plate with a CGP gene regulatory network.
"""

from src.initialisation.initialisation_curve_plate_seedling import CurvedPlate
from src.devo.geometry import Point
from src.environment.materials import Steel_8620
from src.environment.thermal.plate_thermal_element import PlateThermalElement
from src.evodevo.evodevo_utils import get_algorithm
import src.utils.arg_parser as arg_parser

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

def main(args=None):
    """
    Curved plate with a concentrated heat flux, cooled by a film condition.
    Objectives: average node temperature, total volume and devo steps.
    """
    args = arg_parser.get_parser(args)
    # Create environment from stimuli
    thermal_element = PlateThermalElement()
    thermal_element.add_film_condition(all_elements=True, film_coefficient=0.0034, ambient_temperature=293.0)
    thermal_element.add_conc_heat_flux(point=Point(4.54, 4.54, 6.0), magnitude=10.0)
    thermal_element.add_initial_temperature(temperature=293.0)

    # minimise average node temperature
    thermal_element.add_average_node_temp_fit_crit()
    # minimise total volume of plate (in fins)
    thermal_element.add_total_volume_fit_crit()
    # minimise devo steps
    thermal_element.add_devo_steps_fit_crit()

    thermal_element.add_heat_flux_as_network_input()

    thermal_element.save_analysis = True

    environment = [thermal_element]

    Point.reset_instance_count()
    this_plate_seedling = CurvedPlate(environment=environment, material_face=Steel_8620(),
                            pt_1=[0.0, 5.0, 0.0], pt_2=[0.0, 0.0, 0.0],
                            pt_3=[5.0, 0.0, 0.0], length=10, thickness=0.1, cs_type=None)
    seedling = this_plate_seedling.build_seedling(mesh_size=1)

    seedling.add_fin_addition_as_network_output()

    seedling.save_as_csv = True

    if args.make_stl:
        seedling.make_stl = True

    print(f"EvoDevo algorithm: {args.algorithm}")

    algo = get_algorithm(args)
    algo.evo_devo(environment, seedling)

    print("End")

if __name__ == '__main__':
    # Self-stopping is switched by hand: use the --self_stopping line for the
    # stopping condition and the line without it for the fixed-step condition.
    main(['-p', '30', '-e', '50', '-d', '10', '-algo', 'CGP', '--self_stopping'])
    # main(['-p', '30', '-e', '50', '-d', '10', '-algo', 'CGP'])

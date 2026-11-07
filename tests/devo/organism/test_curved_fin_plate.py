"""Test class for a plate like organism in a thermal environment."""

import os
import pytest

import src.environment.stimulus as stim
from src.environment.materials import Steel_8620
from src.environment.thermal.plate_thermal_element import PlateThermalElement
from src.devo.geometry import Point, Plane
from src.initialisation.initialisation_curve_plate_seedling import CurvedPlate


@pytest.fixture
def file_path(save_dir):
    """ Fixture for file path to store files"""
    file_path = os.path.join(save_dir, "calculix")
    if not os.path.exists(file_path):
        os.mkdir(file_path)
    return os.path.join(file_path, "curved_fin_plate")


@pytest.fixture
def error_file_path(save_dir):
    """Fixture for where organism saves its errors."""
    return os.path.join(save_dir, "errors.csv")


@pytest.fixture
def save_dir():
    """Fixture for where organism saves its data."""
    ccx_file_path = os.path.join(os.getcwd(), "tests/data/")
    os.makedirs(os.path.join(ccx_file_path, "calculix"), exist_ok=True)
    return ccx_file_path


@pytest.fixture(name="this_material")
def material_fixture():
    """Fixture for seedling material"""
    return Steel_8620()


@pytest.fixture(name="this_environment", scope='function')
def environment_fixture():
    """Fixture for the environment"""
    thermal_element = PlateThermalElement()
    thermal_element.add_film_condition(all_elements=True, film_coefficient=0.0034, ambient_temperature=293.0)
    thermal_element.add_conc_heat_flux(point=Point(3, 3, 5), magnitude=10.0)
    thermal_element.add_initial_temperature(temperature=293.0)

    return [thermal_element]


@pytest.fixture(name="this_organism")
def plate_seedling_obj_fixture(this_environment, this_material):
    """Fixture for instance of plate organism"""
    Point.reset_instance_count()
    Plane.reset_instance_count()
    this_plate_seedling = CurvedPlate(environment=this_environment, material_face=this_material,
                            pt_1=[0.0, 5.0, 0.0], pt_2=[0.0, 0.0, 0.0],
                            pt_3=[5.0, 0.0, 0.0], length=10, thickness=0.1, cs_type=None)
    seedling = this_plate_seedling.build_seedling()
    return seedling


def test_retrieving_stimuli_points(this_environment, this_organism):
    """Test that the stimuli can be linked with their associated points."""
    env_element = this_environment[0]
    stimuli_entities = this_organism._get_stimuli_entities(env_element)
    stimuli_points = list(stimuli_entities.keys())
    assert isinstance(stimuli_points[1], stim.ConcentratedHeatFlux)
    assert isinstance(stimuli_points[0], stim.FilmCondition)


def test_getting_total_volume_fitness_crit(this_environment, this_organism, save_dir):
    """Test that the fitness can be retrieved."""
    env_element = this_environment[0]
    env_element.add_total_volume_fit_crit()
    fitness, _ = this_organism.get_state(this_environment, save_dir)
    assert round(fitness[0], 1) == 7.8

"""Smoke tests for the curved plate fin growth experiment, both stopping conditions."""

import src.environment.stimulus as stim
from src.devo.geometry import Point, Line
from src.devo.topology import Edge, Face
from src.devo.cell import Triangle
from experiment_curve_plate_LSTM import main as experiment_main


def reset_class_instance_counts():
    """Reset instances of geometry, topology and stimuli"""
    Point.reset_instance_count()
    Line.reset_instance_count()
    Edge.reset_instance_count()
    Face.reset_instance_count()
    Triangle.reset_instance_count()
    stim.FilmCondition.reset_instance_count()
    stim.ConcentratedHeatFlux.reset_instance_count()
    stim.InitialConditions.reset_instance_count()


def test_fixed_steps_run():
    """Run the fixed-step condition"""
    reset_class_instance_counts()
    experiment_main(['-e', '1', '-p', '2', '-d', '3', '-algo', 'CGP'])


def test_self_stopping_run():
    """Run the self-stopping condition"""
    reset_class_instance_counts()
    experiment_main(['-e', '1', '-p', '2', '-d', '3', '-algo', 'CGP', '--self_stopping'])

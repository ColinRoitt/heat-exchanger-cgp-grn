"""Tests for the Line geometric class."""

import pytest
import numpy as np

from OCC.Core.Geom import Geom_Line

from src.devo.geometry import Line, Point


@pytest.fixture
def start_point():
    """Start point fixture."""
    return Point(x=0.0, y=0.0, z=0.0)


@pytest.fixture
def end_point():
    """End point fixture."""
    return Point(x=10.0, y=-10.0, z=10.0)


@pytest.fixture
def line(start_point, end_point):
    """Line object fixture."""
    return Line(start_point=start_point, end_point=end_point)


def test_line_id_increments(start_point, end_point):
    """Test the line id increments when a new line object is instantiated."""
    Line.reset_instance_count()
    assert Line(start_point=start_point, end_point=end_point).id == 0
    assert Line(start_point=start_point, end_point=end_point).id == 1


def test_line_id_resets():
    """Test line class instance count and id can be reset."""
    Line.reset_instance_count()
    assert Line(start_point=start_point, end_point=end_point).id == 0
    assert Line(start_point=start_point, end_point=end_point).id == 1
    Line.reset_instance_count()
    assert Line(start_point=start_point, end_point=end_point).id == 0


def test_line_length(line):
    """Test the Line object returns the correct length."""
    assert np.array_equal(line.length, 17.320508075688775)


def test_line_midpoint(line):
    """Test the Line object returns the correct midpoint as numpy array."""
    assert np.array_equal(line.midpoint, np.array([5.0, -5.0, 5.0]))


def test_points_of_line(line, start_point, end_point):
    """Test the Line object returns the correct points."""
    assert line.points == [start_point, end_point]


def test_line_as_opencascade_object(line):
    """Test that the line object can be represented as an OpenCascade gp_Line object."""
    line_occ = line.as_occ
    assert type(line_occ == Geom_Line)
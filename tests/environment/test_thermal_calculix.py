"""Tests for the Calculix module."""

import pytest
import numpy as np
from collections import defaultdict

import src.environment.thermal.calculix as calculix
from src.devo.geometry import Point


@pytest.fixture
def dat_lines():
    """Load lines of test dat file."""
    dat_file_path = "tests/environment/thermal_test.dat"
    with open(dat_file_path) as dat_file:
        lines = [line.strip() for line in dat_file if not line.isspace()]
    return lines


@pytest.fixture
def inp_file_path():
    """Load lines of test inp file."""
    return "tests/environment/thermal_test"


@pytest.fixture
def msh_file_path():
    """Load lines of test msh file."""
    return "tests/environment/thermal_test.msh"

def test_dat_file_not_empty(dat_lines):
    assert len(dat_lines) != 0

def test_get_total_heat_flux(dat_lines):
    element_res = []
    for ind, line in enumerate(dat_lines):
        marker_line = line.split(" ")
        if marker_line[0] == "heat" and marker_line[1] == "flux" and marker_line[-1] == "0.1000000E+01":
            start_ind = ind + 1

    for line in dat_lines[start_ind:]:
        new_line = list(line.rsplit()[:-1])
        new_line = list(map(float, new_line))
        element_res.append(new_line)

    total_qx = 0
    total_qy = 0
    total_qz = 0

    element_res = np.array_split(np.asarray(element_res), len(element_res)/2)
    for el_label, element in enumerate(element_res):
        label = int(element[:, 0].mean())
        qx = element[:, 2].mean()
        total_qx += qx
        qy = element[:, 3].mean()
        total_qy += qy
        qz = element[:, 4].mean()
        total_qz += qz

    print(total_qx/len(element_res))

    assert total_qx/len(element_res) == -102.51900000000022
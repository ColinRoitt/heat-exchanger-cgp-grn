"""Module for different material classes. Units must be consistent and in mm.
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Material:
    """Class for holding properties on a given material.
    All units must be consistent with mm.
    Please refer below for guidance.
    Quantity | SI (mm)
    -------------------
    Length        | mm
    Force         | N
    Mass          | kg
    Time          | s
    Stress        | MPa (N/mm2)
    Energy        | mJ (10^-3 J)
    Density       | kg/mm3
    Conductivity  | W/mm K @ K
    Specific heat | J/kg K
    Expansion     | /K
    """
    name: str = None
    density: float = None
    youngs_mod: float = None
    poisson_ratio: float = None
    tensile_yield_strength: float = None
    tensile_ultimate_strength: float = None
    conductivity: List[float] = None
    specific_heat: float = None
    expansion: List[float] = None
    emissivity: float = None
    beam_length_costs: Dict[float, float] = None


@dataclass
class Steel_8620(Material):
    """Class to hold material properties of AISI Steel 8620.

    Assuming galvanised steel for emissivity value

    Info from: https://asm.matweb.com/search/SpecificMaterial.asp?bassnum=M862AA
    """
    name: str = "AISI 8620 Steel"
    density: float = 7.85e-6
    youngs_mod: float = 2.05e5
    poisson_ratio: float = 0.29
    tensile_yield_strength: float = 385
    tensile_ultimate_strength: float = 530
    conductivity: List[float] = field(default_factory=lambda: [4.66e-2, 323.0])
    specific_heat: float = 475.0
    expansion: List[float] = field(default_factory=lambda: [12e-6, 293.0])
    emissivity: float = 0.28
    beam_length_costs: Dict[float, float] = field(
        default_factory=lambda: {12.5: 50.00, 15.0: 100.00, 20.0: 125.00, 25.0: 150.00})



@dataclass
class Aluminium_6061T6(Material):
    """Class to hold material properties of Aluminium 6061-T6.

    Info from: https://asm.matweb.com/search/SpecificMaterial.asp?bassnum=ma6061t6
    """
    name: str = "Aluminium 6061-T6"
    density: float = 2.7e-6
    youngs_mod: float = 6.89e4
    poisson_ratio: float = 0.33
    tensile_yield_strength: float = 276
    tensile_ultimate_strength: float = 310
    conductivity: List[float] = field(default_factory=lambda: [0.167, 298.15])
    specific_heat: float = 896.0
    expansion: List[float] = field(default_factory=lambda: [2.36e-5, 295.3])
    beam_length_costs: Dict[float, float] = field(
        default_factory=lambda: {5.0: 2.0, 10.0: 2.2, 20.0: 2.64, 50.0: 3.41})
    

@dataclass
class Aluminium_2024_T4(Material):
    """Class to hold material properties of Aluminium 2024-T4.

    Info from: https://asm.matweb.com/search/SpecificMaterial.asp?bassnum=ma2024t4 
    """
    name: str = "Aluminium 2024-T4"
    density: float = 2.78e-6
    youngs_mod: float = 7.31e4
    poisson_ratio: float = 0.33
    tensile_yield_strength: float = 324
    tensile_ultimate_strength: float = 469
    conductivity: List[float] = field(default_factory=lambda: [0.121, 298.15])
    specific_heat: float = 867.0
    expansion: List[float] = field(default_factory=lambda: [2.32e-5, 293.15])
    # beam_length_costs: Dict[float, float] = field(
    #     default_factory=lambda: {5.0: 2.0, 10.0: 2.2, 20.0: 2.64, 50.0: 3.41})



@dataclass
class PLA_Matt(Material):
    """Class to hold material properties of PLA_Matt.

    Info from: https://filamentive.com/wp-content/uploads/2019/06/TDS_PLAMATTE.pdf and
    https://www.sd3d.com/wp-content/uploads/2017/06/MaterialTDS-PLA_01.pdf
    """
    name: str = "PLA Matt"
    density: float = 1.38e-6
    youngs_mod: float = 3300
    poisson_ratio: float = 0.33
    tensile_yield_strength: float = 46.6
    tensile_ultimate_strength: float = 46.6
    conductivity: List[float] = field(default_factory=lambda: [1.3e-4, 293.0])
    specific_heat: float = 1800
    expansion: List[float] = field(default_factory=lambda: [4.1e-5, 293.0])
    beam_length_costs: Dict[float, float] = field(
        default_factory=lambda: {5.0: 0.1, 10.0: 0.5, 20.0: 1.4, 50.0: 2.0})


@dataclass
class cpTitanium(Material):
    """Class to hold material properties of commercially pure Titanium.

    Info from: https://www.americanelements.com/titanium-commercially-pure-cp-7440-32-6
    """
    name: str = "cp Titanium"
    density: float = 4.54e-6
    youngs_mod: float = 116000
    poisson_ratio: float = 0.32
    tensile_yield_strength: float = 140
    tensile_ultimate_strength: float = 140
    conductivity: List[float] = field(default_factory=lambda: [0.0219, 298.2])
    specific_heat: float = 523.35
    expansion: List[float] = field(default_factory=lambda: [8.6e-6, 298.15])

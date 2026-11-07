from dataclasses import dataclass
from src.devo.geometry import Point


@dataclass
class TotalStrainEnergyFit:
    label = "Total Strain Energy (Nmm)"


@dataclass
class TotalVolumeFit:
    label = "Total Volume (mm^3)"


@dataclass
class MaxStressFit:
    label = "Max Stress (MPa)"


@dataclass
class TotalStressFit:
    label = "Total Stress (MPa)"


@dataclass
class AverageStressFit:
    label = "Average Stress (MPa)"


@dataclass
class MaxDeflectionFit:
    label = "Max Deflection (mm)"


@dataclass
class TotalDeflectionFit:
    label = "Total Deflection (mm)"


@dataclass
class CriticalBucklingLoadFit:
    label = "Buckling Load (N/mm)"
    target: float    

@dataclass
class TotalBucklingFit:
    label = "Total Buckling"


@dataclass
class StiffnessFit:
    label = "Stiffness"


@dataclass
class LatticePropsFit:
    label = "Lattice Props"


@dataclass
class TotalHeatFluxFit:
    label = "Total heat flux (W/mm)"

@dataclass
class AverageHeatFluxFit:
    label = "Average Heat Flux (W/mm)"

@dataclass
class TotalHeatTransferFit:
    label = "Total heat transfer (W)"

@dataclass
class NodeTempFit():
    label = "Node temp (K)"
    x: float
    y: float
    z: float

@dataclass
class AverageNodeTempFit():
    label = "Average node temp (K)"
    target: float

@dataclass
class VolumeFraction():
    label = "Volume Fraction"

@dataclass
class TempDiff():
    label = "Temperature Difference"
    point1: Point
    point2: Point

@dataclass
class EdgeVolume():
    label = "Edge Volume"

@dataclass
class NonDimTemp():
    label = "Non-Dimensional Temperature"

@dataclass
class DevoStepsFit():
    label = "Devo Steps"
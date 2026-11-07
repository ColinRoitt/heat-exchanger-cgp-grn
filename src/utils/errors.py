class CcxElemNumError(Exception):
    """Raised when ccx outputs *ERROR in gen3dnor: more than 100."""
    pass

class CcxElemSizeError(Exception):
    """Raised when ccx outputs *ERROR in gen3dnor: size of estimated."""
    pass

class CcxElemJacobianError(Exception):
    """Raised when ccx outputs *ERROR in e_c3d_th: nonpositive jacobian determinant."""
    pass

class CcxCutBackError(Exception):
    """Raised when ccx cannot converge on a solution."""
    pass

class InvalidDesignError(Exception):
    """Raised when the design is not valid for analysis."""
    pass


class NoLoadInEnvElement(Exception):
    """Raised when no loads have been provided to an environmental element object."""
    pass


class NoFitnessCriteriaInEnvElement(Exception):
    """Raised when no fitness criteria have been provided to an environmental element object."""
    pass


class NoSupportsInEnvElement(Exception):
    """Raised when no supports have been provided to an environmental element object."""
    pass


class NoLoadDisporTempInEnvElement(Exception):
    """Raised when no loads, displacements or temperatures have been provided to an environmental element object."""
    pass


class NoElementsInEnvironment(Exception):
    """Raised when no environmental elements have been provided to an environmental object."""
    pass


class IncorrectEnvironmentalElementType(Exception):
    """Raised when the incorrect environmental element type have been provided to an environmental object."""
    pass


class NoLoadsOnStructure(Exception):
    """Raised when there are no loads induced on structure i.e. if point load is attached to structure."""
    pass


class LoadedPointRequiresRealEdges(Exception):
    """Raised when a loaded point has less than two real edges (2D) or three real edges (3D) attached."""
    pass


class NormalisationTypeError(Exception):
    """Raised when a normalisation type is given that the system can't interpret."""
    pass


class TooFewRealEdgesError(Exception):
    """Raised when too many edges have been made virtual in the environment."""
    pass


class StateVariableDoesNotExist(Exception):
    """Raised a state variable is ask for that does not exist."""
    pass

class SupportPointError(Exception):
    """Raised when a support point does not exist in the organism (seedling)"""
    pass

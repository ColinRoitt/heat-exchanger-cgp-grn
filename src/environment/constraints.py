import numpy as np

from abc import ABC


class Constraint(ABC):
    """Constraint class for the consolidation of multiple constraints.

    Attributes:
        min_constraint: Used to add a minimum constraint.
        max_constraint: Used to add a maximum constraint.
        default_constraint: Default constraint used to set the constraint violation.
    """
    def __init__(self, min_constraint=None, max_constraint=None):
        """Inits the Constraint object with given constraints."""
        self.min_constraint = min_constraint
        self.max_constraint = max_constraint
        self.default_violation = 0.0

    def get_violation(self, query):
        """Returns how much a constraint has been violated."""
        pass


class FailedDesignConstraint(Constraint):
    """Constraint on system when a design has failed.

    Attributes:
        min_constraint: Used to add a minimum constraint. Has no effect on this constraint object.
        max_constraint: Used to add a maximum constraint. Has no effect on this constraint object.
    """
    def __init__(self, min_constraint=None, max_constraint=None):
        super().__init__(min_constraint, max_constraint)
        """Inits the Constraint object with given constraints."""

    def get_violation(self, query):
        """Returns how much a constraint has been violated."""
        pass


class TotalStrainEnergyConstraint(Constraint):
    """Constraint on total strain energy of the organism.

    Attributes:
        min_constraint: Used to add a minimum constraint.
        max_constraint: Used to add a maximum constraint.
    """
    def __init__(self, min_constraint=None, max_constraint=None):
        super().__init__(min_constraint, max_constraint)
        """Inits the Constraint object with given constraints."""

    def get_violation(self, query):
        """Returns how much a constraint has been violated."""
        violation = self.default_violation

        if self.min_constraint is not None and query < self.min_constraint:
            violation = np.sum(np.where(self.min_constraint - query < 0, 0, self.min_constraint - query))
        elif self.max_constraint is not None and query > self.max_constraint:
            violation = np.sum(np.where(query - self.max_constraint < 0, 0, query - self.max_constraint))

        return violation


class TotalVolumeConstraint(Constraint):
    """Constraint on total volume of the organism.

    Attributes:
        min_constraint: Used to add a minimum constraint.
        max_constraint: Used to add a maximum constraint.
    """
    def __init__(self, min_constraint=None, max_constraint=None):
        super().__init__(min_constraint, max_constraint)
        """Inits the Constraint object with given constraints."""

    def get_violation(self, query):
        """Returns how much a constraint has been violated."""
        violation = self.default_violation

        if self.min_constraint is not None and query < self.min_constraint:
            violation = np.sum(np.where(self.min_constraint - query < 0, 0, self.min_constraint - query))
        elif self.max_constraint is not None and query > self.max_constraint:
            violation = np.sum(np.where(query - self.max_constraint < 0, 0, query - self.max_constraint))

        return violation


class TotalDeflectionConstraint(Constraint):
    """Constraint on total deflection of the organism.

    Attributes:
        min_constraint: Used to add a minimum constraint.
        max_constraint: Used to add a maximum constraint.
    """
    def __init__(self, min_constraint=None, max_constraint=None):
        super().__init__(min_constraint, max_constraint)
        """Inits the Constraint object with given constraints."""

    def get_violation(self, query):
        """Returns how much a constraint has been violated."""
        violation = self.default_violation

        if self.min_constraint is not None and query < self.min_constraint:
            violation = np.sum(np.where(self.min_constraint - query < 0, 0, self.min_constraint - query))
        elif self.max_constraint is not None and query > self.max_constraint:
            violation = np.sum(np.where(query - self.max_constraint < 0, 0, query - self.max_constraint))

        return violation


class MaxDeflectionConstraint(Constraint):
    """Constraint on max deflection of the organism.

    Attributes:
        min_constraint: Used to add a minimum constraint.
        max_constraint: Used to add a maximum constraint.
        direction: Direction in which to place the constraint. Options: 'x', 'y', 'z' or 'all'.
    """
    def __init__(self, min_constraint=None, max_constraint=None, direction="all"):
        super().__init__(min_constraint, max_constraint)
        """Inits the Constraint object with given constraints."""
        self.direction = direction
        self.check_direction()

    def check_direction(self):
        """Checks if the value assigned to direction variable is viable."""
        if self.direction != "x" or self.direction != "y" or self.direction != "z" or self.direction != "all":
            print("Direction type given can't be used. Options are 'x', 'y', 'z' or 'all'.")

    def get_violation(self, query):
        """Returns how much a constraint has been violated."""
        violation = self.default_violation

        if self.min_constraint is not None and query < self.min_constraint:
            violation = np.sum(np.where(self.min_constraint - query < 0, 0, self.min_constraint - query))
        elif self.max_constraint is not None and query > self.max_constraint:
            violation = np.sum(np.where(query - self.max_constraint < 0, 0, query - self.max_constraint))

        return violation


class ManLengthCostConstraint(Constraint):
    """Constraint on manufacturing length cost of the organism.

    Attributes:
        min_constraint: Used to add a minimum constraint.
        max_constraint: Used to add a maximum constraint.
    """
    def __init__(self, min_constraint=None, max_constraint=None):
        super().__init__(min_constraint, max_constraint)
        """Inits the Constraint object with given constraints."""

    def get_violation(self, query):
        """Returns how much a constraint has been violated."""
        pass


class TotalBucklingConstraint(Constraint):
    """Constraint on organism member buckling.

    Attributes:
        min_constraint: Used to add a minimum constraint.
        max_constraint: Used to add a maximum constraint.
        safety_factor: Safety factor for buckling - critical stress/member stress
    """
    def __init__(self, min_constraint=None, max_constraint=None, safety_factor = None):
        super().__init__(min_constraint, max_constraint)
        """Inits the Constraint object with given constraints."""
        self.safety_factor = safety_factor

    def get_violation(self, query):
        """Returns how much a constraint has been violated."""

        """
        Query is numpy array of each members buckling ratio - critical stress/member stress
        If this ratio is less than the safety factor, the violation returned is the num of members in violation
        """
        violation = self.default_violation

        if self.min_constraint is not None and query < self.min_constraint:
            violation = self.min_constraint - query
        elif self.max_constraint is not None and query > self.max_constraint:
            violation = query - self.max_constraint
        elif self.safety_factor is not None:
            violation = np.sum(np.count_nonzero(query >= self.safety_factor))

        return violation


class TotalYieldingConstraint(Constraint):
    """Constraint on organism member yielding.

    Attributes:
        min_constraint: Used to add a minimum constraint.
        max_constraint: Used to add a maximum constraint.
        safety_factor: Safety factor for buckling - critical stress/member stress
    """
    def __init__(self, min_constraint=None, max_constraint=None, safety_factor = None):
        super().__init__(min_constraint, max_constraint)
        """Inits the Constraint object with given constraints."""
        self.safety_factor = safety_factor

    def get_violation(self, query):
        """Returns how much a constraint has been violated."""

        """
        Query is numpy array of each members buckling ratio - critical stress/member stress
        If this ratio is less than the safety factor, the violation returned is the num of members in violation
        """
        violation = self.default_violation

        if self.min_constraint is not None and query < self.min_constraint:
            violation = self.min_constraint - query
        elif self.max_constraint is not None:
            violation = np.sum(np.count_nonzero(query >= self.max_constraint))
        elif self.safety_factor is not None:
            violation = np.sum(np.count_nonzero(query >= self.safety_factor))

        return violation    


class TotalBeamYieldingConstraint(Constraint):
    """Constraint on organism member yielding.

    Attributes:
        min_constraint: Used to add a minimum constraint.
        max_constraint: Used to add a maximum constraint.
        safety_factor: Safety factor for buckling - critical stress/member stress
    """
    def __init__(self, min_constraint=None, max_constraint=None, safety_factor = None):
        super().__init__(min_constraint, max_constraint)
        """Inits the Constraint object with given constraints."""
        self.safety_factor = safety_factor

    def get_violation(self, query):
        """Returns how much a constraint has been violated."""

        """
        Query is numpy array of each members buckling ratio - critical stress/member stress
        If this ratio is less than the safety factor, the violation returned is the num of members in violation
        """
        violation = self.default_violation

        if self.min_constraint is not None and query < self.min_constraint:
            violation = self.min_constraint - query
        elif self.max_constraint is not None:
            violation = np.sum(np.count_nonzero(query >= self.max_constraint))
        elif self.safety_factor is not None:
            violation = np.sum(np.count_nonzero(query >= self.safety_factor))

        return violation    

   
class MaxStressConstraint(Constraint):
    """Constraint on max stress of members of the organism.

    Attributes:
        min_constraint: Used to add a minimum constraint.
        max_constraint: Used to add a maximum constraint.

    Query is numpy array of each members stress
    The violation returned is the num of members in violation of maximum stress
    """
    def __init__(self, min_constraint=None, max_constraint=None):
        super().__init__(min_constraint, max_constraint)
        """Inits the Constraint object with given constraints."""

    def get_violation(self, query):
        """Returns how much a constraint has been violated."""
        violation = self.default_violation
        #query = np.max(query)

        if self.min_constraint is not None and np.any(query < self.min_constraint):
            violation = self.min_constraint - query
            violation = np.sum(np.where(self.min_constraint - query < 0, 0, self.min_constraint - query))
        elif self.max_constraint is not None and np.any(query > self.max_constraint):
            violation = np.sum(np.where(query - self.max_constraint < 0, 0, query - self.max_constraint))

        return violation


class VolumeFractionConstraint(Constraint):
    """Constraint on total volume fraction of the organism.
    
    Attributes:
        min_constraint: Used to add a minimum constraint.
        max_constraint: Used to add a maximum constraint.
    """
    def __init__(self, min_constraint=None, max_constraint=None):
        super().__init__(min_constraint, max_constraint)
        """Inits the Constraint object with given constraints."""

    def get_violation(self, query):
        """Returns how much a constraint has been violated."""
        violation = self.default_violation

        if self.min_constraint is not None and query < self.min_constraint:
            violation = np.sum(np.where(query > self.min_constraint, 0, abs(self.min_constraint - query)))
        elif self.max_constraint is not None and query > self.max_constraint:
            violation = np.sum(np.where(query < self.max_constraint, 0, abs(query - self.max_constraint)))
 
        return violation


class ChannelVolumeConstraint(Constraint):
    """Constraint on total volume of the organism channels.
    
    Attributes:
        min_constraint: Used to add a minimum constraint.
        max_constraint: Used to add a maximum constraint.
    """
    def __init__(self, min_constraint=None, max_constraint=None):
        super().__init__(min_constraint, max_constraint)
        """Inits the Constraint object with given constraints."""

    def get_violation(self, query):
        """Returns how much a constraint has been violated."""
        violation = self.default_violation

        if self.min_constraint is not None and query < self.min_constraint:
            violation = np.sum(np.where(query > self.min_constraint, 0, abs(self.min_constraint - query)))
        elif self.max_constraint is not None and query > self.max_constraint:
            violation = np.sum(np.where(query < self.max_constraint, 0, abs(query - self.max_constraint)))
 
        return violation        
    

class MaxTempConstraint(Constraint):
    """Constraint on max temperature of channels of the organism.
    
    Attributes:
        min_constraint: Used to add a minimum constraint.
        max_constraint: Used to add a maximum constraint.

    Query is numpy array of each channel temperatures
    The violation returned is the num of channels in violation of maximum temperature
    """
    def __init__(self, min_constraint=None, max_constraint=None):
        super().__init__(min_constraint, max_constraint)
        """Inits the Constraint object with given constraints."""

    def get_violation(self, query):
        """Returns how much a constraint has been violated."""
        violation = self.default_violation
        query = max(query)

        if self.min_constraint is not None and np.any(query < self.min_constraint):
            violation = self.min_constraint - query
            violation = np.sum(np.where(self.min_constraint - query < 0, 0, self.min_constraint - query))
        elif self.max_constraint is not None and np.any(query > self.max_constraint):
            violation = np.sum(np.where(query - self.max_constraint < 0, 0, query - self.max_constraint))

        return violation


class SeedlingConstraint(Constraint):
    """Constraint which restricts population so that the fitnesses do not exceed a percentage of the seedling
    fitness.

    Attributes:
        min_constraint: Used to add a minimum constraint. This is a percentage of the initial fitness (1.0 == 100%).
        max_constraint: Used to add a maximum constraint. This is a percentage of the initial fitness (1.0 == 100%).
    """
    def __init__(self, min_constraint=None, max_constraint=None):
        super().__init__(min_constraint, max_constraint)
        """Inits the Constraint object with given constraints."""

    def get_violation(self, query, init_fitness):
        """Returns how much a constraint has been violated."""
        violation = np.zeros((len(query),))
        if init_fitness is None:
            query_norm = np.ones((len(query),))
        else:
            query_norm = np.array(query) / init_fitness

        if self.min_constraint is not None:
            min_constraint = np.full((len(query),), self.min_constraint)
            violation = np.where(query_norm < min_constraint, min_constraint - query_norm, violation)
        if self.max_constraint is not None:
            max_constraint = np.full((len(query),), self.max_constraint)
            violation = np.where(query_norm > max_constraint, query_norm - max_constraint, violation)

        return violation

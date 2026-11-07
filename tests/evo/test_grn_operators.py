"""
Test for evo operators
"""
import random
import pytest
from src.evodevo.operators.rankandcrowding import RankAndCrowding


class Population:
    """
    Population class that holds a population of solutions/grns
    """
    def __init__(self):
        self.pop_size = 4
        self.solutions = [Solution() for i in range(self.pop_size)]
        self.rc = RankAndCrowding()


class Solution:
    """
    GRN class that holds the performance of GRNs such as fitness, constraints,
    and constraints violation (CV)
    """
    def __init__(self):
        self.mo_fitness = [random.random(), random.random()]
        self.CV = [0]

        self.data = {}

    def set(self, key, value):
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            self.data[key] = value
        return self

    def get(self, *keys):
        ret = []

        for key in keys:
            if hasattr(self, key):
                v = getattr(self, key)
            elif key in self.data:
                v = self.data[key]
            else:
                v = None

            ret.append(v)

        if len(ret) == 1:
            return ret[0]
        else:
            return tuple(ret)


@pytest.fixture(name="pop")
def population_fixture():
    """Fixture for population class object"""
    pop = Population()
    return pop


@pytest.fixture(name="nds")
def nds_fixture(pop):
    """Fixture for non-dominated sort (nds)"""
    from src.evodevo.operators.nds import nds
    gen = 0
    nds = nds(gen, pop.solutions, rc=pop.rc)
    assert len(nds) == pop.pop_size
    return nds


def test_ts_operator(pop, nds):
    """
    Confirms that number of parent_pairs are the same as number of offspring
    """
    from src.evodevo.operators.nds import get_parent_for_crossover
    n_matings = len(pop.solutions)
    parent_pairs = get_parent_for_crossover(pop.solutions, n_matings, n_parents=2)
    assert len(parent_pairs) == pop.pop_size



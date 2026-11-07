import numpy as np
from src.evodevo.operators.rankandcrowding import RankAndCrowding
from src.evodevo.operators.selection import TournamentSelection, binary_tournament

# rc = RankAndCrowding()
ts = TournamentSelection(func_comp=binary_tournament)


def nds(gen, pop, rc=None):
    """
    Non dominated sorting
    Args:
        gen: generation count to decide n_survival
        pop: actual population to compute rank of each genome
    Returns:
        nds: a list of genome index that should create next generation solutions/genomes
    """
    F = []
    for solution in pop:
        F.append(solution.mo_fitness)
    F = np.array(F)
    n_survive = int(len(pop) /2)
    if gen == 0:
        n_survive = len(pop)
    nds = rc.do(F, pop, n_survive=n_survive)
    # print(gen, nds)
    return nds


def get_parent_for_crossover(pop, n_matings, n_parents=2):
    """
    selection of parent pairs to do crossover and produce offspring
    Args:
        pop: actual population to compute rank of each genome
        n_matings: how many offspring should be generated
        n_parents: how many parents needed to generate a child
    Returns:
        parent_pairs: a list of parent pairs ==> each pair of parent will produce a child solution
    """
    parent_pairs = ts.do(pop, n_matings, n_parents=n_parents)
    return parent_pairs

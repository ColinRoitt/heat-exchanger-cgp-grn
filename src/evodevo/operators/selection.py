import math

import numpy as np

from pymoo.core.selection import Selection
from pymoo.util.misc import random_permuations
from pymoo.util.dominator import Dominator

# ---------------------------------------------------------------------------------------------------------
# Binary Tournament Selection Function
# ---------------------------------------------------------------------------------------------------------


def binary_tournament(pop, P, tournament_type = 'comp_by_dom_and_crowding'):
    """
   Binary tournament selection
   Args:
       pop: actual population to compute rank of each genome
       P: list of parent selected randomly to do tournament
       tournament_type: type of TS
   Returns:
       S: a list of genome pair for crossover and mutatiob
   """
    n_tournaments, n_parents = P.shape

    if n_parents != 2:
        raise ValueError("Only implemented for binary tournament!")

    # tournament_type = algorithm.tournament_type
    S = np.full(n_tournaments, np.nan)

    for i in range(n_tournaments):

        a, b = P[i, 0], P[i, 1]
        a_cv, a_f, b_cv, b_f = pop[a].CV[0], pop[a].mo_fitness, pop[b].CV[0], pop[b].mo_fitness

        rank_a, cd_a = pop[a].get("rank", "crowding")
        rank_b, cd_b = pop[b].get("rank", "crowding")
        # print(i, a_f, b_f, pop[a].mo_fitness, pop[b].mo_fitness)

        # if at least one solution is infeasible
        if a_cv > 0.0 or b_cv > 0.0:
            S[i] = compare(a, a_cv, b, b_cv, method='smaller_is_better', return_random_if_equal=True)

        # both solutions are feasible
        else:

            if tournament_type == 'comp_by_dom_and_crowding':
                rel = Dominator.get_relation(a_f, b_f)
                if rel == 1:
                    S[i] = a
                elif rel == -1:
                    S[i] = b

            elif tournament_type == 'comp_by_rank_and_crowding':
                S[i] = compare(a, rank_a, b, rank_b, method='smaller_is_better')

            else:
                raise Exception("Unknown tournament type.")

            # if rank or domination relation didn't make a decision compare by crowding
            if np.isnan(S[i]):
                S[i] = compare(a, cd_a, b, cd_b, method='larger_is_better', return_random_if_equal=True)

    return S[:, None].astype(int, copy=False)

class TournamentSelection:
    """
      The Tournament selection is used to simulated a tournament between individuals. The pressure balances
      greedy the genetic algorithm will be.
    """

    def __init__(self, func_comp=None, pressure=2, **kwargs):
        """
        Parameters
        Args:
            func_comp: The function to compare two individuals. It has the shape: comp(pop, indices) and returns the winner.
            If the function is None it is assumed the population is sorted by a criterium and only indices are compared.
            pressure: The selection pressure to bie applied. Default it is a binary tournament.
        """

        # selection pressure to be applied
        self.pressure = pressure

        self.func_comp = func_comp
        if self.func_comp is None:
            raise Exception("Please provide the comparing function for the tournament selection!")

    def do(self, pop, n_select, n_parents=1, **kwargs):
        """
        Actual TS happens here
       Args:
           pop: actual population to compute rank of each genome
           n_select: how many to generate
           n_parents: how many parents needed
       Returns:
           S: returns the results of binary tournament
       """
        # number of random individuals needed
        n_random = n_select * n_parents * self.pressure

        # number of permutations needed
        n_perms = math.ceil(n_random / len(pop))

        # get random permutations and reshape them
        P = random_permuations(n_perms, len(pop))[:n_random]
        P = np.reshape(P, (n_select * n_parents, self.pressure))

        # compare using tournament function
        S = self.func_comp(pop, P, **kwargs)

        return np.reshape(S, (n_select, n_parents))


def compare(a, a_val, b, b_val, method, return_random_if_equal=False):
    """
    Actual TS happens here
    Args:
        a: first parent solution
        a_val: fitness of the first parent solution
        b: second parent solution
        b_val: fitness of the second parent solution
        method: method to compare ==> different for minimization and maximization
        return_random_if_equal: if a_val and b_val are the same then return random
    Returns:
        a or b depending on method
   """
    if method == 'larger_is_better':
        if a_val > b_val:
            return a
        elif a_val < b_val:
            return b
        else:
            if return_random_if_equal:
                return np.random.choice([a, b])
            else:
                return None
    elif method == 'smaller_is_better':
        if a_val < b_val:
            return a
        elif a_val > b_val:
            return b
        else:
            if return_random_if_equal:
                return np.random.choice([a, b])
            else:
                return None
    else:
        raise Exception("Unknown method.")

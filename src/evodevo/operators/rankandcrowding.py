import numpy as np
import os
from pymoo.util.randomized_argsort import randomized_argsort
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.core.survival import Survival, split_by_feasibility
from pymoo.core.population import Population
from src.evodevo.operators.metrics import get_crowding_function
from src.evodevo.operators.rc import save_pareto_fitness


class RankAndCrowding(Survival):
    """
    Assign rank and crowding distance to each solution in the given population
    """

    def __init__(self, nds=None, crowding_func="cd", path=None):
        crowding_func_ = get_crowding_function(crowding_func)

        super().__init__(filter_infeasible=True)
        self.nds = nds if nds is not None else NonDominatedSorting()
        self.crowding_func = crowding_func_
        self.path = path
        self.gen = 0

    def do(self, F, pop, *args, n_survive=None, **kwargs):
        """
        Finds pareto front and assign crowding distance and rank to each genome
        Args:
            F: fitness of genomes
            pop: actual population to compute rank of each genome
            n_survive: how many genomes to survive (typically equal  to population size)
        Returns:
            survivors: a list of genome index that survive in the current generation
        """
        # the final indices of surviving individuals
        survivors = []

        # do the non-dominated sorting until splitting front
        fronts = self.nds.do(F, n_stop_if_ranked=n_survive)

        for k, front in enumerate(fronts):

            I = np.arange(len(front))

            # current front sorted by crowding distance if splitting
            if len(survivors) + len(I) > n_survive:

                # Define how many will be removed
                n_remove = len(survivors) + len(front) - n_survive

                # re-calculate the crowding distance of the front
                crowding_of_front = \
                    self.crowding_func.do(
                        F[front, :],
                        n_remove=n_remove
                    )

                I = randomized_argsort(crowding_of_front, order='descending', method='numpy')
                I = I[:-n_remove]

            # otherwise take the whole front unsorted
            else:
                # calculate the crowding distance of the front
                crowding_of_front = \
                    self.crowding_func.do(
                        F[front, :],
                        n_remove=0
                    )

            # save rank and crowding in the individual class
            for j, i in enumerate(front):
                pop[i].set("rank", k)
                pop[i].set("crowding", crowding_of_front[j])

            # extend the survivors by all or selected individuals
            survivors.extend(front[I])

        pf = None
        if len(fronts) > 1:
            # print('More than one front')
            # print(fronts, fronts[0])
            pf = fronts[0]
        else:
            # print('One front with more than N individuals')
            # print(len(fronts), fronts)
            # print(survivors)
            pf = survivors
        pf = [F[s] for s in pf]
        if self.path is not None:
            pareto_filename = os.path.join(self.path, "pareto.csv")
            save_pareto_fitness(np.array(pf), pareto_filename, self.gen)
        self.gen += 1
        return survivors

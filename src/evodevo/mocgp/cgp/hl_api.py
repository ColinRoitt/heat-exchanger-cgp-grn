from typing import Callable, Optional

import numpy as np

from .ea import MuPlusLambda
from .individual import IndividualBase
from .population import Population


def evolve(
    objective: Callable[[IndividualBase], IndividualBase],
    pop: Population = Population(),
    ea: MuPlusLambda = MuPlusLambda(),
    termination_fitness: float = np.inf,
    max_generations: int = np.iinfo(np.int64).max,
    max_objective_calls: int = np.iinfo(np.int64).max,
    print_progress: Optional[bool] = False,
    callback: Optional[Callable[[Population], None]] = None,
) -> Population:
    """
    Evolves a population and returns the history of fitness of parents.

    Parameters
    ----------
    objective : Callable
        An objective function used for the evolution. Needs to take an
        individual (Individual) as input parameter and return
        a modified individual (with updated fitness).
    pop : Population, optional
        A population class that will be evolved. Defaults to population with default parameters.
    ea : EA algorithm instance, optional
        The evolutionary algorithm. Defaults to MuPlusLambda.
    termination_fitness : float, optional
        Minimum fitness at which the evolution is terminated. Defaults to positive infinity.
    max_generations : int, optional
        Maximum number of generations.
        If neither this nor `max_objective_calls` are set, `max_generations` defaults to 1000.
    max_objective_calls: int, optional
        Maximum number of function evaluations.
        Defaults to largest representable integer.
        If neither this nor `max_generations` are set, `max_generations` defaults to 1000.
    print_progress : boolean, optional
        Switch to print out the progress of the algorithm. Defaults to False.
    callback :  callable, optional
        Called after each iteration with the population instance.
        Defaults to None.

    Returns
    -------
    Population
        The evolved population.
    """
    if max_generations == np.iinfo(np.int64).max and max_objective_calls == np.iinfo(np.int64).max:
        max_generations = 50

    # perform evolution
    max_fitness = np.finfo(float).max
    # Main loop: -1 offset since the last loop iteration will still increase generation by one
    gen: int = 0
    pop.generation = gen

    while pop.generation < max_generations+1 and ea.n_objective_calls < max_objective_calls:

        if gen == 0:
            pop._parents = ea._moo_compute_fitness(pop.parents, objective, use_hurdles=False)
            nds = ea.nds(gen, pop.parents, rc=ea.rc)

        gen += 1
        pop.generation = gen
        pop = ea.moo_step(pop, objective, pop.generation)

        assert len(pop.parents) == pop.n_parents

        if print_progress:
            if max_generations < np.iinfo(np.int64).max:
                print(
                    f"\r[{pop.generation}/{max_generations}] max fitness: {max_fitness}\033[K",
                    end="",
                    flush=True,
                )
            else:
                assert False  # should never be reached

        if callback is not None:
            callback(pop)

    if print_progress:
        print()

    return pop

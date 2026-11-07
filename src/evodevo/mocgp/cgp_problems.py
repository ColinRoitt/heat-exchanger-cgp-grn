from src.evodevo.mocgp.cgp.cartesian_graph import CartesianGraph
from cgp_graph import CGPGraph

# CGP code modified from hal-cgp library. Schmidt, Maximilian & Jordan, Jakob (2020) hal-cgp: Cartesian genetic programming in pure Python. 10.5281/zenodo.3889163

class CGPProblem:
    """
    problem class that evaluates genomes fitness
    """
    def __init__(self, devo, seedling, env, run_dir):
        self.devo = devo
        self.seedling = seedling
        self.env = env
        self.run_dir = run_dir
        self.gen_id = 0

    def _evaluate(self, ind):
        """
        This is where actual fitness evaluation happens by running devo

        Args:
            ind: the genome, used to generate grn (a grpah in this case)

        Returns:
            ind: the genome with assigned fitness and constraint
        """
        grn = CGPGraph(ind.gen, ind.pop)
        grn.set_grn(ind.to_numpy())
        grn.graph = CartesianGraph(ind.genome)
        grn.pretty = grn.graph.pretty_str()
        fit, constraint = self.devo(grn=grn, seedling=self.seedling, environment=self.env)
        ind._fitness = fit
        ind.mo_fitness = fit
        print('=================', ind.gen, ind.pop, ind._fitness, ind.idx, '===================')
        grn.save_grn(self.run_dir, fit, constraint)
        return ind

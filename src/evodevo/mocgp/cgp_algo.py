import functools

from src.evodevo.evodevo_parent import EvoDevoAlgo
from src.evodevo.mocgp.cgp_problems import CGPProblem
from src.evodevo.mocgp.cgp import Add, Sub, Mul, Div, IfElse
from src.evodevo.mocgp.cgp.CustomPrimitives import SineFunction, TanFunction, SigmoidFunction, ReluFunction
from src.evodevo.mocgp.cgp import Population, evolve
from src.evodevo.mocgp.cgp.ea import MuPlusLambda


class CGP(EvoDevoAlgo):
    """
    CGP child class that inherit evodevo parent class (EvoDevoAlgo)
    """
    def __init__(self, generations, pop_size, devo_steps, self_stopping=False):
        super().__init__(generations, pop_size, devo_steps, self_stopping=self_stopping)
        self.algo = None
        self.problem = None
        self.genome_parameters = {}
        self.ea_params = None
        self.evolve_params = None

    def initialize_evo(self, env, seedling):
        """
        Initialize evo algorithm parameters and setup the problem

        Args:
            env: environment object
            seedling: organism object
        Returns:
            None
        """
        self.initialize_parameters(seedling, env)
        self.problem = self.setup_problem(seedling, env)

    def setup_problem(self, seedling, env):
        """
        Set up the problem

        Args:
            env: environment object
            seedling: organism object
        Returns:
            problem: the CGPproblem object

        """
        problem = CGPProblem(self.devo, seedling, env, self.run_dir)
        return problem

    def initialize_parameters(self, seedling, env):
        """
        Initialize the parameters needed for CGP to run

        Args:
            env: environment object
            seedling: organism object
        Returns:
            None

        """
        # make an object of custom problem
        num_inputs = seedling.get_number_of_network_inputs(env)
        num_outputs = seedling.get_number_of_network_outputs()
        # ADD STOPPING INPUTS AND OUTPUTS AS NEEDED
        if self.self_stopping:
            num_inputs += 1
            num_outputs += 1

        n_criteria = seedling.get_number_of_fitness_criteria(env)
        n_constr = seedling.get_number_of_constraint_criteria(env)

        self.genome_parameters = {
            "n_inputs": num_inputs,
            "n_outputs": num_outputs,
            "n_columns": num_inputs,
            "n_rows": num_outputs,  # num_outputs,
            # "primitives": (Add, Sub, Mul, Div, IfElse, SineFunction)
            "primitives": (Add, Mul, Sub, Div, IfElse, TanFunction, SigmoidFunction, ReluFunction)
        }

        self.ea_params = {"n_processes": 1, "n_offsprings": self.pop_size}

        self.evolve_params = {"max_generations": self.gen}

    def evo_devo(self, env, seedling, run_dir=None):
        """Run EvoDevo algorithm.

        Args:
            env: environment object
            seedling: organism object
            run_dir: directory to save files

        Returns:
            res: results from evo
        """
        self.run_dir = run_dir
        self.make_run_directory()
        self.write_experiment_csv(env, seedling)
        seedling.save_seedling(self.run_dir)
        self.initialize_evo(env, seedling)
        res = self.minimize()
        return res

    def minimize(self):
        """
        Initialize the population, and run evo for N generations.

        Returns:
            res: results from evo
        """
        print(self.evolve_params )
        pop = Population(n_parents=self.pop_size, genome_params=self.genome_parameters)
        ea = MuPlusLambda(self.ea_params)
        ea.set_rc(self.run_dir)
        ea.n_offsprings = self.pop_size

        obj = functools.partial(
            self.problem._evaluate
        )
        pop = evolve(
            obj, pop, ea, print_progress=True, **self.evolve_params)
        res = pop.pareto_fronts
        return res

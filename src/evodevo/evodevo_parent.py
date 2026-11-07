import os
import csv
import numpy as np

from datetime import datetime
from abc import abstractmethod,ABC
from copy import deepcopy
from src.devo.geometry import Point, Line
from src.devo.topology import Edge


class EvoDevoAlgo(ABC):
    """
    This is the base EvoDevo parent class.
    All child classes will inherit this parent class and
    will override some methods.
    """
    def __init__(self, generations, pop_size, devo_steps, random_seed=2023, self_stopping=False):
        self.gen = generations
        self.pop_size = pop_size
        self.devo_steps = devo_steps
        self.random_seed = random_seed
        self.run_dir = None
        self.self_stopping = self_stopping

    def devo(self, grn, seedling, environment):
        """Devo loop where the actual growth happens.

        Args:
            grn: GRN encoded from specific genome.
            seedling: Organism object.
            environment: Environment object.

        Returns:
            fitness: Fitness of the genome/grn.
            constraint: Constraint of the genome/grn.
        """
        env = deepcopy(environment)
        organism = deepcopy(seedling)
        organism.generation_id = grn.gen_id
        organism.population_id = grn.pop_id

        Point.set_instance_count(len(organism.points))
        Line.set_instance_count(len(organism.lines))
        Edge.set_instance_count(len(organism.edges))

        fitness, constraints = organism.get_state(env, self.run_dir)
        
        self.write_fitness_csv(fitness, environment, organism)
        organism.save_organism(self.run_dir, environment, True)

        organism.clock += 1
        # Using a dictionary here has the length of cells seems to change and this can handle that more easily
        halt_values = {1: -1}
        halt_threshold = 0

        while not self.should_stop(organism.clock, self.devo_steps, fitness, halt_values, halt_threshold):

            cells = organism.get_network_inputs(env)

            outputs = []

            if self.self_stopping:
                for i, cell in enumerate(cells):
                    prev_halt = halt_values.get(i, -1)
                    # APPEND FOR RECURRENCY
                    out = grn.activate(np.append(cell, prev_halt))
                    # out = grn.activate(cell)
                    halt_values[i] = out[-1]
                    output = out[:-1]
                    outputs.append(output)
            else:
                for cell in cells:
                    outputs.append(grn.activate(cell))


            organism.update_with_network_outputs(outputs, env)
            fitness, constraints = organism.get_state(env, self.run_dir)
            self.write_fitness_csv(fitness, environment, organism)
            # save halting values
            if self.self_stopping:
                self.write_halting_csv(halt_values, organism)

            organism.save_organism(self.run_dir, environment)

            if organism.make_stl:
                organism.write_stl(self.run_dir, save_step=False)

            organism.clock += 1
        
        # print(organism.clock, ' out of ', self.devo_steps)

        for element in environment:
            element.remove_analysis_files(self.run_dir)

        return fitness, constraints

    def should_stop(self, clock, devo_steps_max, fitness, halt_values, halt_threshold):
        if clock >= devo_steps_max:
            # print("Devo steps max reached")
            return True
        if fitness[0] == 1e10:
            # print("Constraint reached")
            return True
        # has_halted = any(value > halt_threshold for value in halt_values.values())
        # stop if majority are above threshold
        votes_for = sum(value > halt_threshold for value in halt_values.values())
        has_halted = votes_for / len(halt_values) > 0.7
        if self.self_stopping and has_halted:
            # print("Self stopping")
            return True
        # print("Continue")
        return False
        
    @abstractmethod
    def evo_devo(self, env, seedling, run_dir=None):
        """Used for a EvoDevo run using a specific environment and seedling.

        Args:
            env: Environment object.
            seedling: Organism object.
            run_dir: Directory to save files

        Returns:
            None
        """
        pass

    def make_run_directory(self):
        """Generates the file structure to save data for the EvoDevo run.

        Returns:
            None
        """
        project_dir_path = os.getcwd()
        if project_dir_path.endswith("src"):
            project_dir_path = project_dir_path[:-len("src")]

        data_dir_path = os.path.join(project_dir_path, "data")
        if not os.path.isdir(data_dir_path):
            os.mkdir(data_dir_path)

        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y-%H-%M-%S")

        if self.run_dir is None:
            run_dir_name = f"evo-{self.gen}-pop-{self.pop_size}-devo-{self.devo_steps}-{dt_string}"
            self.run_dir = os.path.join(data_dir_path, run_dir_name)

        if not os.path.isdir(self.run_dir):
            os.mkdir(self.run_dir)

        grn_dir_path = os.path.join(self.run_dir, "grns")
        if not os.path.isdir(grn_dir_path):
            os.mkdir(grn_dir_path)

        calculix_dir_path = os.path.join(self.run_dir, "calculix")
        if not os.path.isdir(calculix_dir_path):
            os.mkdir(calculix_dir_path)

        model_dir_path = os.path.join(self.run_dir, "cad_models")
        if not os.path.isdir(model_dir_path):
            os.mkdir(model_dir_path)

    def write_experiment_csv(self, environment, seedling):
        """Write experiment information to a csv file.

        Returns:
            None
        """
        file_path = os.path.join(self.run_dir, "experiment_info.csv")

        with open(file_path, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["num_evo", self.gen])
            writer.writerow(["num_pop", self.pop_size])
            writer.writerow(["num_devo", self.devo_steps])
            writer.writerow(["random_seed", self.random_seed])
            writer.writerow(["num_inputs", seedling.get_number_of_network_inputs(environment)])
            writer.writerow(["num_outputs", seedling.get_number_of_network_outputs()])

            input_count = 0
            for element in environment:
                for label in element.fitness_labels:
                    writer.writerow([f"network_input_{input_count}", label])
                    input_count += 1

            for i, net_output in enumerate(seedling.network_outputs):
                writer.writerow([f"network_output_{i}", net_output.csv_name])

    def write_fitness_csv(self, fitness, environment, org):
        """Write fitnesses to a csv file.

        Args:
            fitness: List of fitness values.
            environment: An Environment object.
            org: An Organism object.

        Returns:
            None
        """
        fitness_criteria = []
        for element in environment:
            fitness_criteria.extend(element.fitness_labels)

        file_path = os.path.join(self.run_dir, "fitness.csv")
        with open(file_path, 'a', newline='') as csv_file:
            writer = csv.writer(csv_file)

            if (org.generation_id == 0) and (org.population_id == 0) and (org.clock == 0):
                first_line = ["Generation", "Population", "Iteration"]
                first_line.extend(fitness_criteria)
                writer.writerow(first_line)

            a1 = np.array([org.generation_id, org.population_id, org.clock])
            a2 = np.array(fitness).flatten()
            a3 = np.concatenate((a1, a2))
            writer.writerow(iter(a3))

    def write_halting_csv(self, halt, org):
        """Write halting values to a csv file.

        Args:
            halt: Dictionary of halting values for each cell in a devo step.
            org: An Organism object.

        Returns:
            None
        """
        file_path = os.path.join(self.run_dir, "halting_values.csv")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'a', newline='') as csv_file:
            writer = csv.writer(csv_file)

            if (org.generation_id == 0) and (org.population_id == 0) and (org.clock == 0):
                first_line = ["Generation", "Population", "Iteration", "Halt"]
                writer.writerow(first_line)

            a1 = np.array([org.generation_id, org.population_id, org.clock])
            a2 = np.array(list(halt.values())).flatten()
            a3 = np.concatenate((a1, a2))
            writer.writerow(iter(a3))

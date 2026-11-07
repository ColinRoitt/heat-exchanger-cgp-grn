import numpy as np
import functools

class Selection:
    def __int__(self):
        pass

    def dominates(self, fitnesses_1, fitnesses_2):
        # fitnesses_1 is an array of objectives of solution 1 [objective1, objective2 ...]
        larger_or_equal = fitnesses_1 <= fitnesses_2
        larger = fitnesses_1 < fitnesses_2
        if np.all(larger_or_equal) and np.any(larger):
            return True
        return False

    def calculate_pareto_fronts(self, fitnesses):
        # Calculate dominated set for each individual
        domination_sets = []
        domination_counts = []
        for fitnesses_1 in fitnesses:
            current_dimination_set = set()
            domination_counts.append(0)
            for i, fitnesses_2 in enumerate(fitnesses):
                if self.dominates(fitnesses_1, fitnesses_2):
                    current_dimination_set.add(i)
                elif self.dominates(fitnesses_2, fitnesses_1):
                    domination_counts[-1] += 1

            domination_sets.append(current_dimination_set)

        domination_counts = np.array(domination_counts)
        fronts = []
        while True:
            current_front = np.where(domination_counts == 0)[0]
            if len(current_front) == 0:
                # print("Done")
                break
            # print("Front: ",current_front)
            fronts.append(current_front)

            for individual in current_front:
                domination_counts[
                    individual] = -1  # this individual is already accounted for, make it -1 so  ==0 will not find it anymore
                dominated_by_current_set = domination_sets[individual]
                for dominated_by_current in dominated_by_current_set:
                    domination_counts[dominated_by_current] -= 1

        return fronts

    def calculate_crowding_metrics(self, fitnesses, fronts):
        num_objectives = fitnesses.shape[1]
        num_individuals = fitnesses.shape[0]

        # Normalise each objectives, so they are in the range [0,1]
        # This is necessary, so each objective's contribution have the same magnitude to the crowding metric.
        fitnesses_min = np.min(fitnesses, axis=0)
        fitnesses_max = np.max(fitnesses, axis=0)
        a = fitnesses - fitnesses_min
        b = fitnesses_max - fitnesses_min
        fitnesses = np.divide(a, b, out=np.zeros_like(a), where=b != 0)

        crowding_metrics = np.zeros(num_individuals)

        for front in fronts:
            for objective_i in range(num_objectives):
                sorted_front = sorted(front, key=lambda x: fitnesses[x, objective_i])

                crowding_metrics[sorted_front[0]] = np.inf
                crowding_metrics[sorted_front[-1]] = np.inf
                if len(sorted_front) > 2:
                    for i in range(1, len(sorted_front) - 1):
                        crowding_metrics[sorted_front[i]] += fitnesses[sorted_front[i + 1], objective_i] - fitnesses[
                            sorted_front[i - 1], objective_i]

        return crowding_metrics

    def fronts_to_nondomination_rank(self, fronts):
        nondomination_rank_dict = {}
        for i, front in enumerate(fronts):
            for x in front:
                nondomination_rank_dict[x] = i
            # print(nondomination_rank_dict)
        return nondomination_rank_dict

    def nondominated_sort(self, nondomination_rank_dict, crowding):
        num_individuals = len(crowding)
        indices = list(range(num_individuals))

        def nondominated_compare(a, b):
            # returns 1 if a dominates b, or if they equal, but a is less crowded
            # return -1 if b dominates a, or if they equal, but b is less crowded
            # returns 0 if they are equal in every sense

            if nondomination_rank_dict[a] > nondomination_rank_dict[b]:  # domination rank, smaller better
                return -1
            elif nondomination_rank_dict[a] < nondomination_rank_dict[b]:
                return 1
            else:
                if crowding[a] < crowding[b]:  # crowding metrics, larger better
                    return -1
                elif crowding[a] > crowding[b]:
                    return 1
                else:
                    return 0

        non_dominated_sorted_indices = sorted(indices, key=functools.cmp_to_key(nondominated_compare),
                                              reverse=True)  # decreasing order, the best is the first
        return non_dominated_sorted_indices

    def crowd_and_non_dominated_indices(self, fitnesses):
        fronts = self.calculate_pareto_fronts(fitnesses)
        # print(fronts)
        nondomination_rank_dict = self.fronts_to_nondomination_rank(fronts)
        crowding = self.calculate_crowding_metrics(fitnesses, fronts)

        # Sort the population
        non_dominated_sorted_indices = self.nondominated_sort(nondomination_rank_dict, crowding)

        return crowding, non_dominated_sorted_indices, nondomination_rank_dict




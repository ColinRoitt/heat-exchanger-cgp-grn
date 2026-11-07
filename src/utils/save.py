import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from pymoo.indicators.hv import Hypervolume


def write_csv(file_path, generation_id, population_id, iteration_id, data):
    """Save data as a flat csv file.

    Args:
        file_path: File path of csv file.
        data: Data to save to csv.

    Returns:
        None
    """
    with open(file_path, 'a', newline='') as csv_file:
        writer = csv.writer(csv_file)

        a1 = np.array([generation_id, population_id, iteration_id])
        a2 = data.flatten()
        a3 = np.concatenate((a1, a2))
        writer.writerow(iter(a3))


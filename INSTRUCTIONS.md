# Running the experiments

This code evolves fin growth on a heated curved plate, using a Cartesian Genetic
Programming (CGP) gene regulatory network. Each design is simulated with CalculiX.

## Requirements

- Linux
- [conda](https://docs.conda.io/en/latest/miniconda.html)
- **CalculiX `ccx` version 2.17**, available on your `PATH` as `ccx`

## Setup

From the repository root:

```
conda env create -f environment.yml
conda activate fin_growth
export PYTHONPATH=$PWD
```

## Running

Params are set at the bottom of the experiment file.

```python
main(['-p', '30', '-e', '50', '-d', '10', '-algo', 'CGP', '--self_stopping'])  # self-stopping
# main(['-p', '30', '-e', '50', '-d', '10', '-algo', 'CGP'])                   # fixed steps
```

Then run:

```
python experiment_curve_plate_LSTM.py
```

| Flag | Meaning |
|---|---|
| `-p` | Population size |
| `-e` | Number of generations |
| `-d` | Development limit (`10` gives 9 growth updates and 10 simulations, including the seedling) |
| `--self_stopping` | Adds the recurrent stop input and output to the GRN |

## Output

Each run writes a folder in `data/`:

| File | Contents |
|---|---|
| `fitness.csv` | Temperature, volume and devo steps for every individual at every step |
| `pareto.csv` | Surviving population at each generation |
| `halting_values.csv` | Per-cell stop value at each step (self-stopping runs only) |
| `network_inputs/heat_flux.csv` | Per-cell heat flux at each step |
| `grns/` | Saved genomes |


Loading a genome saved by the runs reported in the paper also needs PyTorch
(`pip install torch`), because those pickles reference it. Running the experiments
does not.

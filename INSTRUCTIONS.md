# Running the experiments

This code evolves fin growth on a heated curved plate, using a Cartesian Genetic
Programming (CGP) gene regulatory network. Each design is simulated with CalculiX.

## Requirements

- Linux
- [conda](https://docs.conda.io/en/latest/miniconda.html)
- **CalculiX `ccx` version 2.17**, available on your `PATH` as `ccx`

The CalculiX version matters. Version 2.22 changes the format of its `.dat` output
(`for setALLELEMENTS` instead of `for set ALLELEMENTS`), which the results parser
cannot read. Check your version with `ccx -v`.

## Setup

From the repository root:

```
conda env create -f environment.yml
conda activate fin_growth
export PYTHONPATH=$PWD
```

## Running

The stopping condition is chosen by hand at the bottom of
`experiment_curve_plate_LSTM.py`. Leave one `main(...)` line active:

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

A full run takes about 14 hours on one core. For a quick check, use a small
population and few generations, for example `-p 4 -e 2 -d 10`.

If you start several runs at once, start them at least a second apart. Run folders
are named by timestamp, and two runs starting in the same second will collide.

## Output

Each run writes a folder in `data/`:

| File | Contents |
|---|---|
| `fitness.csv` | Temperature, volume and devo steps for every individual at every step |
| `pareto.csv` | Surviving population at each generation |
| `halting_values.csv` | Per-cell stop value at each step (self-stopping runs only) |
| `network_inputs/heat_flux.csv` | Per-cell heat flux at each step |
| `grns/` | Saved genomes |

Two details when reading these files:

- The GRN receives the natural log of the heat flux. `heat_flux.csv` logs the raw value.
- In `halting_values.csv`, the first two cell columns are in swapped order (cell 1, then cell 0).

Loading a genome saved by the runs reported in the paper also needs PyTorch
(`pip install torch`), because those pickles reference it. Running the experiments
does not.

## Tests

```
pytest tests
```

The smoke tests in `tests/smoke` need `ccx` on your `PATH`.

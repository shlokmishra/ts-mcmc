# ts-mcmc

MCMC for phylogenetic trees using tskit for efficient storage. Implements Kingman coalescent prior with Jukes-Cantor sequence likelihood, plus Stein thinning for post-processing MCMC chains.

## Core Files

| File | Description |
|------|-------------|
| `tree.py` | Tree data structure, Kingman coalescent prior, Jukes-Cantor likelihood (Felsenstein's pruning), gradients |
| `mcmc.py` | MCMC loop with SPR moves, time resampling, mutation rate inference |
| `recorder.py` | Efficient storage of MCMC samples into tskit TreeSequence |

## Stein Thinning (`stein_thinning_trees/`)

Post-hoc thinning to select representative samples minimizing Kernel Stein Discrepancy.

| File | Description |
|------|-------------|
| `distance.py` | Tree distance metrics (KC distance, node-time distance) |
| `kernel.py` | Gaussian/IMQ kernels, Stein kernel for trees |
| `stein.py` | KSD computation, `TreeSteinDiscrepancy` class |
| `thinning.py` | Greedy thinning algorithm, `thin_trees()`, `compare_thinning_methods()` |

## Testing & Validation (`testing/`)

| Notebook/File | What it tests |
|---------------|---------------|
| `test_likelihood.ipynb` | Likelihood function verification |
| `test_mcmc.ipynb` | MCMC convergence diagnostics (R-hat, ESS, trace plots) |
| `test_stein_thinning.ipynb` | Stein vs naive thinning comparison with visualizations |
| `test_stein_thinning.py` | Unit tests for Stein thinning (run with `pytest`) |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run MCMC tests
cd testing && jupyter notebook test_mcmc.ipynb

# Run Stein thinning tests
pytest testing/test_stein_thinning.py -v

# Demo Stein thinning
cd testing && jupyter notebook test_stein_thinning.ipynb
```

## Usage Example

```python
from tree import coalescence_tree_with_sequences
from recorder import Recorder
from mcmc import kingman_mcmc
import stein_thinning_trees as stt

# Generate data and run MCMC
tree, seqs = coalescence_tree_with_sequences(sample_size=6, n_states=2, seq_length=50, mutation_rate=1.0)
tree.sequences = seqs
recorder = Recorder(6, 50)
kingman_mcmc(tree, recorder, pi=[0.5, 0.5], steps=1000)

# Apply Stein thinning
indices = stt.thin_trees(recorder, n_points=100)
thinned_rates = [recorder.mutation_rates[i] for i in indices]
```

## Key References

- Riabiz et al. (2022) - Optimal thinning of MCMC output
- Kelleher et al. (2016) - tskit for tree sequence storage

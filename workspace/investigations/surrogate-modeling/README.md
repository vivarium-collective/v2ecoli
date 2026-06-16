# Surrogate Modeling — a neural-network emulator of the v2ecoli baseline

This investigation demonstrates how to build a **neural-network surrogate** of
the v2ecoli whole-cell model using
[`pbg-torch`](https://github.com/vivarium-collective/pbg-torch), and evaluates
how faithfully — and how much faster — the surrogate reproduces baseline
behavior.

A surrogate is a learned, drop-in process-bigraph `Process` that approximates
the per-step dynamics of another `Process`/`Composite` over a curated panel of
observables. Once trained it rolls out autoregressively at a tiny fraction of
the simulator's cost.

## How a surrogate is made (the pipeline)

The whole pipeline lives in [`code/`](code/) and is built on `pbg-torch`'s
generic surrogate machinery:

1. **Observable panel** (`observables.py`). The baseline state is deeply nested
   (pint Quantities, per-agent keys, dict-valued exchange fluxes,
   high-dimensional count/flux vectors). `PanelLayout` discovers a stable,
   serializable column ordering and flattens each step into a single float
   vector. The **broad / high-dimensional** panel spans every category:

   | group | source | dim |
   |---|---|---|
   | growth / mass | `listeners.mass.*` | 7 |
   | exchange fluxes | `environment.exchange` | 87 |
   | metabolic fluxes | `listeners.fba_results.base_reaction_fluxes` | 2820 |
   | protein abundance | `listeners.monomer_counts` | 4309 |
   | transcript abundance | `listeners.rna_counts.mRNA_cistron_counts` | 4345 |
   | chromosome state | `listeners.replication_data.number_of_oric` | 1 |

   (~11.5k observables total.)

2. **Sample the target** (`sample_baseline.py`). Roll out an ensemble of
   baseline trajectories (multiple seeds), extracting the observable vector
   before and after each 1-second step to form `(state_t, state_{t+1})`
   transition pairs — a `pbg_torch.TransitionDataset`.

3. **Train** (`train_surrogate.py`). One whole trajectory is held out for test;
   `pbg_torch.train_surrogate` fits a residual-MLP that predicts the normalized
   per-step delta of the panel. The checkpoint is self-contained
   (weights + spec + normalization).

4. **Evaluate + profile** (`evaluate_surrogate.py`). On the held-out trajectory,
   report one-step R²/RMSE per observable group; roll the surrogate out
   autoregressively for the scalar observables; and benchmark surrogate
   inference speed against the full simulator.

5. **Figures** (`make_figures.py`). Interactive Plotly report (per-group
   fidelity, rollout overlays, speedup).

## Studies

- **sm-00-data-collection** — build the baseline transition-dataset ensemble.
- **sm-01-nn-surrogate** — train, evaluate fidelity per category, and profile
  the speedup.

## Scope and honesty

This is a **demonstration-grade** build trained on a modest *local* ensemble.
The low-dimensional, slowly-varying observables (mass, growth, exchange) are
captured well one-step; the very high-dimensional flux and abundance vectors are
harder on modest data and are reported honestly. The headline robust result is
the **inference speedup** (~10³×), which holds regardless of fidelity.

Two documented paths to production quality (left as follow-ups):
- **More data on the Mac mini** — many more seeds / multi-generation rollouts
  via Ray, which the pipeline already supports (just more `--seeds`).
- **A latent encode–decode surrogate** for the high-dimensional groups
  (autoencoder + latent dynamics), a natural `pbg-torch` extension beyond the
  current direct residual-MLP.

## Reproduce

```bash
# one-time: install the surrogate library into the v2ecoli venv
uv pip install --python .venv/bin/python torch
uv pip install --python .venv/bin/python "pbg-torch @ git+https://github.com/vivarium-collective/pbg-torch"

cd code
PY=../../../../.venv/bin/python   # the shared v2ecoli venv (has torch + pbg-torch)
$PY sample_baseline.py --seeds 0 1 2 3 4 5 6 7 --n-steps 250 --out ../run
$PY train_surrogate.py --data ../run --hidden 256 256 --epochs 200
$PY evaluate_surrogate.py --data ../run
$PY make_figures.py --data ../run --out ../run/report.html
```

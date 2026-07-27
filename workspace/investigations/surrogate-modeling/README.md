# Surrogate Modeling — a neural-network emulator of the v2ecoli baseline

This investigation builds a **neural-network surrogate** of the v2ecoli
whole-cell model using
[`pbg-torch`](https://github.com/vivarium-collective/pbg-torch) and evaluates
how faithfully and how much faster the surrogate reproduces baseline behavior.

A surrogate is a learned, drop-in process-bigraph `Process` that approximates
the per-step dynamics of another `Process`/`Composite` over a curated panel of
observables. Once trained it rolls out autoregressively at a fraction of the
simulator's cost.

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

5. **Figures** (`report_figures.py`). Self-contained interactive Plotly reports
   written to `reports/figures/<study>/` (auto-discovered as dashboard embeds):
   panel composition + per-seed trajectories (dataset); per-group R², one-step
   parity, per-column R² distribution, and autoregressive rollout (surrogate).

## Studies

- **sm-00-data-collection** — build the broad ~11.5k-observable transition-dataset
  ensemble (8 seeds).
- **sm-01-nn-surrogate** — the test: under 8-fold cross-validation, does the
  neural net beat trivial baselines (persistence, mean-delta, linear)?
- **sm-02-utility-and-limits** — what the cheap **linear** emulator (the one that
  works) is worth: amortized break-even vs. the simulator, and the limits.
- **sm-03-improving-the-surrogate** — follow-up: does multi-step rollout-loss
  training close the gap?
- **sm-04-through-division** — boundary test: follow one daughter lineage across
  a real division and ask whether the emulators hold across the cell_mass
  halving, or fail to represent the discontinuity.

## What we found

A first pass looked like a success — one-step **R²≈1.0** on growth/mass —
but rigor overturned it:

- **One-step R²≈1.0 was a persistence artifact.** Under 8-fold CV, persistence,
  mean-delta, linear, and the neural net all score ~1.000 one-step, because cell
  mass per second is smooth. One-step fidelity measured nothing.
- **In multi-step rollout, a linear model wins** (median nRMSE 0.019, 0/8
  unstable). The neural net is no better and destabilizes on 2/8 folds; it is
  not justified for this target.
- **The broad 11.5k-observable panel is unlearnable from the observable view** —
  0% of observables beat persistence; they depend on the cell's hidden state.
- **The cheap linear emulator is useful** (sm-02): it pays back its 8 training
  sims after ~8 evaluations and sweeps 5,000 trajectories in 0.02 s (vs ~24 h on
  the WCM).
- **Rollout-loss training helps but doesn't overturn this** (sm-03): it removes
  the instability (0/8) and improves the net (0.053 vs 0.069), yet linear still
  wins. Rollout-loss is the technique to carry to nonlinear targets.

- **The emulator does not survive division** (sm-04): following one daughter
  lineage across a real division, every emulator (linear and neural) emulates
  coarse growth within a generation (linear one-step nRMSE 0.0006) but none
  represents the cell_mass halving at the boundary — fed the true pre-division
  state, each predicts continued growth while the truth halves (ratio 0.498),
  a ~6200× one-step error jump. The discontinuity is not learnable from the
  observable view; the emulator is valid only within a generation.

The result is a **scoping** one, not a capability claim. sm-04 closed the
top documented next step (rollout to/through **division**) with another
informative negative that localizes what the observable view is missing.
Remaining next steps: a **latent encode–decode** model that carries cell-cycle
phase (would address both the high-dim unlearnability and the division
boundary — see sm-04's `followup_proposals`), and **more data on the Mac mini**
for the high-dimensional groups.

## Reproduce

```bash
# pbg-torch is a declared v2ecoli dependency (pyproject.toml), so `uv sync`
# installs it; NeuralProcess then appears in the dashboard Registry.

cd code
PY=../../../../.venv/bin/python   # the shared v2ecoli venv (has torch + pbg-torch)
# 1. sample the broad-panel ensemble (sm-00)
$PY sample_baseline.py --seeds 0 1 2 3 4 5 6 7 --n-steps 250 --out ../run
# 2. broad surrogate (sm-01): train, evaluate, figure
$PY train_surrogate.py --data ../run --hidden 256 256 --epochs 200
$PY evaluate_surrogate.py --data ../run
$PY report_figures.py --kind dataset   --data ../run --out ../../../../reports/figures/sm-00-data-collection/dataset_overview.html
$PY report_figures.py --kind surrogate --data ../run --out ../../../../reports/figures/sm-01-nn-surrogate/broad_panel_coverage.html
# 3. compact surrogate (sm-01): slice growth/mass, retrain, figure
$PY slice_compact.py --src ../run --dst ../run_compact --groups mass chromosome
$PY train_surrogate.py --data ../run_compact --hidden 64 64 --epochs 400 --lr 5e-3
$PY evaluate_surrogate.py --data ../run_compact
$PY report_figures.py --kind surrogate --data ../run_compact --out ../../../../reports/figures/sm-01-nn-surrogate/compact_surrogate.html
```

### sm-04 — through division

```bash
cd code
# The shared v2ecoli venv does not ship pbg-torch / plotly / the newer
# bigraph_schema (with .contract) this branch needs; the worktree vendors them
# into .deps (gitignored). Prepend the worktree AND .deps to PYTHONPATH:
PY=~/code/v2ecoli/.venv/bin/python
PP=<worktree>:<worktree>/.deps
# 1. follow 6 lineages to/through one division (~tick 2526; ~1244 s wall)
PYTHONPATH=$PP $PY sample_through_division.py --seeds 0 1 2 3 4 5 \
    --max-steps 3400 --tail 120 --groups mass chromosome --out ../run_through_division
# 2. 6-fold leave-one-lineage-out CV of the emulators across the boundary (dt=8 s)
PYTHONPATH=$PP $PY eval_through_division.py --data ../run_through_division \
    --hidden 64 64 --epochs 80 --rollout-k 10 --stride 8
# 3. figure
PYTHONPATH=$PP $PY make_through_division_figure.py --data ../run_through_division \
    --out ../../../../reports/figures/sm-04-through-division/through_division.html
```


# variant-sweep-phenotype-demo

A **public, perturbation-agnostic template** for sweeping a config-declared
variant index over the whole-config vEcoli node
(`v2ecoli.composites.vecoli.vecoli`) and comparing a phenotype observable
across the sweep axis. It carries no model content — every entry's
`whole_config` is `""` and `observable_bulk_ids` is `[]`. Copy it and fill in
the specifics in your own (possibly private) fork.

## What this template does

`study.yaml` defines one `baseline` condition (`variant: 0`, unperturbed) and
two `variants` (`variant: 1`, `variant: 2`), each pointing at the same
generic composite, `v2ecoli.composites.vecoli.vecoli`. That composite loads
a fork config natively via `EcoliSim` when `whole_config` is set, and selects
one entry of that config's declared `variants` grid by index. With
`whole_config` left empty (as in this template), the composite falls back to
the unperturbed baseline for every entry — so the template is runnable
out-of-the-box, but structurally inert until you point it at real content.

## How to instantiate it

1. **Copy this directory** to a new study name in your own workspace or
   downstream repo, e.g. `workspace/studies/<my-sweep>/`.
2. **Point `whole_config`** (in `baseline` and every entry of `variants`) at
   a fork config file that declares a `variants` block — any config that
   `EcoliSim` can load natively. This is the only place model-specific
   content enters the study.
3. **Set `variant` per entry** to the index into that config's `variants`
   grid you want to run (`0` is conventionally the unperturbed baseline;
   `1`, `2`, … select successive grid points).
4. **List `observable_bulk_ids`** with the bulk molecule ids you want
   emitted as observables for each run — these become the columns you
   compare across the sweep axis.
5. Add or remove `variants` entries to match the size of the grid you want
   to sweep; each entry's `name` should be unique and descriptive.

## Running the sweep

Run each condition under the workbench (a single generation is sufficient —
a whole-config sweep is meant to be read at an early, dose-landed timepoint,
not carried through multiple divisions). Each run emits the observables
declared in `observable_bulk_ids` into its history store.

## Comparing results

Feed the emitted store from each condition into
`v2ecoli.library.phenotype_sweep.collect_sweep` to assemble the sweep into a
single structure, then `v2ecoli.library.phenotype_sweep.sweep_endpoints` to
extract the endpoint values used for cross-variant comparison. The result is
a table/plot of the
chosen observable(s) as a function of the swept variant index — usable for
any perturbation-agnostic phenotype comparison, independent of what
`whole_config` and `observable_bulk_ids` end up pointing at.

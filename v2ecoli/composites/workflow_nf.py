"""``workflow_nf`` — the campaign shape, as a DAG a task scheduler can render.

The general campaign is a two-level scatter with a gather:

    ParCa(variant v)  ──►  cache_v
                             └─► for each seed m:  LineageStep(v, m)  ──┐
    ... for each variant ...                                            ├─►  analysis
                                                                        ┘

**This document is rendered, never ``run()`` in-process.** Its ParCa and analysis
nodes carry no simulation logic at all — only ports and a ``nextflow_script()``.

Two deliberate choices, both recorded because the obvious alternative is wrong:

* **ParCa uses the CLI, not ``--build``.** v2ecoli's registered ``parca`` generator
  is structural: it carries ``raw_data=None`` and does not set ``run_steps_on_init``,
  so ``Composite(doc).run(n)`` would advance ``global_time`` and **run nothing,
  exiting 0**. Emitting ``v2ecoli-parca`` as a script keeps the failure honest.

* **Analysis is not wrapped.** ``v2ecoli-analyze`` is already atomic and
  ``s3://``-capable; wrapping it would duplicate ``run_analyses``' own fan-out.

⛔ **The gather does not construct yet, and this is a document-model limit, not a bug
here.** Wiring one ``sweep_dirs`` port to a *list* of stores — the obvious spelling of an
N×M→1 fan-in — fails inside ``Composite.__init__`` → ``core.realize`` → ``resolve`` with
``TypeError: unhashable type: 'list'``, **before the renderer ever runs**. That is the same
error process-bigraph#201 recorded for ``Mix``, and it has the same cause: a port maps to
one store path, so a flat sibling list cannot express a fan-in at all.

The scatter half is unaffected and renders correctly — measured at 2 variants × 2 seeds:
six process blocks, each lineage consuming *its own* variant's cache channel. So
``include_analysis`` defaults to **False**: the generator produces a document that builds
and renders, and the gather is gated behind a flag that currently raises, rather than
shipping a generator nobody can construct.

The route through is #201's sub-workflow emission, which requires the lineages to be a
**nested Composite** rather than flat siblings — ``take:``/``emit:`` with chained binary
mixes. That is the next change, and it is a restructuring of this document rather than a
new framework feature.

The per-variant ParCa node is what makes a *strain* sweep real rather than nominal:
``new_genes`` / ``bundle_overrides`` reach **ParCa**, so each variant gets its own
cache. Threading them to the lineage instead — which is what today's
``lineage_ray_batch`` façade does by collapsing ``variants`` into
``config_overrides`` — gives one shared cache for the whole sweep, i.e. N runs of
the same genotype wearing different labels.
"""

from __future__ import annotations

import shlex
from typing import Any

from process_bigraph import Step
from viva_superpowers.composite_generator import composite_generator

# Mirrors viva-api's production _parca_command chain, which is the proven
# invocation. The order is load-bearing: v2ecoli-parca emits only the raw
# parca_state.pkl, so build_cache.py must hydrate it into the loadable bundle,
# and build_cache.py reads a GZIPPED fixture.
#
# --new-genes / --bundle-overrides go to v2ecoli-parca ONLY. build_cache.py's CLI
# has neither flag (confirmed against a real crash, viva-api#410); its own
# save_sim_input already writes a correct, strain-specific cache_version.json
# because ParCa received the flags one command earlier.
_PARCA_CHAIN = (
    "v2ecoli-parca --mode {mode} --cpus {cpus} -o {simdata} --cache-dir {cache}{strain_flags}"
    " && gzip -f -k {simdata}/parca_state.pkl"
    " && python scripts/build_cache.py --fixture {simdata}/parca_state.pkl.gz --cache {cache}"
    " && cp {simdata}/parca_state.pkl.gz {cache}/parca_state.pkl.gz"
)


class ParcaTaskStep(Step):
    """One ParCa build, as a task. Carries no simulation logic — ports + a script."""

    config_schema = {
        "variant_index": {"_type": "integer", "_default": 0},
        "variant_name": {"_type": "string", "_default": "baseline"},
        "new_genes": {"_type": "string", "_default": ""},
        "bundle_overrides": {"_type": "string", "_default": ""},
        "mode": {"_type": "string", "_default": "fast"},
        "cpus": {"_type": "integer", "_default": 8},
        "cache_dir": {"_type": "string", "_default": "out/cache"},
        "simdata_dir": {"_type": "string", "_default": "out/parca"},
    }

    nextflow_port_decls = {"cache_dir": 'path "cache"'}

    def inputs(self) -> dict[str, Any]:
        return {}

    def outputs(self) -> dict[str, Any]:
        return {"cache_dir": {"_type": "string", "_is_file": True}}

    def nextflow_script(self) -> str:
        flags = ""
        new_genes = str(self.config.get("new_genes") or "")
        overrides = str(self.config.get("bundle_overrides") or "")
        # "off" is v2ecoli-parca's own default for --new-genes; passing it
        # explicitly and omitting it are the same build, so omit.
        if new_genes and new_genes != "off":
            flags += f" --new-genes {shlex.quote(new_genes)}"
        if overrides:
            flags += f" --bundle-overrides {shlex.quote(overrides)}"
        return _PARCA_CHAIN.format(
            mode=self.config.get("mode", "fast"),
            cpus=int(self.config.get("cpus", 8)),
            simdata=self.config.get("simdata_dir", "out/parca"),
            cache=self.config.get("cache_dir", "out/cache"),
            strain_flags=flags,
        )

    def update(self, state: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError(
            "ParcaTaskStep is a task declaration, not an in-process step: it is rendered "
            "and its nextflow_script() is what runs. Calling update() would silently "
            "produce no cache while reporting success.")


class AnalysisTaskStep(Step):
    """The gather: one analysis over every sweep the campaign produced."""

    config_schema = {
        "experiment_id": {"_type": "string", "_default": "default"},
        "out_dir": {"_type": "string", "_default": "out/analysis"},
        "modules": {"_type": "quote", "_default": []},
    }

    nextflow_port_decls = {"sweep_dirs": "path sweeps", "report": 'path "analysis"'}

    def inputs(self) -> dict[str, Any]:
        # _cardinality many: this is the N x M -> 1 fan-in.
        return {"sweep_dirs": {"_type": "string", "_is_file": True, "_cardinality": "many"}}

    def outputs(self) -> dict[str, Any]:
        return {"report": {"_type": "string", "_is_file": True}}

    def nextflow_script(self) -> str:
        modules = list(self.config.get("modules") or [])
        module_flag = f" --modules {shlex.quote(','.join(modules))}" if modules else ""
        return (
            f"v2ecoli-analyze --experiment-id {shlex.quote(str(self.config.get('experiment_id', 'default')))}"
            f" --out-dir {shlex.quote(str(self.config.get('out_dir', 'out/analysis')))}"
            f"{module_flag}"
        )

    def update(self, state: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError(
            "AnalysisTaskStep is a task declaration, not an in-process step; see "
            "ParcaTaskStep.update.")


def _variant_specs(variants: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Normalize the variant list. A campaign with no variants is ONE baseline
    variant, not zero -- zero would render an empty workflow that exits 0."""
    if not variants:
        return [{"variant_index": 0, "variant_name": "baseline"}]
    out = []
    for i, v in enumerate(variants):
        spec = dict(v)
        spec.setdefault("variant_index", i)
        spec.setdefault("variant_name", f"variant_{i}")
        out.append(spec)
    return out


@composite_generator(
    name="workflow_nf",
    description=(
        "The campaign DAG for task-granularity dispatch: one ParCa per variant, each feeding "
        "that variant's M LineageStep tasks, all N x M gathering into one analysis. Rendered, "
        "never run in-process -- the ParCa and analysis nodes are script declarations."
    ),
    parameters={
        "n_seeds": {"type": "integer", "default": 2,
                    "description": "Seed-lineages per variant."},
        "n_generations": {"type": "integer", "default": 1,
                          "description": "Generations per lineage."},
        "base_seed": {"type": "integer", "default": 0,
                      "description": "First seed; seeds are contiguous per variant."},
        "variants": {"type": "array", "default": None,
                     "description": (
                         "Per-variant strain inputs, each a dict that may carry variant_name, "
                         "new_genes and bundle_overrides. These reach PARCA, giving each variant "
                         "its own cache. None means a single baseline variant.")},
        "experiment_id": {"type": "string", "default": "workflow_nf"},
        "out_dir": {"type": "string", "default": "out/workflow"},
        "max_duration_per_gen": {"type": "number", "default": 3600.0},
        "parca_mode": {"type": "string", "default": "fast"},
        "parca_cpus": {"type": "integer", "default": 8},
        "include_analysis": {"type": "boolean", "default": False,
                             "description": (
                                 "Append the N x M -> 1 gather node. DEFAULT FALSE, and that is "
                                 "a real limitation rather than a preference: see the module "
                                 "docstring. A flat port wired to many stores cannot be "
                                 "constructed at all.")},
    },
)
def build_workflow_nf(
    n_seeds: int = 2,
    n_generations: int = 1,
    base_seed: int = 0,
    variants: list[dict[str, Any]] | None = None,
    experiment_id: str = "workflow_nf",
    out_dir: str = "out/workflow",
    max_duration_per_gen: float = 3600.0,
    parca_mode: str = "fast",
    parca_cpus: int = 8,
    include_analysis: bool = False,
    **_ignored: Any,
) -> dict[str, Any]:
    state: dict[str, Any] = {}
    sweep_paths: list[list[str]] = []

    for spec in _variant_specs(variants):
        vi = int(spec["variant_index"])
        vname = str(spec["variant_name"])
        parca_node = f"parca_v{vi}"
        cache_store = f"cache_v{vi}"

        state[parca_node] = {
            "_type": "step",
            "address": "local:ParcaTaskStep",
            "config": {
                "variant_index": vi,
                "variant_name": vname,
                "new_genes": spec.get("new_genes", ""),
                "bundle_overrides": spec.get("bundle_overrides", ""),
                "mode": parca_mode,
                "cpus": parca_cpus,
                "cache_dir": f"out/cache_v{vi}",
                "simdata_dir": f"out/parca_v{vi}",
            },
            "inputs": {},
            "outputs": {"cache_dir": [cache_store]},
        }

        for m in range(int(n_seeds)):
            seed = int(base_seed) + m
            node = f"lineage_v{vi}_s{seed}"
            sweep_store = f"sweep_v{vi}_s{seed}"
            sweep_paths.append([sweep_store])
            # Everything that distinguishes this lineage lives in THIS config,
            # which is staged as its own file. No sibling shares it.
            config: dict[str, Any] = {
                "seed": seed,
                "lineage_seed": seed,
                "generations": int(n_generations),
                "max_duration_per_gen": float(max_duration_per_gen),
                "experiment_id": experiment_id,
                "out_dir": f"{out_dir}/v{vi}/seed_{seed}",
                "variant_index": vi,
                "variant_name": vname,
            }
            # Omitted, not empty -- see LineageStep for why the distinction matters.
            if spec.get("injected_processes"):
                config["injected_processes"] = spec["injected_processes"]
            if spec.get("config_overrides"):
                config["config_overrides"] = spec["config_overrides"]

            state[node] = {
                "_type": "step",
                "address": "local:LineageStep",
                "config": config,
                "inputs": {"cache_dir": [cache_store]},
                "outputs": {"sweep_dir": [sweep_store]},
            }

    if include_analysis:
        state["analysis"] = {
            "_type": "step",
            "address": "local:AnalysisTaskStep",
            "config": {"experiment_id": experiment_id, "out_dir": f"{out_dir}/analysis"},
            "inputs": {"sweep_dirs": sweep_paths},
            "outputs": {"report": ["report"]},
        }
    return {"state": state}

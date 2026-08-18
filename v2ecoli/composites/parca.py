"""ParCa composite generator — registers the 9-step ParCa pipeline as a
first-class ``@composite_generator`` peer of ``baseline`` / ``colony`` /
``baseline_millard``.

Why this wrapper exists
-----------------------
The ParCa pipeline is *already* a process-bigraph composite — see
``v2ecoli/processes/parca/composite.py`` (``build_parca_composite`` /
``build_parca_document``). What it lacked was **registration**: with no
``@composite_generator`` it was invisible to ``v2ecoli.build_composite``,
the generator registry, and the dashboard's composite catalog, so it
couldn't sit alongside the simulation architectures for accessibility.
This module closes that gap.

Structural, not auto-run
------------------------
``build_composite("parca")`` returns the pipeline *document* (the 9 wired
steps + nested stores) but does **not** set ``run_steps_on_init``, so
constructing it does not fire the multi-hour fit. That keeps catalog/registry
listing cheap and side-effect-free. Actually *running* ParCa still goes
through the executable entry points, which load the ``KnowledgeBaseEcoli``
``raw_data`` and fire the pipeline:

    v2ecoli-parca                       # console script (preferred)
    python scripts/parca_run.py
    build_parca_composite(raw_data, ...)  # programmatic

The registered document carries ``raw_data=None`` in Step 1's config (the
real KB is injected by those runners), matching ``build_parca_document``.

Genotype identity — declarative, not effective
----------------------------------------------
``bundle_manifest`` / ``bundle_overrides`` name the ecoli-sources bundle a
build's genotype is defined by. A perturbed genome (a knockout, a knockdown)
is a *different manifest*, so the manifest is what distinguishes one
genotype's ParCa build from another's.

They are **declarative**: this generator returns a structural document and
deliberately does not construct a ``KnowledgeBaseEcoli`` (that costs seconds
and would make registry listing expensive — see "Structural, not auto-run"
above). The KB is still built and injected by the runners, which already
accept the same identity via ``v2ecoli-parca --bundle-manifest-path``.

What declaring them buys today:

- a **study** can name the genotype it builds in ``conditions.baseline.params``
  and have it schema-validated, instead of recording it in prose;
- the emitted document carries the identity on the step that consumes
  ``raw_data``, so a saved document says which genome it was for;
- ``InitializeStep`` **cross-checks** the declared manifest against the bundle
  the injected ``raw_data`` was actually built from and warns on a mismatch —
  the failure worth catching, since a silent mismatch fits successfully and
  attributes the result to the wrong genotype.

Making them *effective* — i.e. having a study run construct the KB from the
declared manifest without going through the CLI — needs the study runner to
drive ParCa, and is deliberately out of scope here.

**Caveat, and it is why ``new_genes`` sits beside them:** the declarative story
above holds only while a real ``raw_data`` is injected. When it is not (the
workbench path), ``InitializeStep`` builds the KB itself from these fields — so
on that path they ARE effective. ``new_genes`` is effective on that path too and
is not declarative in any case: it changes the genome the fit is built from,
which is not a claim about identity but a change of input.
"""

from typing import Any

from viva_superpowers.composite_generator import composite_generator

from v2ecoli.processes.parca.composite import build_parca_document, STEP_ORDER
from v2ecoli.processes.parca.schema import register_parca_schema
from v2ecoli.processes.parca.steps import ALL_STEP_CLASSES


def register_parca_core(core: Any) -> Any:
    """Register ParCa step classes + types on ``core``.

    Mirrors ``allocate_core(top=ALL_STEP_CLASSES)`` + ``register_parca_schema``
    from ``build_parca_composite``, but applied to an already-allocated core
    (the one ``build_composite`` hands the generator). Declared as a
    ``core_extensions`` entry too, so the dashboard's subprocess runner applies
    the same registrations to the core it runs against.
    """
    # Step classes resolve via the ``local:<ClassName>`` link registry — the
    # same path ``allocate_core(top=ALL_STEP_CLASSES)`` uses. Register each
    # under its class name so the document's ``address: local:InitializeStep``
    # (etc.) resolve on this core.
    core.register_links({name: cls for name, cls in ALL_STEP_CLASSES.items()})
    register_parca_schema(core)
    return core


@composite_generator(
    name="parca",
    description=(
        "ParCa parameter-calculation pipeline — the 9-step fit "
        "(initialize → input_adjustments → … → final_adjustments) that "
        "produces sim_data. Structural document; run via the v2ecoli-parca "
        "CLI / build_parca_composite (which load the KnowledgeBase raw_data)."
    ),
    parameters={
        "debug": {
            "type": "boolean",
            "default": False,
            "description": "Run Steps in debug mode (extra validation/logging).",
        },
        "cpus": {
            "type": "integer",
            "default": 1,
            "description": "Parallelism for the condition-fitting Steps (4, 5).",
        },
        "cache_dir": {
            "type": "string",
            "default": "",
            "description": "Optional cache directory passed to BasalSpecsStep.",
        },
        "bundle_manifest": {
            "type": "string",
            "default": "",
            "description": (
                "Path to the ecoli-sources bundle manifest this build's "
                "genotype is defined by. A perturbed genome (knockout, "
                "knockdown) is a different manifest, so this is what "
                "distinguishes one genotype's ParCa build from another's. "
                "Empty means the installed default bundle (wild type). "
                "DECLARATIVE — see the note in the module docstring."
            ),
        },
        "bundle_overrides": {
            "type": "string",
            "default": "",
            "description": (
                "Optional overrides manifest layered on top of "
                "bundle_manifest, matching SourceBundle(overrides=...). "
                "Applied AFTER v2ecoli's own overrides, never instead of "
                "them, so naming one cannot silently revert v2ecoli's "
                "diverged flat files. This is how a private payload adds "
                "keys (e.g. a strain's new-gene inputs) to the baseline."
            ),
        },
        "new_genes": {
            "type": "string",
            "default": "",
            "description": (
                "Name of a new_gene_data subdirectory to insert into the "
                "genome (e.g. a heterologous pathway supplied by "
                "bundle_overrides); empty means none. Unlike the two bundle "
                "fields this is NOT declarative — it changes the genome the "
                "fit is built from."
            ),
        },
    },
    default_n_steps=len(STEP_ORDER),
    core_extensions=[register_parca_core],
)
def parca(core: Any = None, *, debug: bool = False, cpus: int = 1,
          cache_dir: str = "", bundle_manifest: str = "",
          bundle_overrides: str = "", new_genes: str = "") -> dict:
    """Build the ParCa pipeline document (structure only — does not run).

    Args:
        core: bigraph-schema core. ``build_composite`` passes the one it will
            wrap the document with; we register the ParCa steps/types on it
            here so the document's ``local:`` step addresses resolve. (When
            called with ``core=None`` — e.g. document serialization — the
            caller is responsible for registration via ``core_extensions``.)
        debug, cpus, cache_dir: forwarded to the relevant Step configs.
        bundle_manifest, bundle_overrides: the genotype this build is for,
            recorded on InitializeStep's config. Declarative — see the
            module docstring's "Genotype identity" note.
        new_genes: name of a new_gene_data subdirectory to insert (its flat
            inputs typically arriving via bundle_overrides). Not declarative —
            it changes the genome the fit is built from.

    Returns:
        A process-bigraph document dict (the 9-step pipeline state). No
        ``run_steps_on_init`` key, so constructing the Composite does not fire
        the pipeline.
    """
    if core is not None:
        register_parca_core(core)
    # include_store_skeleton=True so the dashboard explorer renders the 9 steps
    # as a connected pipeline (empty store nodes give loom layout anchors)
    # rather than collapsing them onto the origin. The committed models/parca.pbg
    # stays steps-only (build_parca_document default).
    return {"state": build_parca_document(
        debug=debug, cpus=cpus, cache_dir=cache_dir,
        bundle_manifest=bundle_manifest, bundle_overrides=bundle_overrides,
        new_genes=new_genes,
        include_store_skeleton=True)}

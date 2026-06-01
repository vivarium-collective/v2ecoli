"""ParCa composite generator — registers the 9-step ParCa pipeline as a
first-class ``@composite_generator`` peer of ``baseline`` / ``colony`` /
``millard_pdmp_baseline``.

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
"""

from typing import Any

from pbg_superpowers.composite_generator import composite_generator

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
    },
    default_n_steps=len(STEP_ORDER),
    core_extensions=[register_parca_core],
)
def parca(core: Any = None, *, debug: bool = False, cpus: int = 1,
          cache_dir: str = "") -> dict:
    """Build the ParCa pipeline document (structure only — does not run).

    Args:
        core: bigraph-schema core. ``build_composite`` passes the one it will
            wrap the document with; we register the ParCa steps/types on it
            here so the document's ``local:`` step addresses resolve. (When
            called with ``core=None`` — e.g. document serialization — the
            caller is responsible for registration via ``core_extensions``.)
        debug, cpus, cache_dir: forwarded to the relevant Step configs.

    Returns:
        A process-bigraph document dict (the 9-step pipeline state). No
        ``run_steps_on_init`` key, so constructing the Composite does not fire
        the pipeline.
    """
    if core is not None:
        register_parca_core(core)
    return {"state": build_parca_document(debug=debug, cpus=cpus, cache_dir=cache_dir)}

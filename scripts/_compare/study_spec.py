"""Load investigation + study YAML as the single comparison spec.

The study YAML is the source of truth (no manifest JSON). A study's `name` is the
store/verdict/card key; its `condition` is the biological vEcoli condition the
ensemble simulates (these differ for disambiguated studies, e.g. `basal_4x4` runs
the `basal` condition with 4 seeds). The investigation YAML carries the shared
execution context (ParCa caches, reference engine, default cards). A study's
`config` is a condition name OR a reference-config path -- a "swap" (e.g.
MetabolismRedux) is just a config path that drives both engines identically.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from pathlib import Path

import yaml

from scripts._compare.reference import ReferenceEngine

REPO = Path(__file__).resolve().parent.parent.parent
INVEST_ROOT = REPO / "workspace" / "investigations"
STUDIES_ROOT = REPO / "workspace" / "studies"
DEFAULT_INVEST = "whole-cell-model-comparison"
# Cards that GATE (pass/fail). The multi-seed `statistical` card (Welch t-test over
# >=4 seeds) is the gold standard for trajectory reproduction; `parca` gates the
# t=0 initial-state match. The single-seed `standard` card is DELIBERATELY NOT a
# gate: one seed's trajectory cannot separate a real port divergence from stochastic
# seed noise (v2ecoli's own 4-seed spread on succinate is ~8-9%, larger than the
# single-seed "mismatch" it was flagged for). `standard` is kept as an illustrative
# trajectory card only. See project memory `project_v2e_compare_singleseed_stochastic`.
GRADED = {"statistical", "parca"}   # cards that produce a gating test
_DEFAULT_CARDS = ["summary", "config", "parca", "standard", "statistical"]
_DEFAULT_V2_CACHE = "out/cache_full"
_DEFAULT_VE_CACHE = "out/compare_harness/vecoli_parca"


@dataclass
class StudySpec:
    name: str                 # store/verdict/card key (study identity)
    condition: str            # biological vEcoli condition (--condition)
    seeds: int
    gens: int
    cards: list
    invest_name: str
    v2_cache: str
    ve_cache: str
    study_path: str
    config: str = ""           # a condition name, or a reference-config path
                                # driving a process swap on BOTH engines (e.g.
                                # metabolism_redux); a bare name = no swap
    reference: "ReferenceEngine | None" = None   # how to run the reference engine
    max_steps_per_gen: int = 15000  # per-generation tick budget; lower it for a
                                    # short-horizon run of an expensive swap (e.g.
                                    # MetabolismRedux solves an LP per tick)
    inject_processes: list = dc_field(default_factory=list)
    # Extra FORK processes to inject alongside the ones `config` declares. A
    # swapped fork process can depend on a companion fork process — usually a
    # listener it reads a store from — that arrives on the fork via the config's
    # standard `processes` list, which injection does not carry. Naming it here
    # keeps the declaration with the study that needs it, rather than requiring
    # an edit to a shared fork config. Explicit by design: inferring which ports
    # lack a writer from a vivarium-1.0 ports_schema means guessing at port
    # direction.
    observables: list = dc_field(default_factory=list)  # arbitrary "group.leaf" listener
                                    # paths to emit on BOTH arms as measurements
    generation_lower_bound: int = 0  # grade only generations >= this. 0 = every
                                    # generation, including pre-settling ones.
                                    # ⚠ ANALYSIS-TIME, unlike exchange_flux_basis
                                    # below: it does not change what the run
                                    # emits, only how the card aggregates what
                                    # was emitted, so it is threaded to the card
                                    # through study state rather than ridden to
                                    # the engines as a flag.
    exchange_flux_basis: str = "counts"   # "counts" | "gdcw" — WHICH QUANTITY
                                    # the exchange_flux leaves carry, on BOTH
                                    # arms. counts is a lineage-cumulative
                                    # molecule total (its time-average is not a
                                    # rate); gdcw is mmol/gDCW/h. Different
                                    # measurements, not different units.
    exchange_fluxes: dict = dc_field(default_factory=dict)  # {leaf: exchange_key}
                                    # metabolic exchange fluxes to emit onto
                                    # listeners.exchange_flux.<leaf> on BOTH arms
                                    # (e.g. the violacein card's rate/yield inputs)
    observable_bulk_ids: list = dc_field(default_factory=list)  # bulk molecule ids
                                    # to grade as config-specific KPIs, emitted on
                                    # BOTH arms under listeners.observable_bulk.<id>
                                    # (violacein titer, antibiotic drug-target complex)

    @property
    def graded_cards(self) -> list:
        return [c for c in self.cards if c in GRADED]


def studies_root_for(inv_dir) -> Path:
    """Where a workspace keeps its studies.

    Studies live as a SIBLING of ``investigations/`` under the same workspace
    root (``inv_dir`` is always ``<workspace>/investigations/<name>``). Derived
    from ``inv_dir`` rather than hardcoding ``REPO`` so a caller pointing at an
    alternate or test workspace resolves studies within THAT workspace.

    ⚠ Both the legacy members path and ``specs_from_configs`` resolve through
    here. The hardcoded form used to appear only in a ``study_path=`` string --
    cosmetic, since nothing read it. It is load-bearing now that companions are
    read from that file, and an investigation rooted outside ``REPO`` would have
    read none of them with no error.
    """
    # NOT the module-level STUDIES_ROOT: that is bound at import, so a test (or
    # any caller) repointing REPO would not reach it. Looked up at call time.
    return (Path(inv_dir).parent.parent / "studies") if inv_dir \
        else (REPO / "workspace" / "studies")


def exchange_flux_basis_from_study_yaml(study_path, fallback: str = "counts") -> str:
    """Read `exchange_flux_basis` from a study.yaml, for the SAME reason
    `companions_from_study_yaml` exists.

    The investigation route builds specs from `comparison.configs[]` entries,
    which do not carry study.yaml keys — so without this a study declaring
    `gdcw` in the file the investigation NAMES would silently run on `counts`.
    ⚠ And that failure is worse than the companion one it mirrors: a dropped
    companion or flux map yields MISSING leaves, which is visible; a dropped
    basis yields leaves that are present and carrying the other quantity, which
    is not. `fallback` is whatever the configs[] entry or defaults resolved to,
    so an investigation-level declaration still wins where the study is silent.
    """
    path = Path(study_path)
    if not path.exists():
        return fallback
    try:
        import yaml
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001 — a study we cannot read keeps the fallback
        return fallback
    if not isinstance(doc, dict):
        return fallback
    comp = doc.get("comparison") if isinstance(doc.get("comparison"), dict) else {}
    # ⚠ `comparison:` ONLY — deliberately not top-level. An earlier version read
    # top-level first, which _spec_from_study never looks at, so the two spec
    # routes disagreed on the same file and a run could emit one quantity while
    # the card graded another. Every other per-study measurement key
    # (exchange_fluxes, observables, observable_bulk_ids) resolves from
    # `comparison:`, and this now matches them.
    v = comp.get("exchange_flux_basis")
    return str(v) if v else fallback


def _validate_generation_window(name, lower_bound, gens, *, source: str) -> int:
    """Refuse a window that would grade NOTHING. Returns the validated bound.

    ⛔ WHY REFUSE. A bound excluding every generation leaves the card nothing to
    compute, so the axis goes `ungraded` — which the shared severity model scores
    0, i.e. no worse than a pass. A study that windows itself out therefore
    SILENTLY RELAXES the gate it was written to enforce.

    ⚠ WHY HERE AND NOT IN `StudySpec.__post_init__`, where this started. The rule
    is not an invariant of a StudySpec: it is an invariant of RESOLVING TWO
    INDEPENDENTLY-AUTHORED DECLARATIONS — an investigation-level default and a
    per-study generation count. A constructor can see neither which file supplied
    the bound nor whether an author supplied it at all, and enforcing it there
    produced two bad failures:
      * an investigation declaring `defaults.generation_lower_bound: 5` with ONE
        member legitimately running `generations: 1` failed the ENTIRE
        investigation load — every unrelated member included — because
        `load_investigation` builds every spec;
      * a `configs[]` entry with `gens: 0` raised an error naming
        `generation_lower_bound`, a key its author had never written.
    Resolving here also lets the message say WHERE the numbers came from.

    ⚠ SCOPE, because the comment this replaces overclaimed: this compares the
    bound against the REQUESTED generation count. A run that dies early — OOM,
    wall-clock, step budget — still windows out every achieved cell at grading
    time, and nothing here can see that. **This closes the arithmetic typo, not
    the failure class.**
    """
    lower_bound = int(lower_bound)
    gens = int(gens)
    if lower_bound < 0:
        raise ValueError(
            f"{name}: generation_lower_bound must be >= 0 (got {lower_bound}, "
            f"from {source})")
    if gens >= 1 and lower_bound >= gens:
        raise ValueError(
            f"{name}: generation_lower_bound={lower_bound} (from {source}) "
            f"excludes every generation of a {gens}-generation run — nothing "
            f"would be graded. Lower the bound, or raise `generations`.")
    return lower_bound


def _first_declared(*values, default=0):
    """First value that was actually DECLARED, i.e. not None.

    ⛔ NOT `a or b or default`. `0` is a legitimate declaration of
    `generation_lower_bound` — "grade every generation, deliberately" — and it is
    falsy, so an `or` chain silently discards an explicit opt-out in favour of
    the investigation-level default. This module already guards that case in
    `generation_lower_bound_from_study_yaml`; the precedence chain feeding its
    `fallback` has to guard it too, or the reader's care is undone one call up.
    """
    for v in values:
        if v is not None:
            return v
    return default


def generation_lower_bound_from_study_yaml(study_path, fallback: int = 0) -> int:
    """Read `comparison.generation_lower_bound` from a study.yaml.

    Mirrors `exchange_flux_basis_from_study_yaml` deliberately — same bridge,
    same precedence, same `comparison:`-only rule — because the investigation
    route builds specs from `comparison.configs[]` entries that carry no
    study.yaml keys, and a study declaring a window in the file the
    investigation NAMES would otherwise silently grade every generation.

    ⛔ `comparison:` ONLY, never top-level. Two parse routes reading different
    spellings of one key is not hypothetical in this file: `gens` (configs[]
    route) and `generations` (study.yaml route) already differ, and BOTH
    silently default. A third such split would be the same defect again.

    ⚠ A window is not a cosmetic filter: it is the difference between a mean
    over settled cells and one dragged toward the pre-settling generations. The
    reference config's own analyses declare `generation_lower_bound: 5`.
    """
    path = Path(study_path)
    if not path.exists():
        return fallback
    try:
        import yaml
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001 — a study we cannot read keeps the fallback
        return fallback
    if not isinstance(doc, dict):
        return fallback
    comp = doc.get("comparison") if isinstance(doc.get("comparison"), dict) else {}
    v = comp.get("generation_lower_bound")
    if v is None:
        return fallback
    # ⛔ A DECLARED value that cannot be read is an ERROR, not a fallback — and
    # this is deliberately asymmetric with the missing-file branch above. A study
    # we cannot open declared nothing; a study that declared
    # `generation_lower_bound: post-burn-in` DID declare, and silently grading
    # every generation instead is the same gate-relaxing failure the validator
    # refuses at the other end of the range. An earlier version returned the
    # fallback here and a test PINNED that silence as intended.
    # ⚠ bool is an int subclass: `true` would read as 1 and `false` as 0, which
    # is never what an author meant by a generation index.
    if isinstance(v, bool) or not isinstance(v, (int, str)):
        raise ValueError(
            f"{study_path}: comparison.generation_lower_bound must be an "
            f"integer generation index (got {v!r})")
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        raise ValueError(
            f"{study_path}: comparison.generation_lower_bound must be an "
            f"integer generation index (got {v!r})") from None


def companions_from_study_yaml(study_path) -> list:
    """Read `inject_processes` from a study.yaml — the ONE surface that declares it.

    Top-level first, then `comparison:` — mirroring how `config` resolves.
    ⚠ Precedence is by TRUTHINESS, not presence: an empty list at top level is
    falsy and falls through, so top-level cannot clear a companion declared
    below.

    Both spec routes read through here. The investigation route builds specs
    from `comparison.configs[]` entries, which do NOT carry this key, and it
    already knows each study's yaml path — so without this it would silently
    ignore a declaration sitting in the file it names, and the study.yaml would
    look correct while doing nothing on one of two first-class routes.
    """
    path = Path(study_path)
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    comp = data.get("comparison") or {}
    return list(data.get("inject_processes") or comp.get("inject_processes") or [])


def is_reference_config(config) -> str | bool:
    """True when `config` is a reference-config PATH rather than a bare condition
    name. The single definition of that distinction: it decides whether a run
    drives a process swap at all, and three copies of the predicate had already
    started to accumulate."""
    return str(config).endswith(".json")


def check_companions_are_reachable(inject_processes, config, where) -> None:
    """Raise when companions are declared on a route/config that cannot inject them.

    Companions are injected only on the ``--from-vecoli-config`` path
    (``run_comparison_ensemble`` guards the injection block on it), so a bare
    condition name silently discards the declaration. Applied on BOTH spec
    routes -- a guard that only covers one of them is how a declaration ends up
    looking correct while doing nothing."""
    if inject_processes and not is_reference_config(config):
        raise ValueError(
            f"{where}: inject_processes={list(inject_processes)} but config="
            f"{config!r} is a bare condition name, not a reference-config path. "
            "Companion processes are injected only on the --from-vecoli-config "
            "path, so this declaration would be silently discarded.")


def _invest_dir(ref: str) -> Path:
    """Resolve an investigation NAME or a path (to investigation.yaml or its dir)."""
    p = Path(ref)
    if p.name == "investigation.yaml":
        return p.parent
    if p.is_dir() and (p / "investigation.yaml").exists():
        return p
    return INVEST_ROOT / ref


def _context(inv_dir: Path) -> dict:
    """Shared execution context from an investigation.yaml's `comparison` block."""
    path = inv_dir / "investigation.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    data = data or {}
    comp = data.get("comparison") or {}
    # Canonical model (post-#390): investigations reference top-level
    # workspace/studies/<slug>/ via `members:`. Fall back to the legacy
    # `studies:` key for any investigation not yet migrated.
    members = data.get("members") or data.get("studies") or []
    defaults = comp.get("defaults") or {}
    return {
        "invest_name": data.get("name", inv_dir.name),
        "members": members,
        "studies": members,
        "reference": ReferenceEngine.from_spec(comp.get("reference") or {}),
        "configs": comp.get("configs") or [],
        "v2_cache": comp.get("v2_cache", _DEFAULT_V2_CACHE),
        "ve_cache": comp.get("ve_cache", _DEFAULT_VE_CACHE),
        "defaults": defaults,
        "default_cards": defaults.get("cards") or list(_DEFAULT_CARDS),
        "inv_dir": inv_dir,
    }


def specs_from_configs(ctx: dict) -> list:
    """One StudySpec per `comparison.configs[]` entry -- a config is the unit."""
    defaults = ctx.get("defaults") or {}
    out = []
    for entry in ctx["configs"]:
        name = entry["name"]
        _gens = int(entry.get("gens", defaults.get("generations",
                                                   defaults.get("gens", 1))))
        # ⚠ `_spec_from_study` has always validated this; this route did not, so
        # a `gens: 0` entry fell through to the window validator and raised an
        # error naming `generation_lower_bound` — a key its author never wrote.
        # Validate the number the author actually supplied, where they supplied it.
        if _gens < 1:
            raise ValueError(
                f"{name}: comparison.configs[] entry has gens={_gens}; "
                f"must be >= 1")
        _glb = _validate_generation_window(
            name,
            generation_lower_bound_from_study_yaml(
                studies_root_for(ctx.get("inv_dir")) / name / "study.yaml",
                fallback=int(_first_declared(
                    entry.get("generation_lower_bound"),
                    defaults.get("generation_lower_bound")))),
            _gens,
            source="study.yaml / configs[] entry / investigation defaults")
        cfg = entry.get("config", name)
        study_yaml = studies_root_for(ctx.get("inv_dir")) / name / "study.yaml"
        companions = companions_from_study_yaml(study_yaml)
        check_companions_are_reachable(companions, cfg, str(study_yaml))
        out.append(StudySpec(
            name=name,
            condition=entry.get("condition", name),
            config=cfg,
            seeds=int(entry.get("seeds", defaults.get("seeds", 4))),
            gens=int(entry.get("gens", defaults.get("generations", defaults.get("gens", 1)))),
            cards=list(entry.get("cards") or defaults.get("cards") or list(_DEFAULT_CARDS)),
            invest_name=ctx["invest_name"],
            v2_cache=ctx["v2_cache"],
            ve_cache=ctx["ve_cache"],
            reference=ctx["reference"],
            study_path=str(study_yaml),
            max_steps_per_gen=int(entry.get("max_steps_per_gen") or 15000),
            inject_processes=companions,
            # Measurement declarations: per-config entry wins, else the
            # investigation `defaults` block (so a whole investigation can share
            # one measurement set). Mirrors the study.yaml path in _spec_from_study.
            observables=list(entry.get("observables")
                             or defaults.get("observables") or []),
            exchange_fluxes=dict(entry.get("exchange_fluxes")
                                 or defaults.get("exchange_fluxes") or {}),
            # The study.yaml gets the LAST word, via the same bridge companions
            # use: this route builds from configs[] entries, which carry no
            # study.yaml keys, so a study declaring the basis in the file this
            # investigation names would otherwise silently run on the fallback.
            exchange_flux_basis=exchange_flux_basis_from_study_yaml(
                study_yaml,
                fallback=str(entry.get("exchange_flux_basis")
                             or defaults.get("exchange_flux_basis")
                             or "counts")),
            generation_lower_bound=_glb,
            observable_bulk_ids=list(entry.get("observable_bulk_ids")
                                     or defaults.get("observable_bulk_ids") or []),
        ))
    return out


def _spec_from_study(study_path: Path, ctx: dict) -> StudySpec:
    data = yaml.safe_load(study_path.read_text(encoding="utf-8")) or {}
    comp = data.get("comparison") or {}
    name = data.get("name") or study_path.parent.name
    condition = data.get("condition")
    if not condition:
        raise ValueError(f"{study_path}: study has no `condition` (the biological "
                         f"vEcoli condition to simulate)")
    raw_seeds = comp.get("seeds", 1)
    raw_gens = comp.get("generations", 1)
    seeds = int(raw_seeds) if raw_seeds is not None else 1
    gens = int(raw_gens) if raw_gens is not None else 1
    if seeds < 1 or gens < 1:
        raise ValueError(f"{study_path}: comparison.seeds/generations must be >= 1 "
                         f"(got seeds={seeds}, generations={gens})")
    # `config` is the new name; `from_vecoli_config` is read for backward
    # compat with study.yaml content not yet migrated (see task-2 report).
    inject_processes = companions_from_study_yaml(study_path)
    config = (data.get("config") or comp.get("config")
              or data.get("from_vecoli_config") or comp.get("from_vecoli_config")
              or name)
    # A companion is only ever injected on the `--from-vecoli-config` path
    # (run_comparison_ensemble.py guards the injection block on it). A study whose
    # `config` is a bare condition name drives no injection at all, so a companion
    # declared there would be passed to the runner and silently discarded — the
    # fail-without-erroring shape. Say so instead.
    check_companions_are_reachable(inject_processes, config, study_path)

    return StudySpec(
        name=name,
        condition=condition,
        seeds=seeds,
        gens=gens,
        cards=list(comp.get("cards") or ctx["default_cards"]),
        invest_name=ctx["invest_name"],
        v2_cache=ctx["v2_cache"],
        ve_cache=ctx["ve_cache"],
        config=config,
        inject_processes=inject_processes,
        reference=ctx["reference"],
        study_path=str(study_path),
        max_steps_per_gen=int(comp.get("max_steps_per_gen") or 15000),
        observables=list(comp.get("observables")
                         or (ctx.get("defaults") or {}).get("observables") or []),
        exchange_fluxes=dict(comp.get("exchange_fluxes")
                             or (ctx.get("defaults") or {}).get("exchange_fluxes") or {}),
        # Same helper as the investigation route — one reader, one precedence.
        exchange_flux_basis=exchange_flux_basis_from_study_yaml(
            study_path,
            fallback=str((ctx.get("defaults") or {}).get("exchange_flux_basis")
                         or "counts")),
        generation_lower_bound=_validate_generation_window(
            name,
            generation_lower_bound_from_study_yaml(
                study_path,
                fallback=int(_first_declared(
                    (ctx.get("defaults") or {}).get("generation_lower_bound")))),
            gens,
            source=f"{study_path} / investigation defaults"),
        observable_bulk_ids=list(comp.get("observable_bulk_ids")
                                 or (ctx.get("defaults") or {}).get("observable_bulk_ids") or []),
    )


def load_investigation(ref: str) -> tuple[dict, list]:
    """Return (context, [StudySpec, ...]) for an investigation name or path.

    Supports BOTH investigation schemas. If `comparison.configs[]` is present
    (the config-is-the-unit model, post-Task-6), specs are built directly via
    specs_from_configs() -- the same source run/init/scaffold already use, so
    render/run/scaffold/init all agree on one spec list. Otherwise (legacy
    investigations not yet migrated off `members:`), studies are loaded in
    the order listed in `members:` (legacy `studies:` for any un-migrated
    investigation); each is resolved from the canonical TOP-LEVEL
    workspace/studies/<slug>/study.yaml (Study Pipeline registry model,
    post-#390) -- NOT the legacy nested inv_dir/studies/<slug>/. A listed
    study whose study.yaml is missing is skipped.
    """
    inv_dir = _invest_dir(ref)
    if not (inv_dir / "investigation.yaml").exists():
        raise FileNotFoundError(f"investigation not found: {inv_dir}/investigation.yaml")
    ctx = _context(inv_dir)
    if ctx.get("configs"):
        return ctx, specs_from_configs(ctx)
    # Top-level studies live as a SIBLING of investigations/ under the same
    # workspace root (inv_dir is always <workspace>/investigations/<name>);
    # derive it from inv_dir rather than hardcoding REPO so a caller pointing
    # at an alternate/test workspace resolves studies within that workspace.
    studies_root = studies_root_for(inv_dir)
    specs = []
    for sname in ctx["members"]:
        sp = studies_root / sname / "study.yaml"
        if sp.exists():
            specs.append(_spec_from_study(sp, ctx))
    return ctx, specs


def load_study(ref: str) -> StudySpec:
    """Return a StudySpec for a study NAME (under the canonical investigation) or
    a path (to a study.yaml or its dir). The investigation `comparison` context
    (caches, reference engine) is resolved via the study's `investigation:`
    back-reference."""
    p = Path(ref)
    if p.name == "study.yaml":
        sp = p
    elif p.is_dir() and (p / "study.yaml").exists():
        sp = p / "study.yaml"
    else:
        sp = STUDIES_ROOT / ref / "study.yaml"
    if not sp.exists():
        raise FileNotFoundError(f"study not found: {sp}")
    data = yaml.safe_load(sp.read_text(encoding="utf-8")) or {}
    inv_dir = INVEST_ROOT / data.get("investigation", DEFAULT_INVEST)
    return _spec_from_study(sp, _context(inv_dir))

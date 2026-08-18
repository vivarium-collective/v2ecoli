"""Comparison operands — one shape, whatever side of a comparison they are on.

A report card grades **two operands**. What determines how an operand is stored
and resolved is the PATH it arrived by — promoted into a curated payload, or
materialised from an artifact inside the investigation — **not** whether it is a
measurement or a simulation. Kind stops deciding *where an operand lives* and
keeps deciding *how it is graded*.

Why this module exists, concretely. ``vs_experiment`` resolved its two sides
through two different mechanisms that produced two different in-memory types:

    measured   loader -> pandas frame, VectorObservationSchema shape
    simulated  json   -> {entity_id: value}, a bare dict

so the join could only ever be written one-way-round. That was invisible while
the model was always the sim side and the reference always the measurement. It
stops being invisible the moment a comparison is exp<->exp (MS#8 #2) or
sim<->sim (MS#8 #8), which is two of the six comparisons the CD1 evaluation asks
for.

All three resolvers return the SAME shape, so a caller cannot tell them apart
without reading ``path`` — which is the property worth having (Layer 1)::

    promoted_operand   promoted        a curated payload slot
    fixture_operand    fixture         a baked model fixture
    run_operand        in-investigation a live run, via the run-keyed cache

``run_operand`` is the path this module's opening sentence always described and
no code implemented. Adding it is what makes "does a live run collapse into the
operand contract?" a question with an answer rather than an assertion: the
resolver is ~40 lines, it adds no schema, and Layer 2 below did not change at
all to accept it.

**Layer 2 — the join below is symmetric.** Nothing in ``_join_vectors``
distinguishes "the model side" from "the reference side": both operands are
filtered by THEIR OWN detection rule, both contribute coverage counts, and both
are checked for a usable value. ``kind`` may be read here to decide *how to
grade* (e.g. whether ``detection`` carries information); it must never be read
to decide *how to load* — that stays Layer 1's contract, kind-blind by
construction.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

#: Columns a resolved operand's frame is guaranteed to carry. This is the
#: measured tier's own vocabulary (``VectorObservationSchema``) rather than a new
#: one, because the curated simulated operands already validate against it — the
#: only field that says which side of a comparison a row is on is ``kind``, and
#: the public schema already allowed ``model_predicted``. A promoted simulation
#: needed no schema work at all, which is the strongest evidence the path/kind
#: split is right.
OPERAND_COLUMNS = ("entity_id", "symbol", "mean_geometric", "detection", "kind")


@dataclass(frozen=True)
class Operand:
    """One side of a comparison: rows, plus how they came to be.

    ``frame`` carries at least ``OPERAND_COLUMNS``. ``path`` is the only field
    that varies with resolution mechanism; ``kind`` is a grading input and must
    never be branched on to decide how to LOAD.
    """
    frame: pd.DataFrame
    path: str                       # promoted | fixture | in-investigation
    kind: str                       # measured | model_predicted | theoretical_max
    label: str                      # human-facing, for provenance panels
    meta: dict = field(default_factory=dict)

    @property
    def values(self) -> dict:
        """``entity_id -> mean_geometric``: the raw id-space view a join needs.

        **Nulls only** are dropped — a null centre is not a number and cannot
        enter a map keyed for arithmetic. Zeros and negatives are deliberately
        KEPT: whether a non-positive value excludes an entity depends on which
        side of the join it is on (an entity absent from the other operand's
        id-space and an entity present-but-zero are different facts, counted
        separately), so that decision belongs to the join and not here. Filtering
        it in this property would silently change what ``n_shared`` means.
        """
        return {str(r.entity_id): float(r.mean_geometric)
                for r in self.frame.itertuples()
                if r.mean_geometric is not None and pd.notna(r.mean_geometric)}

    @property
    def declared_zeros(self) -> set:
        """Entities this operand recorded as a MEASURED ZERO — *"we looked and
        counted none"* — as distinct from ones it says nothing about.

        **Why this is not reachable through ``values``, and must not be made
        so.** A true zero cannot be represented as a centre on the log scale:
        the geometric mean of all-zero replicates is undefined, so the promoted
        tier records it as a NULL geometric centre, with the fact itself carried
        in the counts — ``n_pos`` (0) and ``n`` (> 0) — and in
        ``mean_arithmetic`` (0.0). ``values`` drops nulls, correctly, and in
        doing so drops the recorded zero along with genuinely absent rows. So
        the payload honours the true-zero-vs-missing distinction
        (`comparison-operands-plan` D5) and the consumer cannot see it — and the
        loss is invisible, because it presents as a null rather than a deletion.

        ⚠ **That is a statement about ``values``, not a licence to key on the
        null.** Which statistic sits in ``mean_geometric`` is a property of the
        *presentation*, not of the record: a card may substitute a different
        centre before grading, and `vs_experiment` does — it grades the
        ARITHMETIC centre (matching the prior CD1 notebooks), swapping it into
        that column. Under that substitution nothing is null. The counts are
        invariant to it, which is why this keys on them; see the implementation
        note below.

        The fix is a sibling view, deliberately **not** a wider ``values``:
        emitting zeros from ``values`` would change what ``n_shared`` means under
        every card already rendered, which is the exact hazard ``values``' own
        docstring exists to prevent. So:

        * ``values`` and ``declared_zeros`` are **disjoint by construction** —
          this returns only rows ``values`` excluded.
        * a consumer that wants the distinction opts in; one that does not is
          bit-for-bit unaffected.

        Empty for a synthesised operand (``fixture``/``in-investigation``): a
        model has no limit of detection, so its zeros arrive as a real ``0.0``
        centre and ``values`` already keeps them. The asymmetry is real and
        belongs to the measured tier alone.

        Why it matters, concretely: in a ΔtrpR ΔtnaA cultivation the measured
        ``trpR`` is 0.0 TPM across every replicate. It is the single most
        informative row in the comparison — the knockout, visible in the data —
        and today it is the one row that reaches no grader at all.
        """
        cols = self.frame.columns
        if "n_pos" not in cols or "n" not in cols:
            return set()
        # The record of a true zero lives in the COUNTS, not in the centre:
        # `n_pos == 0` (no positive replicate) with `n > 0` (something was
        # actually measured) is the fact. `n > 0` is what separates a measured
        # zero from a row nobody measured.
        #
        # ⚠ Do NOT also test `mean_geometric.isna()`. On the promoted tier the
        # two coincide exactly — measured on a ΔtrpR ΔtnaA transcriptome, both
        # select the same 145 of 4252 rows — so the null test looks free. It is
        # not: a CONSUMER may legitimately present this operand with a different
        # statistic in the `mean_geometric` column (`vs_experiment` does exactly
        # that, substituting the arithmetic centre the cards grade), and then no
        # row is null and this returns EMPTY. Keying on the counts is invariant
        # to that substitution; keying on the centre is not.
        none_positive = pd.to_numeric(
            self.frame["n_pos"], errors="coerce").fillna(-1) == 0
        some_measured = pd.to_numeric(
            self.frame["n"], errors="coerce").fillna(0) > 0
        return {str(e) for e in
                self.frame.loc[none_positive & some_measured, "entity_id"]}

    def __len__(self) -> int:
        return len(self.frame)


def promoted_operand(cultivation_group_id: str, overlay_bundle: Path | str,
                     observable: str, units: str) -> Operand | None:
    """Resolve an operand promoted into a curated payload, or None.

    Selected from the slot's own fields rather than by parsing ``canonical_key``
    — the units component differs by modality (``normalized_abundance`` for the
    proteome, ``TPM`` for the transcriptome, ``counts_per_cell`` for a
    simulation), so key-shape assumptions break.

    Returns None rather than raising when the slot or its rows are absent: a
    report card degrades, it does not fail. That is NOT true of the fixture
    resolver below, and the asymmetry is deliberate — see there.
    """
    from ecoli_sources.validation import (load_vector_observations,
                                          read_vector_observations)
    slots = load_vector_observations(overlay_paths=[overlay_bundle],
                                     include_primary=False)
    slot = next((s for s in slots.values()
                 if str(s.get("cultivation_group_id")) == cultivation_group_id
                 and str(s.get("observable")) == observable
                 and str(s.get("units")) == units), None)
    if slot is None:
        return None
    # validate=False at render time: a payload defect should be caught by CI,
    # not surface as a broken report.
    frame = read_vector_observations(slot, validate=False)
    if frame.empty:
        return None
    kinds = {str(k) for k in frame.get("kind", pd.Series(dtype=str)).unique()}
    return Operand(
        frame=frame,
        path="promoted",
        # One operand should not mix kinds; if a payload ever does, say so
        # rather than silently picking the first.
        kind="+".join(sorted(kinds)) if kinds else "unknown",
        label=f"{cultivation_group_id} {observable} ({units})",
        meta={"cultivation_group_id": cultivation_group_id,
              "observable": observable, "units": units,
              "phase": slot.get("phase"), "window": slot.get("window"),
              "replicate_basis": _replicate_basis(cultivation_group_id,
                                                  overlay_bundle),
              "source": "promoted payload slot"},
    )


def _replicate_basis(cultivation_group_id: str,
                     overlay_bundle: Path | str) -> str | None:
    """What ONE replicate of this cultivation group IS — reactor, flask, well,
    cell or seed — or None when the payload predates the column.

    Carried on the operand because it lives in the cultivation REGISTRY rather
    than the vector table, so nothing downstream could otherwise see it. It is
    resolved and RENDERED here and deliberately goes no further: what a grader
    should DO with it (the ratio band, the degenerate rank test, what replaces
    Mann-Whitney when n=6,000 cells meets n=3 reactors) is a separate change with
    its own rubric argument. This only stops that change being foreclosed.

    Returns None rather than raising on any failure. A missing basis is an
    ordinary state — the column was introduced when `n_reps` was split into a
    count plus a declared basis, and any payload pinned before that split simply
    does not carry it. Reporting that absence as a defect would be wrong."""
    try:
        from ecoli_sources.validation import load_cultivation_groups
        # The registry is a sibling of the bundle, not the bundle itself — the
        # same resolution `vs_experiment._registry_row` uses for the provenance
        # panel, kept identical so the two cannot disagree about which registry
        # describes a group.
        groups = load_cultivation_groups(
            overlay_registries=[Path(overlay_bundle).parent / "cultivations.tsv"])
    except Exception:                                  # noqa: BLE001
        return None
    basis = (groups.get(cultivation_group_id) or {}).get("replicate_basis")
    return str(basis) if basis else None


def fixture_operand(fixtures_dir: Path, fixture: str, map_key: str,
                    *, kind: str = "model_predicted",
                    declared_in: str = "the caller's axis table") -> Operand | None:
    """Resolve an operand from a baked model fixture, or None if absent.

    ``map_key`` is passed in rather than inferred: the proteome fixture names its
    map ``by_id`` (EcoCyc *monomer* ids) and the transcriptome names its
    ``by_gene_id`` (EcoCyc *gene* ids). Those are different identifier spaces, so
    the differing names carry real information — generic code written against one
    silently misses the other.

    A MISSING FIXTURE and a MISSING MAP KEY are different facts and must not
    collapse into the same empty return:

    * fixture absent -> ``None``. Legitimate and expected; the caller declines to
      render rather than half-rendering.
    * fixture present but ``map_key`` absent -> **raise**. That is a contract
      violation between this card and whatever baked the fixture, and returning
      nothing instead would be silent — the join would find zero shared entities
      and the axis would render empty, reading as "the two sides don't overlap"
      rather than "we asked for a key that isn't there".

    The frame synthesised here is a faithful lift, not an invention: a baked
    fixture is an ensemble mean per entity, so every row is ``detected`` (a model
    has no limit of detection) and the centre goes to ``mean_geometric`` because
    that is the field the log-log join reads. ``detection`` therefore carries no
    information on this side — which is true of the promoted simulated operands
    too, and is exactly the asymmetry Layer 2 has to make explicit rather than
    inherit.
    """
    f = Path(fixtures_dir) / fixture
    if not f.is_file():
        return None
    blob = json.loads(f.read_text(encoding="utf-8"))
    if map_key not in blob:
        raise KeyError(
            f"{fixture} carries no '{map_key}' map — it has "
            f"{sorted(k for k in blob if k.startswith('by_'))!r}. The fixture's "
            f"id-map name changed under the card; update the map name declared "
            f"in {declared_in} and re-run the join. "
            f"(fixture id_key: {blob.get('id_key')!r})")
    values = blob.get(map_key) or {}
    if not values:
        return None
    frame = pd.DataFrame({
        "entity_id": [str(k) for k in values],
        "symbol": ["" for _ in values],
        "mean_geometric": [float(v) for v in values.values()],
        "detection": ["detected"] * len(values),
        "kind": [kind] * len(values),
    })
    return Operand(
        frame=frame, path="fixture", kind=kind,
        label=f"{fixture}:{map_key}",
        meta={"id_key": blob.get("id_key"), "n_cells": blob.get("n_cells"),
              "gen_lb": blob.get("gen_lb"), "fixture": fixture,
              "map_key": map_key, "source": "baked model fixture"},
    )


#: Observable name -> the ``(group, name)`` node ``card_vectors.extract_vectors``
#: files it under. Named here rather than taken from the caller so that a study
#: declares an OBSERVABLE ("transcriptome"), the same word the promoted tier uses,
#: and does not have to know the extractor's internal grouping.
_RUN_VECTOR_NODES = {
    "transcriptome": ("omics", "transcriptome"),
    "proteome": ("omics", "proteome"),
}


def _keyed(vector, entity_ids) -> tuple[dict, int, int]:
    """``({entity_id: value}, n_unmapped, n_collisions)``; collisions are SUMMED.

    Deliberately identical to ``scripts/bake_model_omics.py::_keyed``. The live
    path and the bake path must produce the same numbers from the same sweep, or
    "we re-ran it live" silently becomes "we re-ran it live AND changed the
    aggregation". Summing is right for the cistron -> gene case, where several
    mRNA cistrons legitimately map to one EcoCyc gene id and the gene's abundance
    is their total.
    """
    out: dict[str, float] = {}
    n_unmapped = n_collisions = 0
    for k, v in zip(entity_ids, vector):
        k = "" if k is None else str(k)
        if not k or k == "None":
            n_unmapped += 1
            continue
        if k in out:
            n_collisions += 1
        out[k] = out.get(k, 0.0) + float(v)
    return out, n_unmapped, n_collisions


def run_operand(sweep_dir: str | Path, entity_ids,
                *, observable: str = "transcriptome",
                generation_lower_bound: int = 0,
                out_dir: str | None = None,
                kind: str = "model_predicted",
                declared_in: str = "the caller's axis table") -> Operand | None:
    """Resolve an operand from a LIVE run inside the investigation, or None.

    The third path, and the one the module docstring already described ("or
    materialised from an artifact inside the investigation") while no code
    implemented it. A sweep is read through the run-keyed cache
    (``sim_vector_cache.load_or_extract``), so the expensive parquet scan happens
    once per run and every later render is a cache read.

    **``entity_ids`` is passed in, never inferred — and this is the whole
    correctness story, not an ergonomic choice.** The cache stores a POSITIONAL
    vector (``card_vectors`` files ``{"vector": [...], "n_cells": N}``); the ids
    live in the ``sim_data`` that produced the run. For a wild-type arm any
    current ``sim_data`` would do, which is exactly what makes the trap invisible:
    **a ParCa-level knockout splices the genome, so the KO arm's mRNA cistron set
    is NOT the wild type's.** Keying a KO sweep with WT labels silently attributes
    every value past the deleted locus to the wrong gene. The caller holds the
    ``sim_data`` that produced this sweep and is the only party that can be right
    about this, so it supplies the labels — the same reasoning that makes
    ``fixture_operand`` take ``map_key`` rather than guess it.

    Three outcomes, kept distinct because collapsing them is how a comparison
    silently grades nothing:

    * observable not recorded by this sweep -> ``None``. Legitimate; the caller
      declines to render that axis rather than half-rendering it.
    * width mismatch between ``entity_ids`` and the vector -> **raise**. That is
      the KO trap above, and returning ``None`` would present a label/data
      disagreement as an absent measurement.
    * resolved -> an ``Operand`` shaped exactly like the other two paths.

    ⚠ **Provenance is honest about what it cannot know.** ``run_commit`` is
    whatever the sweep actually recorded and is ``None`` when it recorded
    nothing — never the extracting tree's HEAD, which would look authoritative
    and mean something else (``comparison-operands-plan`` §5).

    ⚠ **A run operand is invisible to CI by design.** The cache is machine-local
    (``sim_vector_cache``' own contract), so a card grading this path renders
    where the sweep lives, and its verdict is committed rather than recomputed in
    CI. That is a different integrity story from a promoted or fixture operand
    and belongs in the consuming study's ``assumptions:``.
    """
    from v2ecoli.library.sim_vector_cache import load_or_extract

    node_at = _RUN_VECTOR_NODES.get(observable)
    if node_at is None:
        raise KeyError(
            f"unknown observable {observable!r} for a run operand — known: "
            f"{sorted(_RUN_VECTOR_NODES)}. Declared in {declared_in}.")
    group, name = node_at

    env = load_or_extract(str(sweep_dir), generation_lower_bound, out_dir)
    node = (env.get("vectors", {}).get(group) or {}).get(name)
    if not node:
        return None
    vector = node.get("vector") or []
    if not vector:
        return None

    ids = list(entity_ids)
    if len(ids) != len(vector):
        raise ValueError(
            f"entity_ids/vector width mismatch for {observable!r}: "
            f"{len(ids)} ids vs {len(vector)} values from {sweep_dir}. These "
            f"labels did not come from the sim_data that produced this sweep — "
            f"a ParCa-level knockout changes the cistron set, so a KO arm must "
            f"be keyed with ITS OWN sim_data. Fix the label source declared in "
            f"{declared_in}; do not truncate to the shorter of the two.")

    values, n_unmapped, n_collisions = _keyed(vector, ids)
    if not values:
        return None

    run = env.get("run", {})
    prov = env.get("provenance", {})
    experiment_id = run.get("experiment_id")
    frame = pd.DataFrame({
        "entity_id": list(values),
        "symbol": ["" for _ in values],
        "mean_geometric": [float(v) for v in values.values()],
        # A run has no limit of detection, so every row is `detected` and this
        # column carries no information on this side — the same asymmetry the
        # fixture path has, made explicit rather than inherited.
        "detection": ["detected"] * len(values),
        "kind": [kind] * len(values),
    })
    return Operand(
        frame=frame, path="in-investigation", kind=kind,
        label=f"{experiment_id or Path(sweep_dir).name}:{observable}",
        meta={"source": "live run (run-keyed sim-vector cache)",
              "observable": observable,
              "experiment_id": experiment_id,
              "sweep_dir": str(sweep_dir),
              "gen_lb": int(generation_lower_bound),
              "n_cells": node.get("n_cells"),
              "run_commit": prov.get("run_commit"),
              "extracted_at_commit": prov.get("extracted_at_commit"),
              "extractor_version": (env.get("extractor") or {}).get("version"),
              "n_unmapped": n_unmapped, "n_collisions": n_collisions},
    )


# ── Layer 2 — the symmetric join ──────────────────────────────────────────────

_PPM = 1_000_000.0


def _loglog_r2(sim, exp):
    """Log-log Pearson r² (concordance) over positive pairs — scale-invariant,
    the right stat for absolute abundances on different units."""
    import math
    pairs = [(s, e) for s, e in zip(sim, exp) if s > 0 and e > 0]
    if len(pairs) < 3:
        return None, 0
    xs = [math.log10(e) for _, e in pairs]
    ys = [math.log10(s) for s, _ in pairs]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    r = sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else 0.0
    return r * r, n


#: Kinds whose ``detection`` column carries information. A model has no limit of
#: detection, so a promoted simulation records EVERY entity as ``detected`` — the
#: sim curation states this as "no limit of detection exists". Filtering such an
#: operand on ``detection`` is therefore a no-op that READS as a check, which is
#: worse than not filtering: it implies a test was applied that cannot fail.
#:
#: This is D5 (``omics-vector-payload-plan``) mechanised — a measured zero may sit
#: below a limit of detection, a simulated zero is a real zero.
_DETECTION_INFORMATIVE_KINDS = ("measured",)


def _detection_is_informative(operand) -> bool:
    """Whether ``operand``'s ``detection`` column says anything at all.

    ⚠ This is the ONE sanctioned read of ``kind`` on the join path. ``kind``
    decides how an operand is GRADED; it must never decide how one is LOADED.
    The resolvers above are kind-blind by contract, and a branch on kind
    appearing inside one of them is a bug, not an extension."""
    return str(operand.kind) in _DETECTION_INFORMATIVE_KINDS


#: How each kind is named in rendered prose. The card must be able to describe a
#: comparison whose sides are BOTH measurements (MS#8 #2) or BOTH simulations
#: (MS#8 #8), so the prose takes its nouns from `kind` rather than assuming one
#: side is "the model".
_SIDE_NOUN = {"measured": "the measurement",
              "model_predicted": "the model",
              "theoretical_max": "the theoretical maximum"}


def _side_noun(kind: str) -> str:
    return _SIDE_NOUN.get(str(kind), f"the {kind} side")


def _partition(operand) -> tuple[dict, set, set]:
    """``(usable_rows, declared_absent, id_space)`` for one operand.

    * ``usable_rows`` — ``entity_id -> row``, the rows whose value may enter a fit.
    * ``declared_absent`` — ids this operand looked for and did NOT find. Empty
      whenever ``detection`` is uninformative for this kind, which is the honest
      encoding: a simulation makes no absence claims, so it declares none.
    * ``id_space`` — every id the operand carries a row for, usable or not. This
      is deliberately the FRAME's ids rather than the usable ones: an entity whose
      centre is null is one the operand knows about but cannot quantify, which is
      a different fact from one it never covered, and collapsing them is exactly
      the conflation this function exists to undo."""
    informative = _detection_is_informative(operand)
    usable, absent = {}, set()
    for row in operand.frame.to_dict("records"):
        eid = str(row["entity_id"])
        if informative and str(row.get("detection") or "") != "detected":
            absent.add(eid)
            continue
        usable[eid] = row
    return usable, absent, usable.keys() | absent


def _declared_absent_within(absent: set, other_id_space: set) -> int:
    """How many of ``absent`` the other operand actually covers.

    An absence claim is only *evidence* where the other side has something to
    disagree with; outside its id-space it is just two panels differing."""
    return sum(1 for e in absent if e in other_id_space)


def _join_vectors(a, b) -> dict:
    """Join two operands on their shared id-space and renormalize. **Symmetric.**

    Nothing here distinguishes "the model side" from "the reference side". Both
    operands are filtered by THEIR OWN rule, both contribute coverage counts, and
    both are checked for a usable value. The only asymmetry left is
    ``n_provisional``, and it is genuinely one-sided rather than an oversight —
    see the note where it is counted.

    Returns the two renormalized vectors as ``a`` and ``b``, and each side's
    panel sizes as ``n_{a,b}_rows`` / ``n_{a,b}_detected``. ``exp``, ``sim``,
    ``n_measured`` and ``n_detected`` are **deprecated aliases** for the side-A
    fields, retained only so a vendoring tree can migrate independently of a
    sync; new callers should not use them.

    Order matters and is the decided one, and the id-space test comes FIRST:
    filter each operand by its own detection rule → intersect the id-spaces →
    renormalize BOTH sides to ppm over that shared set → let the caller take the
    log-log R² over ``mean_geometric``.

    Renormalizing *after* the intersection is what makes the comparison
    scale-free: both vectors then sum to 1e6 over exactly the same entities, so
    R² measures agreement in relative abundance rather than in panel coverage.

    ``mean_geometric`` is the log-space centre — the only mean coherent with the
    table's ``sd_log10``, and the log-log scatter is what the card grades.
    ``mean_arithmetic`` is for additive uses only.

    ⚠ **Why side B is checked for positivity at all**, given a fixture B is always
    positive: ``_loglog_r2`` filters non-positive pairs INTERNALLY and does so
    AFTER ``_ppm``. So an unchecked non-positive B value would still be counted in
    ``n_shared``, still shift every other B value through the ppm denominator, and
    then vanish from the fit with nothing recording the gap. That is a latent
    correctness defect the moment a promoted simulation appears on either side,
    not merely missing bookkeeping."""
    a_rows, a_absent, a_ids = _partition(a)
    b_rows, b_absent, b_ids = _partition(b)

    ids, syms, a_raw, b_raw, prov = [], [], [], [], 0
    a_outside = b_outside = a_nonpos = b_nonpos = 0

    def _positive(row) -> bool:
        # NaN is the only value that is not equal to itself, which is how a null
        # centre arrives here once pandas has been through it. Checked without
        # pandas so this module keeps its (deliberately small) import surface.
        v = row.get("mean_geometric")
        return v is not None and v == v and float(v) > 0

    # Buckets are EXCLUSIVE and ordered: an entity outside the other id-space is
    # never also counted as non-positive. The id-space test first is what keeps
    # this inert against the pre-symmetry behaviour, which tested membership
    # before it tested the value.
    for eid, row in a_rows.items():
        if eid not in b_ids:
            a_outside += 1
        elif eid in b_absent:
            # Already carried by `n_declared_absent_by_b`. Falling through to the
            # non-positive bucket would count one entity under two different
            # facts, which breaks the mirror property: "B looked and did not find
            # it" and "B has no usable value for it" are different claims and the
            # first is the more specific.
            pass
        elif not _positive(row):
            # Null/zero centre: `mean_geometric` is null when no replicate was
            # positive, i.e. the entity was detected and genuinely measured as
            # zero. A zero has no place on a log axis, but it is a real result,
            # not a gap, so it is counted rather than folded in with coverage.
            a_nonpos += 1
        elif eid not in b_rows or not _positive(b_rows[eid]):
            b_nonpos += 1
        else:
            ids.append(eid)
            syms.append(str(row.get("symbol") or "") or eid)
            a_raw.append(float(row["mean_geometric"]))
            b_raw.append(float(b_rows[eid]["mean_geometric"]))
            # One-sided ON PURPOSE: `zeros_excluded_provisional` is a note about
            # how raw DIA zeros were treated by an instrument pipeline. It is a
            # property of a measurement's processing, not a fact both sides of a
            # comparison can have, so mirroring it would invent a claim.
            if str(row.get("notes") or "") == "zeros_excluded_provisional":
                prov += 1

    for eid in b_rows:
        if eid not in a_ids:
            b_outside += 1

    def _ppm(vec):
        tot = sum(vec)
        return [v * _PPM / tot for v in vec] if tot > 0 else vec

    a_ppm, b_ppm = _ppm(a_raw), _ppm(b_raw)

    return {
        "ids": ids, "symbols": syms,
        # ``a``/``b`` match the convention every other paired field in this dict
        # already uses (``kind_a``, ``n_nonpositive_b``, ``detection_informative_a``
        # …). The two vectors and the two panel counts were the last fields named
        # for a measurement-vs-model split this function's own docstring says it
        # does not make — and the names were not merely untidy:
        #
        #   * the FIRST operand was returned under the key ``exp``, regardless of
        #     kind, so a consumer plotting ``exp`` on the x-axis mislabelled its
        #     axes whenever the caller passed the model first (R² is symmetric,
        #     so no score was ever wrong — only the labels);
        #   * a sim↔sim join reported ``n_measured`` about a simulation.
        "a": a_ppm, "b": b_ppm,
        "n_shared": len(ids),
        # Panel sizes, now reported for BOTH sides. Only side A's were exposed
        # before, which is itself the asymmetry: a reader could see how much of
        # the first operand's panel survived detection and not the second's.
        "n_a_rows": int(len(a.frame)), "n_a_detected": int(len(a_rows)),
        "n_b_rows": int(len(b.frame)), "n_b_detected": int(len(b_rows)),
        # DEPRECATED aliases — identical values, kept so a downstream tree that
        # vendors this file can migrate on its own schedule instead of in
        # lockstep with a sync. Remove once sms-ecoli's vs_experiment.py reads
        # the symmetric names.
        "exp": a_ppm, "sim": b_ppm,
        "n_measured": int(len(a.frame)),
        "n_detected": int(len(a_rows)),
        # Coverage: present and usable on one side, no row at all on the other.
        "n_a_outside_b_idspace": a_outside,
        "n_b_outside_a_idspace": b_outside,
        # Declared absence: the side LOOKED and did not find it, and the other
        # side covers it. Structurally zero for any operand whose `detection` is
        # uninformative — rendered anyway, with the reason, because a reader who
        # cannot see the count cannot tell "no disagreement" from "this
        # comparison cannot express disagreement".
        "n_declared_absent_by_a": _declared_absent_within(a_absent, b_ids),
        "n_declared_absent_by_b": _declared_absent_within(b_absent, a_ids),
        # Covered by the other side, but this side has no usable (positive) value.
        "n_nonpositive_a": a_nonpos,
        "n_nonpositive_b": b_nonpos,
        "n_provisional": prov,
        # Grading inputs the prose needs to explain a structural zero.
        "kind_a": str(a.kind), "kind_b": str(b.kind),
        "detection_informative_a": _detection_is_informative(a),
        "detection_informative_b": _detection_is_informative(b),
    }

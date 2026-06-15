"""Stage 1 sanity-check parameter overrides.

Mutates ``raw_data`` in memory after ``KnowledgeBaseEcoli`` loads the
canonical flat TSVs and before any ParCa step runs. Canonical TSVs are
never modified — the experimental delta lives entirely in this file.

Source: ``Parameters for WCM (Stage 1: heuristic values) - Stage 1.pdf``
(see ``docs/stage1_parameter_comparison.html`` for the full
parameter-by-parameter comparison against v2ecoli defaults).

Final canonical config: ``STAGE1_OVERRIDE_SKIP=d_period``
=========================================================
Stage 1 prescribed both C=70 min (via replisome_rate) AND D=30 min. The
diagnostic in ``logs/diag*.log`` showed that the two timing parameters
TOGETHER make ParCa's Step 6 ECOS promoter-binding solver infeasible
(each alone passes; the pair fails; the global C+D=100 min change shifts
Cooper-Helmstetter DNA mass at every condition in a way the P-fit can't
absorb across all 43 condition keys).

We drop the D=30 override and keep:
  - replisome_rate=552.58 nt/s   → C ≈ 70 min (Stage 1's slow replication)
  - dnaA translation_efficiency=1.0  (Hansen & Atlung 2018 "1 protein/mRNA")
  - DARS1/DARS2 widening + DATA row (dormant data, no current consumer)
  - D=20 (canonical, kept)

Effective C+D = 90 min; non-overlapping at acetate (136 min doubling).

Selective application
=====================
Each override has a string tag. To skip one, set
``STAGE1_OVERRIDE_SKIP=<tag1>,<tag2>,...`` in the environment before
running ``v2ecoli-parca``. Used for pinpointing which override breaks
ParCa Step 6 (promoter binding ECOS solver). Tags:

  d_period           — override 1
  replisome_rate     — override 2
  translation_eff    — override 3
  dars1              — override 4 (dormant data, never breaks anything)
  dars2              — override 5 (dormant data)
  data_row           — override 6 (dormant data)

The ``with_aa`` doubling-time bump is a forced side-effect of
``d_period``; it auto-applies only when ``d_period`` is enabled.

Overrides applied (when not skipped):

  1. d_period                  : 20 min       → 30 min
  2. replisome_elongation_rate : 967 nt/s     → 552.58 nt/s
                                 (gives C ≈ 70 min from genome length
                                 4,641,652 nt / rate / 2 forks)
  3. dnaA translation eff.     : 0.35         → 1.0  (Hansen & Atlung 2018)
  4. DARS1 window              : 813107-813141 → 813086-813186
  5. DARS2 window              : 2969135-2969169 → 2969112-2969367
  6. add DATA row              : 4392732-4392914 (extragenic-site)

Loaded by ``v2ecoli/cli/parca.py`` via ``--overrides-module
studies.stage1_sanity_check.overrides``.
"""

from __future__ import annotations

import os

from v2ecoli.processes.parca.wholecell.utils import units


def _skip_set() -> set[str]:
    raw = os.environ.get("STAGE1_OVERRIDE_SKIP", "")
    return {tag.strip() for tag in raw.split(",") if tag.strip()}


def _bump_conditions() -> dict[str, float]:
    """Diagnostic-only knob. Parse ``STAGE1_BUMP_CONDITIONS=basal=110,with_aa=110``."""
    raw = os.environ.get("STAGE1_BUMP_CONDITIONS", "")
    out: dict[str, float] = {}
    for token in raw.split(","):
        token = token.strip()
        if not token or "=" not in token:
            continue
        name, val = token.split("=", 1)
        out[name.strip()] = float(val)
    return out


def apply(raw) -> None:
    """Mutate ``raw`` in place. Honors ``STAGE1_OVERRIDE_SKIP`` env var."""

    skip = _skip_set()
    bumps = _bump_conditions()
    applied: list[str] = []
    skipped: list[str] = []

    # Diagnostic-only: bump specific conditions' doubling times.
    # Used to test which conditions drive the Step 6 ECOS failure under
    # the d_period + replisome_rate combo (C+D = 100 min). Conditions with
    # τ < 100 have overlapping replication under Stage 1 timing; bumping
    # them to τ ≥ 100 removes the overlap.
    for cond_name, new_tau in bumps.items():
        row = next(
            (r for r in raw.condition.condition_defs if r.get("condition") == cond_name),
            None,
        )
        if row is None:
            raise RuntimeError(f"bump: condition {cond_name!r} not in condition_defs")
        row["doubling time"] = float(new_tau) * units.min
        applied.append(f"BUMP {cond_name} doubling time → {new_tau:.1f} min (diagnostic)")

    # 1: d_period (parameters.tsv)
    if "d_period" in skip:
        skipped.append("d_period")
    else:
        if "d_period" not in raw.parameters:
            raise RuntimeError("expected raw.parameters['d_period']")
        raw.parameters["d_period"] = 30.0 * units.min
        applied.append("d_period:                    20 min → 30 min")

        # Side-effect of D=30 global: with_aa's 25-min doubling time
        # now trips the ``if doubling_time < d_period: raise`` guard in
        # ``growth_rate_dependent_parameters.py:450``. Bump with_aa to
        # 60 min — its post-fit values are never consumed by acetate.
        if "with_aa" not in bumps:
            with_aa = next(
                (r for r in raw.condition.condition_defs if r.get("condition") == "with_aa"),
                None,
            )
            if with_aa is None:
                raise RuntimeError("expected with_aa row in raw.condition.condition_defs")
            with_aa["doubling time"] = 60.0 * units.min
            applied.append("with_aa doubling time:       25 min → 60 min (side-effect of D=30)")

    # 2: replisome_elongation_rate (parameters.tsv)
    if "replisome_rate" in skip:
        skipped.append("replisome_rate")
    else:
        if "replisome_elongation_rate" not in raw.parameters:
            raise RuntimeError(
                "expected raw.parameters['replisome_elongation_rate']"
            )
        raw.parameters["replisome_elongation_rate"] = 552.58 * (units.nt / units.s)
        applied.append("replisome_elongation_rate:   967 → 552.58 nt/s (C ≈ 70 min)")

    # 3: dnaA translation efficiency (translation_efficiency.tsv)
    # Columns: geneId, name, translationEfficiency
    if "translation_eff" in skip:
        skipped.append("translation_eff")
    else:
        dnaA = next(
            (r for r in raw.translation_efficiency if r.get("geneId") == "EG10235"),
            None,
        )
        if dnaA is None:
            raise RuntimeError(
                "expected EG10235 (dnaA) row in raw.translation_efficiency"
            )
        dnaA["translationEfficiency"] = 1.0
        applied.append("dnaA translation_efficiency: 0.35 → 1.0")

    # 4: dna_sites.tsv DARS1 window
    if "dars1" in skip:
        skipped.append("dars1")
    else:
        dars1 = next((r for r in raw.dna_sites if r.get("id") == "DARS1"), None)
        if dars1 is None:
            raise RuntimeError("expected DARS1 row in raw.dna_sites")
        dars1["left_end_pos"] = 813086
        dars1["right_end_pos"] = 813186
        applied.append("DARS1 window:                813107-813141 → 813086-813186")

    # 5: dna_sites.tsv DARS2 window
    if "dars2" in skip:
        skipped.append("dars2")
    else:
        dars2 = next((r for r in raw.dna_sites if r.get("id") == "DARS2"), None)
        if dars2 is None:
            raise RuntimeError("expected DARS2 row in raw.dna_sites")
        dars2["left_end_pos"] = 2969112
        dars2["right_end_pos"] = 2969367
        if dars2.get("common_name") is None:
            dars2["common_name"] = "DARS2"
        applied.append("DARS2 window:                2969135-2969169 → 2969112-2969367")

    # 6: dna_sites.tsv add DATA row
    if "data_row" in skip:
        skipped.append("data_row")
    else:
        if any(r.get("id") == "DATA" for r in raw.dna_sites):
            raise RuntimeError("DATA row already present in raw.dna_sites")
        raw.dna_sites.append(
            {
                "id": "DATA",
                "common_name": "datA",
                "synonyms": ["datA locus"],
                "type": "extragenic-site",
                "left_end_pos": 4392732,
                "right_end_pos": 4392914,
                "direction": None,
            }
        )
        applied.append("DATA row:                    added (4392732-4392914)")

    print(f"[stage1_sanity_check] applied {len(applied)} overrides "
          f"(skipped {sorted(skipped) if skipped else 'none'}):")
    for line in applied:
        print(f"    {line}")

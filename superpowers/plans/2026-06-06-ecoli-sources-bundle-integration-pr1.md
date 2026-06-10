# ecoli-sources Bundle Integration (PR 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make v2ecoli's ParCa read its flat input data from the external `ecoli-sources` package via a canonical-key bundle resolver, keeping v2ecoli's 3 divergent biology files as local overrides, with byte-identical ParCa output.

**Architecture:** A new `SourceBundle` resolver merges the ecoli-sources default manifest (`reference_bundle.tsv`) with a 3-row v2ecoli override spec into one effective `{canonical_key → absolute_path}` index. `KnowledgeBaseEcoli` is refactored so its file loaders take a *logical relative path* (for attribute nesting, unchanged) plus a *physical absolute path* (resolved through the bundle). Because the attribute tree is still derived from the same logical paths, the produced `raw_data` is identical by construction; a deep-compare parity test proves it before the 130 local files are deleted.

**Tech Stack:** Python, pandas (already a dep), `ecoli-sources` (new git-pinned dep, brings pandera transitively), pytest, uv.

---

## Spec reference

`docs/superpowers/specs/2026-06-06-ecoli-sources-bundle-integration-design.md`

## File structure

- **Create** `v2ecoli/processes/parca/reconstruction/ecoli/sources.py` — `SourceBundle` resolver + `relpath_to_key` helper. One responsibility: canonical-key/relpath → absolute path.
- **Create** `v2ecoli/processes/parca/reconstruction/ecoli/parca_overrides.tsv` — 3-row v2ecoli override manifest.
- **Create** `v2ecoli/processes/parca/reconstruction/ecoli/flat_overrides/` — the 3 divergent files, relocated from `flat/`.
- **Modify** `v2ecoli/processes/parca/reconstruction/ecoli/knowledge_base_raw.py` — loaders take `(rel_path, abs_path)`; constructor takes `bundle`; load loops + new-gene checks resolve through the bundle.
- **Modify** `v2ecoli/cli/parca.py` — `--bundle-manifest-path` arg, thread to KB.
- **Modify** `pyproject.toml` — add `ecoli-sources` dep + `[tool.uv.sources]` pin; update `[tool.setuptools.package-data]`.
- **Create** `tests/test_source_bundle.py` — resolver unit + bundle-parity tests.
- **Create** `tests/test_kb_bundle_parity.py` — KB deep-compare parity (legacy vs bundle).
- **Delete** the 130 byte-identical files under `flat/` (Task 8).

---

## Task 1: Add ecoli-sources dependency

**Files:**
- Modify: `pyproject.toml:11-13` (dependencies), `pyproject.toml:92-94` (`[tool.uv.sources]`)

- [ ] **Step 1: Pick the pin commit**

Run:
```bash
gh api repos/vivarium-collective/ecoli-sources/commits/main --jq '.sha'
```
Record the full SHA as `<ECOLI_SOURCES_SHA>` for the next step.

- [ ] **Step 2: Add the dependency and the uv source pin**

In `pyproject.toml`, add to `[project].dependencies` (after `"pandas",` at line ~33):
```toml
    "ecoli-sources",
```
Add to `[tool.uv.sources]` (after the bigraph-schema line):
```toml
ecoli-sources = { git = "https://github.com/vivarium-collective/ecoli-sources.git", rev = "<ECOLI_SOURCES_SHA>" }
```

- [ ] **Step 3: Install and verify the import resolves**

Run:
```bash
uv pip install -e . 2>&1 | tail -5
.venv/bin/python -c "import ecoli_sources, schemas; from ecoli_sources import BUNDLE_PATH, DATA_DIR; print('BUNDLE_PATH', BUNDLE_PATH.is_file()); print('schemas', hasattr(__import__('schemas'), 'ReferenceBundleSchema'))"
```
Expected: `BUNDLE_PATH True` and `schemas True`.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build(parca): add git-pinned ecoli-sources dependency"
```

---

## Task 2: SourceBundle resolver

**Files:**
- Create: `v2ecoli/processes/parca/reconstruction/ecoli/sources.py`
- Test: `tests/test_source_bundle.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_source_bundle.py
import os
from pathlib import Path

import pytest

from v2ecoli.processes.parca.reconstruction.ecoli.sources import (
    SourceBundle,
    relpath_to_key,
)


def test_relpath_to_key_strips_ext_and_joins_with_double_underscore():
    assert relpath_to_key("genes.tsv") == "genes"
    assert relpath_to_key(os.path.join("condition", "media", "MIX0-55.tsv")) == "condition__media__MIX0-55"
    assert relpath_to_key(os.path.join("cell_wall", "murein_strand_length_distribution.csv")) == "cell_wall__murein_strand_length_distribution"


def test_default_bundle_resolves_a_known_key():
    b = SourceBundle()
    p = b.path("genes")
    assert p.is_file()


def test_resolve_relpath_routes_through_key():
    b = SourceBundle()
    assert b.resolve_relpath("genes.tsv") == b.path("genes")


def test_missing_key_raises_naming_the_key():
    b = SourceBundle()
    with pytest.raises(KeyError, match="no_such_key"):
        b.path("no_such_key")


def test_override_replaces_base_row(tmp_path):
    # base manifest with one key
    data_root = tmp_path / "data"
    (data_root / "flat").mkdir(parents=True)
    (data_root / "flat" / "genes.tsv").write_text("base")
    base = data_root / "reference_bundle.tsv"
    base.write_text("canonical_key\tsource_path\tdescription\tschema_name\n"
                    "genes\tflat/genes.tsv\tg\t\n")
    # override pointing genes elsewhere
    ov_root = tmp_path / "ov"
    (ov_root / "flat_overrides").mkdir(parents=True)
    (ov_root / "flat_overrides" / "genes.tsv").write_text("override")
    ov = ov_root / "parca_overrides.tsv"
    ov.write_text("canonical_key\tsource_path\tdescription\tschema_name\n"
                  "genes\tflat_overrides/genes.tsv\tg\t\n")

    b = SourceBundle(base_manifest=base, overrides=ov, validate=False)
    assert b.path("genes").read_text() == "override"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_source_bundle.py -q`
Expected: FAIL with `ModuleNotFoundError: ...sources`.

- [ ] **Step 3: Implement the resolver**

```python
# v2ecoli/processes/parca/reconstruction/ecoli/sources.py
"""
Resolver for the ecoli-sources data bundle.

A *bundle* maps each ``canonical_key`` (an addressable data role in ParCa) to a
source file. The default reference bundle ships with ``ecoli-sources``
(``ecoli_sources.BUNDLE_PATH``). v2ecoli layers a small override manifest on
top so its locally-diverged flat files (equilibrium / metabolism biology) win
over the upstream defaults without copying the whole 135-key manifest.

Ported and adapted from CovertLab/vEcoli's ``wholecell/io/sources.py``
(PR #426); the override-merge is a v2ecoli addition.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

import pandas as pd

PathLike = Union[str, os.PathLike]

# Default location of the v2ecoli override spec (sibling of this module).
_DEFAULT_OVERRIDES = Path(__file__).resolve().parent / "parca_overrides.tsv"


def relpath_to_key(rel_path: str) -> str:
    """Map a flat-relative file path to its bundle canonical key.

    ``condition/media/MIX0-55.tsv`` -> ``condition__media__MIX0-55``. Strips a
    single trailing extension and replaces path separators with ``__``.
    """
    norm = rel_path.replace(os.sep, "/")
    stem, _, _ext = norm.rpartition(".")
    norm = stem if stem else norm  # no extension -> keep as-is
    return norm.replace("/", "__")


class SourceBundle:
    """Resolve canonical keys / flat-relpaths to absolute source paths."""

    def __init__(
        self,
        base_manifest: Optional[PathLike] = None,
        overrides: Optional[PathLike] = None,
        validate: bool = True,
    ):
        if base_manifest is None:
            from ecoli_sources import BUNDLE_PATH
            base_manifest = BUNDLE_PATH
        base_manifest = Path(base_manifest).resolve()
        if not base_manifest.is_file():
            raise FileNotFoundError(f"Bundle manifest not found: {base_manifest}")

        index: dict[str, Path] = {}
        base_root = base_manifest.parent
        index.update(self._read_manifest(base_manifest, base_root))

        if overrides is None and _DEFAULT_OVERRIDES.is_file():
            overrides = _DEFAULT_OVERRIDES
        if overrides is not None:
            overrides = Path(overrides).resolve()
            index.update(self._read_manifest(overrides, overrides.parent))

        self._index = index
        if validate:
            self._validate(base_manifest, overrides)

    @staticmethod
    def _read_manifest(manifest: Path, root: Path) -> dict[str, Path]:
        df = pd.read_csv(manifest, sep="\t", comment="#")
        out: dict[str, Path] = {}
        for _, row in df.iterrows():
            out[str(row["canonical_key"])] = (root / str(row["source_path"])).resolve()
        return out

    def _validate(self, base_manifest: Path, overrides: Optional[Path]) -> None:
        # Best-effort: reuse ecoli-sources' Pandera schema on the merged set;
        # always verify every resolved path exists.
        try:
            from schemas import ReferenceBundleSchema  # ecoli-sources package
            rows = [{"canonical_key": k, "source_path": str(p)} for k, p in self._index.items()]
            ReferenceBundleSchema.validate(pd.DataFrame(rows), lazy=True)
        except ImportError:
            pass
        missing = {k: p for k, p in self._index.items() if not p.is_file()}
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} bundle key(s) resolve to missing files: "
                f"{sorted(missing)[:5]}..."
            )

    def path(self, canonical_key: str) -> Path:
        try:
            return self._index[canonical_key]
        except KeyError:
            raise KeyError(f"canonical_key not in bundle: {canonical_key}")

    def resolve_relpath(self, rel_path: str) -> Path:
        return self.path(relpath_to_key(rel_path))

    def has_key(self, canonical_key: str) -> bool:
        return canonical_key in self._index

    def keys_with_prefix(self, prefix: str) -> list[str]:
        return [k for k in self._index if k.startswith(prefix)]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_source_bundle.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add v2ecoli/processes/parca/reconstruction/ecoli/sources.py tests/test_source_bundle.py
git commit -m "feat(parca): SourceBundle resolver for ecoli-sources (base+override merge)"
```

---

## Task 3: v2ecoli override spec + relocated divergent files

**Files:**
- Create: `v2ecoli/processes/parca/reconstruction/ecoli/flat_overrides/{equilibrium_reactions.tsv,equilibrium_reaction_rates.tsv,metabolic_reactions_added.tsv}`
- Create: `v2ecoli/processes/parca/reconstruction/ecoli/parca_overrides.tsv`
- Test: append to `tests/test_source_bundle.py`

- [ ] **Step 1: Copy the 3 divergent files into flat_overrides/**

Run:
```bash
cd /Users/eranagmon/code/v2ecoli
F=v2ecoli/processes/parca/reconstruction/ecoli
mkdir -p $F/flat_overrides
cp $F/flat/equilibrium_reactions.tsv $F/flat/equilibrium_reaction_rates.tsv $F/flat/metabolic_reactions_added.tsv $F/flat_overrides/
ls $F/flat_overrides/
```
Expected: the 3 files listed.

- [ ] **Step 2: Write the override manifest**

Create `v2ecoli/processes/parca/reconstruction/ecoli/parca_overrides.tsv` (tab-separated):
```
canonical_key	source_path	description	schema_name
equilibrium_reactions	flat_overrides/equilibrium_reactions.tsv	v2ecoli DnaA-ATP hydrolysis equilibrium (PR #123)	
equilibrium_reaction_rates	flat_overrides/equilibrium_reaction_rates.tsv	v2ecoli DnaA-ATP hydrolysis rates (PR #123)	
metabolic_reactions_added	flat_overrides/metabolic_reactions_added.tsv	v2ecoli metabolic additions (v2parca merge #16)	
```

- [ ] **Step 3: Write the bundle-parity test**

Append to `tests/test_source_bundle.py`:
```python
import hashlib

V2_FLAT = Path(__file__).resolve().parents[1] / "v2ecoli/processes/parca/reconstruction/ecoli/flat"
OVERRIDE_KEYS = {"equilibrium_reactions", "equilibrium_reaction_rates", "metabolic_reactions_added"}


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def test_override_keys_point_to_local_flat_overrides():
    b = SourceBundle()
    for key in OVERRIDE_KEYS:
        p = b.path(key)
        assert "flat_overrides" in str(p), f"{key} should resolve to a local override"
        assert p.is_file()


@pytest.mark.skipif(not V2_FLAT.exists(), reason="local flat/ deleted post-migration")
def test_inherited_keys_match_ecoli_sources_content():
    # For every still-present local flat file NOT overridden, the bundle's
    # resolved file must be byte-identical (guards against upstream drift).
    b = SourceBundle()
    mismatches = []
    for f in V2_FLAT.rglob("*"):
        if not f.is_file() or f.name == "sequence.fasta":
            continue
        rel = f.relative_to(V2_FLAT)
        key = relpath_to_key(str(rel))
        if key in OVERRIDE_KEYS or not b.has_key(key):
            continue
        if _sha(f) != _sha(b.path(key)):
            mismatches.append(key)
    assert mismatches == [], f"bundle content drifted from local flat for: {mismatches}"
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_source_bundle.py -q`
Expected: PASS. `test_inherited_keys_match_ecoli_sources_content` must report **no mismatches** (this is the content-audit gate confirming only the 3 override files differ).

- [ ] **Step 5: Commit**

```bash
git add v2ecoli/processes/parca/reconstruction/ecoli/flat_overrides v2ecoli/processes/parca/reconstruction/ecoli/parca_overrides.tsv tests/test_source_bundle.py
git commit -m "feat(parca): v2ecoli override bundle keeps diverged equilibrium/metabolism files"
```

---

## Task 4: Refactor KnowledgeBaseEcoli loaders to take (rel_path, abs_path)

**Files:**
- Modify: `v2ecoli/processes/parca/reconstruction/ecoli/knowledge_base_raw.py:351-393` (`_load_tsv`, `_load_parameters`)

This task is a pure refactor that preserves behaviour: attribute nesting is derived from the *logical relative path* instead of slicing `file_name` against `dir_name`. No test flips yet; the existing legacy path must still pass.

- [ ] **Step 1: Replace `_load_tsv` and `_load_parameters` signatures**

In `knowledge_base_raw.py`, replace `_load_tsv` (lines 351-361) with:
```python
    def _load_tsv(self, rel_path, abs_path):
        path = self
        parts = rel_path.replace(os.sep, "/").split("/")
        for sub_path in parts[:-1]:
            if not hasattr(path, sub_path):
                setattr(path, sub_path, DataStore())
            path = getattr(path, sub_path)
        attr_name = parts[-1].split(".")[0]
        setattr(path, attr_name, [])

        rows = read_tsv(str(abs_path))
        setattr(path, attr_name, rows)
```
Replace `_load_parameters` (lines 370-393) header + path navigation similarly:
```python
    def _load_parameters(self, rel_path, abs_path):
        path = self
        parts = rel_path.replace(os.sep, "/").split("/")
        for sub_path in parts[:-1]:
            if not hasattr(path, sub_path):
                setattr(path, sub_path, DataStore())
            path = getattr(path, sub_path)
        attr_name = parts[-1].split(".")[0]
        param_dict = {}

        with io.open(str(abs_path), "rb") as csvfile:
            reader = tsv.dict_reader(csvfile)
            for row in reader:
                value = json.loads(row["value"])
                if row["units"] != "":
                    unit = eval(row["units"])  # risky!
                    unit = units.getUnit(unit)  # strip
                    value = value * unit
                param_dict[row["name"]] = value

        setattr(path, attr_name, param_dict)
```

- [ ] **Step 2: Update the three load-loop call sites (legacy path) to the new signature**

In `__init__` (lines 315-324), change the loops to pass `(rel, abs)` using the legacy `FLAT_DIR` for now:
```python
        for filename in self.list_of_dict_filenames:
            self._load_tsv(filename, os.path.join(FLAT_DIR, filename))

        for filename in self.list_of_parameter_filenames:
            self._load_parameters(filename, os.path.join(FLAT_DIR, filename))

        self.genome_sequence = self._load_sequence(
            os.path.join(FLAT_DIR, SEQUENCE_FILE)
        )
```

- [ ] **Step 3: Run an existing ParCa-touching test to confirm no behaviour change**

Run: `.venv/bin/python -m pytest tests/ -q -k "parca or knowledge or raw" 2>&1 | tail -15`
Expected: same pass/fail set as before this task (no new failures). If none match, run a smoke KB build:
```bash
.venv/bin/python -c "from v2ecoli.processes.parca.reconstruction.ecoli.knowledge_base_raw import KnowledgeBaseEcoli as K; kb=K(operons_on=True, remove_rrna_operons=False, remove_rrff=False, stable_rrna=False); print('genes', len(kb.genes), 'tu', len(kb.transcription_units))"
```
Expected: prints non-zero counts.

- [ ] **Step 4: Commit**

```bash
git add v2ecoli/processes/parca/reconstruction/ecoli/knowledge_base_raw.py
git commit -m "refactor(parca): KB loaders take (rel_path, abs_path); attr tree from logical path"
```

---

## Task 5: Wire the bundle into KnowledgeBaseEcoli + parity test

**Files:**
- Modify: `knowledge_base_raw.py` (constructor `bundle` param; resolve loads + new-gene checks via bundle)
- Test: `tests/test_kb_bundle_parity.py`

- [ ] **Step 1: Write the failing parity test (deep compare legacy vs bundle)**

```python
# tests/test_kb_bundle_parity.py
import pytest

from v2ecoli.processes.parca.reconstruction.ecoli.knowledge_base_raw import (
    KnowledgeBaseEcoli,
)
from v2ecoli.processes.parca.reconstruction.ecoli.sources import SourceBundle


def _snapshot(kb):
    """Flatten KB into a comparable dict of {attr_path: value}."""
    from v2ecoli.processes.parca.reconstruction.ecoli.knowledge_base_raw import DataStore
    out = {}

    def walk(obj, prefix):
        for name, val in vars(obj).items():
            key = f"{prefix}.{name}" if prefix else name
            if isinstance(val, DataStore):
                walk(val, key)
            else:
                out[key] = val
    walk(kb, "")
    return out


FLAGS = dict(operons_on=True, remove_rrna_operons=False, remove_rrff=False, stable_rrna=False)


def test_bundle_kb_matches_legacy_kb():
    legacy = KnowledgeBaseEcoli(**FLAGS)              # reads local flat/
    bundled = KnowledgeBaseEcoli(bundle=SourceBundle(), **FLAGS)
    a, b = _snapshot(legacy), _snapshot(bundled)
    assert a.keys() == b.keys(), set(a) ^ set(b)
    diffs = [k for k in a if repr(a[k]) != repr(b[k])]
    assert diffs == [], f"raw_data differs for: {diffs[:10]}"
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_kb_bundle_parity.py -q`
Expected: FAIL — `KnowledgeBaseEcoli.__init__() got an unexpected keyword argument 'bundle'`.

- [ ] **Step 3: Add `bundle` param and route loads through it**

In `knowledge_base_raw.py` `__init__` signature (line 176-183) add `bundle=None`:
```python
    def __init__(
        self,
        operons_on: bool,
        remove_rrna_operons: bool,
        remove_rrff: bool,
        stable_rrna: bool,
        new_genes_option: str = "off",
        bundle=None,
    ):
        self._bundle = bundle
```
Add a resolver helper (method on the class):
```python
    def _resolve(self, rel_path):
        if self._bundle is not None:
            return self._bundle.resolve_relpath(rel_path)
        return os.path.join(FLAT_DIR, rel_path)
```
Change the three load loops (Task 4 Step 2 location) to:
```python
        for filename in self.list_of_dict_filenames:
            self._load_tsv(filename, self._resolve(filename))

        for filename in self.list_of_parameter_filenames:
            self._load_parameters(filename, self._resolve(filename))

        self.genome_sequence = self._load_sequence(self._resolve(SEQUENCE_FILE))
```
Change the new-gene existence checks (lines 273-306) to consult the bundle when present:
```python
        if self.new_genes_option != "off":
            new_gene_subdir = new_genes_option
            new_gene_path = os.path.join("new_gene_data", new_gene_subdir)
            if self._bundle is not None:
                assert self._bundle.keys_with_prefix(
                    f"new_gene_data__{new_gene_subdir}__"
                ), "This new_genes_data subdirectory is invalid."
            else:
                assert os.path.isdir(os.path.join(FLAT_DIR, new_gene_path)), (
                    "This new_genes_data subdirectory is invalid."
                )
```
And the per-file new-gene checks (the `genes/rnas/proteins/...` loop and the optional rnaseq file) — replace each `os.path.isfile(os.path.join(FLAT_DIR, file_path))` with:
```python
                if self._bundle is not None:
                    present = self._bundle.has_key(relpath_to_key(file_path))
                else:
                    present = os.path.isfile(os.path.join(FLAT_DIR, file_path))
                assert present, (
                    f"File {f}.tsv must be present in the new_genes_data"
                    f" subdirectory {new_gene_subdir}."
                )
```
Add the import at top of file (after line 16):
```python
from v2ecoli.processes.parca.reconstruction.ecoli.sources import relpath_to_key
```

- [ ] **Step 4: Run the parity test**

Run: `.venv/bin/python -m pytest tests/test_kb_bundle_parity.py -q`
Expected: PASS. (This proves bundle-loaded `raw_data` is identical to the legacy local-flat `raw_data`.)

- [ ] **Step 5: Commit**

```bash
git add v2ecoli/processes/parca/reconstruction/ecoli/knowledge_base_raw.py tests/test_kb_bundle_parity.py
git commit -m "feat(parca): KnowledgeBaseEcoli loads via SourceBundle (parity-tested vs flat)"
```

---

## Task 6: CLI + config — `--bundle-manifest-path`

**Files:**
- Modify: `v2ecoli/cli/parca.py:48-77` (args), `:91-95` (KB construction)

- [ ] **Step 1: Add the CLI argument**

In `v2ecoli/cli/parca.py`, after the `--no-operons` argument (line 64), add:
```python
    parser.add_argument(
        "--bundle-manifest-path", type=str, default=None,
        help="Path to an ecoli-sources bundle manifest (default: the installed "
             "ecoli-sources reference bundle + v2ecoli overrides).")
```

- [ ] **Step 2: Build a bundle and pass it to KB**

Replace the `raw = KnowledgeBaseEcoli(...)` call (lines 91-95) with:
```python
    from v2ecoli.processes.parca.reconstruction.ecoli.sources import SourceBundle
    bundle = SourceBundle(base_manifest=args.bundle_manifest_path)
    raw = KnowledgeBaseEcoli(
        operons_on=not args.no_operons,
        remove_rrna_operons=False, remove_rrff=False, stable_rrna=False,
        bundle=bundle,
    )
```

- [ ] **Step 3: Smoke-run the CLI in fast mode**

Run:
```bash
.venv/bin/v2ecoli-parca --mode fast -o out/_plan_smoke --cpus 2 2>&1 | tail -20
```
Expected: ParCa initializes and loads raw_data without file-not-found errors (it may run several steps; interrupt once past raw_data load if slow).

- [ ] **Step 4: Commit**

```bash
git add v2ecoli/cli/parca.py
git commit -m "feat(parca): --bundle-manifest-path CLI flag threads a SourceBundle into KB"
```

---

## Task 7: Delete the 130 local flat files + update packaging

**Files:**
- Delete: 130 files under `v2ecoli/processes/parca/reconstruction/ecoli/flat/` (keep `sequence.fasta`? — see Step 1)
- Modify: `pyproject.toml:64-66` (`[tool.setuptools.package-data]`)
- Modify: `knowledge_base_raw.py` (make `bundle` non-optional in the default CLI path; keep legacy fallback only if a flat file still exists)

- [ ] **Step 1: Confirm sequence.fasta is in the bundle, then delete local flat files**

The bundle has key `sequence` (verified). Delete every local flat file whose key the bundle provides AND that is not an override:
```bash
cd /Users/eranagmon/code/v2ecoli
F=v2ecoli/processes/parca/reconstruction/ecoli/flat
.venv/bin/python - <<'PY'
import os
from pathlib import Path
from v2ecoli.processes.parca.reconstruction.ecoli.sources import SourceBundle, relpath_to_key
F = Path("v2ecoli/processes/parca/reconstruction/ecoli/flat")
b = SourceBundle()
OVERRIDE = {"equilibrium_reactions","equilibrium_reaction_rates","metabolic_reactions_added"}
deleted = 0
for f in sorted(F.rglob("*")):
    if not f.is_file():
        continue
    key = "sequence" if f.name == "sequence.fasta" else relpath_to_key(str(f.relative_to(F)))
    if key in OVERRIDE:
        continue
    if b.has_key(key):
        f.unlink(); deleted += 1
    else:
        print("KEPT (no bundle key):", f.relative_to(F))
print("deleted", deleted)
PY
git status --short v2ecoli/processes/parca/reconstruction/ecoli/flat | head
```
Expected: `deleted 130` (or 131 incl. sequence.fasta), with **no** "KEPT (no bundle key)" lines. Investigate any KEPT line before continuing.

- [ ] **Step 2: Remove now-empty flat subdirectories**

Run:
```bash
find v2ecoli/processes/parca/reconstruction/ecoli/flat -type d -empty -delete
ls v2ecoli/processes/parca/reconstruction/ecoli/flat 2>/dev/null || echo "flat/ gone"
```

- [ ] **Step 3: Update package-data**

In `pyproject.toml`, replace line 66:
```toml
"v2ecoli.processes.parca.reconstruction.ecoli" = ["flat/**/*"]
```
with:
```toml
"v2ecoli.processes.parca.reconstruction.ecoli" = ["flat_overrides/*", "parca_overrides.tsv"]
```

- [ ] **Step 4: Make the bundle the default in KB and drop the dead legacy branch**

In `knowledge_base_raw.py`, default the bundle when none is passed, so direct `KnowledgeBaseEcoli(...)` callers (tests, scripts) still work after deletion:
```python
        if bundle is None:
            from v2ecoli.processes.parca.reconstruction.ecoli.sources import SourceBundle
            bundle = SourceBundle()
        self._bundle = bundle
```
`_resolve` can now drop the FLAT_DIR branch:
```python
    def _resolve(self, rel_path):
        return self._bundle.resolve_relpath(rel_path)
```
Leave `FLAT_DIR` defined (the parity test in Task 5 references the legacy path only while files exist; that test is now skipped automatically once `flat/` is gone — confirm in Step 6).

- [ ] **Step 5: Update the Task 5 parity test to skip cleanly when flat/ is gone**

In `tests/test_kb_bundle_parity.py`, guard `test_bundle_kb_matches_legacy_kb` so it skips after deletion (the legacy read would now fail):
```python
import os
FLAT = "v2ecoli/processes/parca/reconstruction/ecoli/flat"

@pytest.mark.skipif(not os.path.isdir(FLAT), reason="local flat/ deleted post-migration")
def test_bundle_kb_matches_legacy_kb():
    ...
```

- [ ] **Step 6: Run the full bundle/KB test set**

Run: `.venv/bin/python -m pytest tests/test_source_bundle.py tests/test_kb_bundle_parity.py -q`
Expected: PASS, with the two legacy-comparison tests reporting `skipped` (flat/ gone). The default-bundle KB build still works:
```bash
.venv/bin/python -c "from v2ecoli.processes.parca.reconstruction.ecoli.knowledge_base_raw import KnowledgeBaseEcoli as K; kb=K(operons_on=True, remove_rrna_operons=False, remove_rrff=False, stable_rrna=False); print('genes', len(kb.genes))"
```
Expected: non-zero gene count.

- [ ] **Step 7: Commit**

```bash
git add -A v2ecoli/processes/parca/reconstruction/ecoli pyproject.toml tests/test_kb_bundle_parity.py
git commit -m "feat(parca): drop 130 local flat files; ParCa sources from ecoli-sources bundle"
```

---

## Task 8: End-to-end byte-identity gate + docs

**Files:**
- Modify: `docs/converting_vivarium_processes.md`? No — Create `doc/data_ingestion` note; update `AGENTS.md` ParCa section (separately, since AGENTS.md has an unrelated uncommitted edit — coordinate with the user).
- Test: rely on existing cache-build + parity harness.

- [ ] **Step 1: Rebuild the ParCa cache and run the existing parity/smoke suite**

Run:
```bash
.venv/bin/python -m pytest tests/ -q -k "cache or parca or arch_parity or behavior" 2>&1 | tail -25
```
Expected: same pass set as a pre-migration baseline run. Investigate any new failure — a genuine raw_data difference would surface here or in Task 5's parity test.

- [ ] **Step 2: Full-mode cache build sanity (optional, slow)**

Run (only if validating full ParCa): `.venv/bin/v2ecoli-parca --mode full -o out/_full_check --cpus 4` and confirm it completes and produces `parca_state.pkl`. Compare downstream sim_data fingerprint to the frozen `models/parca/parca_state.pkl.gz` baseline if a comparison helper exists.

- [ ] **Step 3: Write a short consumer doc**

Create `docs/parca_data_bundle.md` describing: where ParCa data now comes from (ecoli-sources), the override mechanism (`parca_overrides.tsv` + `flat_overrides/`), and `--bundle-manifest-path` usage. (Model it on this plan's spec.)

- [ ] **Step 4: Commit**

```bash
git add docs/parca_data_bundle.md
git commit -m "docs(parca): document ecoli-sources bundle + override mechanism"
```

- [ ] **Step 5: Open the PR**

```bash
git push -u origin feat/ecoli-sources-bundle
gh pr create --base main --title "feat(parca): source ParCa inputs from ecoli-sources bundle" \
  --body "Ports CovertLab/vEcoli #426's bundle integration to v2ecoli. ParCa now reads its 133-file flat input surface through a SourceBundle resolver pinned to ecoli-sources; v2ecoli's 3 diverged biology files (equilibrium ×2 from #123, metabolic_reactions_added) stay local via a 3-row override manifest. raw_data parity proven by tests/test_kb_bundle_parity.py before the 130 identical local files were deleted. Spec: docs/superpowers/specs/2026-06-06-ecoli-sources-bundle-integration-design.md. Stacked PR 2 (multi-ParCa runner) to follow."
```

---

## Self-review notes

- **Spec coverage:** dependency (T1), resolver + override merge (T2), override files (T3), KB rewire (T4–T5), CLI/config (T6), deletion + package-data (T7), byte-identity gate + docs (T8). All spec sections covered. The `--bundle-manifest-path` × override interaction (spec) is realized: T6 passes only `base_manifest`; overrides always layer via `SourceBundle`'s default.
- **Risk note (carry into execution):** Task 5's parity test is the load-bearing gate. If it shows diffs, do NOT proceed to deletion (Task 7) — the variant/flag→key mapping or a content drift is wrong. The `operons_on=True` default matches the CLI; if other flag combinations (`remove_rrff`, `new_genes_option="gfp"`) are exercised by ParCa configs in this repo, add a parametrized parity case per combination before Task 7.
- **AGENTS.md:** intentionally not edited here (it carries an unrelated uncommitted change in the working tree); coordinate that edit with the user.

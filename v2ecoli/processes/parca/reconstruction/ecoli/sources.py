"""
Resolver for the ecoli-sources data bundle.

A *bundle* maps each ``canonical_key`` (an addressable data role in ParCa) to a
source file. The default reference bundle ships with ``ecoli-sources``
(``ecoli_sources.BUNDLE_PATH``). v2ecoli layers a small override manifest on
top so its locally-diverged flat files (equilibrium / metabolism biology) win
over the upstream defaults without copying the whole 135-key manifest.

Overrides form a **chain**: v2ecoli's own defaults first, then any manifests the
caller supplies, each applied over the last. That is what lets a private payload
(e.g. a strain's new-gene flat inputs, keyed ``new_gene_data__<strain>__*``)
ADD keys to the public baseline without displacing either the baseline or
v2ecoli's own divergences.

Ported and adapted from CovertLab/vEcoli's ``wholecell/io/sources.py``
(PR #426); the override-merge is a v2ecoli addition.
"""
from __future__ import annotations

import json
import os
import warnings
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
    # Strip a single trailing extension from the BASENAME only, so a dotted
    # directory component (e.g. ``dir.v2/no_ext``) isn't mangled.
    slash = norm.rfind("/")
    head, tail = norm[: slash + 1], norm[slash + 1 :]
    dot = tail.rfind(".")
    if dot > 0:  # >0 keeps no-extension names and leading-dot files intact
        tail = tail[:dot]
    return (head + tail).replace("/", "__")


class SourceBundle:
    """Resolve canonical keys / flat-relpaths to absolute source paths."""

    def __init__(
        self,
        base_manifest: Optional[PathLike] = None,
        overrides: Optional[Union[PathLike, list]] = None,
        validate: bool = True,
    ):
        """``overrides`` is one manifest path or a list of them, applied in
        order ON TOP of v2ecoli's own defaults — never instead of them."""
        if base_manifest is None:
            from ecoli_sources import BUNDLE_PATH
            base_manifest = BUNDLE_PATH
        base_manifest = Path(base_manifest).resolve()
        if not base_manifest.is_file():
            raise FileNotFoundError(f"Bundle manifest not found: {base_manifest}")

        index: dict[str, Path] = {}
        base_root = base_manifest.parent
        index.update(self._read_manifest(base_manifest, base_root))

        # A variant bundle (ecoli-sources ``compose_variant_bundle``) ships a
        # ``genotype.json`` sidecar next to its manifest whose ``overridden_keys``
        # lists the keys the variant GENERATED (e.g. a knockout's recomputed
        # ``dna_sites``). The v2ecoli override exists to win over the shared
        # UPSTREAM base, NOT over a variant's own divergences: a whole-file
        # override that re-states thousands of otherwise-unchanged rows would
        # silently clobber a variant's corrected key back to pre-perturbation
        # coordinates, which validates and hashes stably and only fails at ParCa
        # build time (#466). So a key the variant generated is protected from the
        # override; every other key is overridden exactly as before.
        variant_keys = self._variant_generated_keys(base_manifest)
        #: Public, because it is not only an override-precedence detail: a
        #: consumer needs to know whether a variant GENERATED a key in order to
        #: notice that it generated one the build is about to ignore (a
        #: ``knockdown()`` bundle fitted with ``rnaseq_source: reference``).
        #: Empty set for a plain base bundle.
        self.variant_generated_keys = variant_keys
        #: Keys a CALLER contributed through the override chain, populated below.
        #:
        #: Deliberately EXCLUDES v2ecoli's own ``_DEFAULT_OVERRIDES`` link, which
        #: is applied to every build: those four keys are this repo's standing
        #: divergence from upstream ecoli-sources, not something a caller
        #: supplied for this run. Conflating them would make a consumer's
        #: "did someone deliberately supply this?" check answer yes on a stock
        #: build the moment v2ecoli moved a watched key into its own overrides —
        #: a guard that fires on every build teaches people to ignore it.
        #:
        #: The sidecar covers only what a *generator* wrote next to the base
        #: manifest. An override manifest carries no ``genotype.json``, so a
        #: payload that supplies a key through ``--bundle-overrides`` is
        #: invisible to ``variant_generated_keys`` — and that is now a real
        #: entry point, not a hypothetical one. Both sets together are what
        #: "this key did not come from the stock reference bundle" means; see
        #: :attr:`externally_supplied_keys`.
        self.override_supplied_keys: set[str] = set()

        # Overrides are a CHAIN, applied in order, and v2ecoli's own defaults
        # are always the first link.
        #
        # This used to be a single file where a caller-supplied ``overrides``
        # REPLACED ``_DEFAULT_OVERRIDES``. That is a silent-wrong-answer bug the
        # moment anyone passes one: v2ecoli's four locally-diverged flat files
        # (dna_sites, equilibrium_reactions, equilibrium_reaction_rates,
        # metabolic_reactions_added) would revert to their upstream ecoli-sources
        # versions with no warning, validation still passing, and the ParCa
        # quietly fitting different biology. Nothing passed ``overrides`` yet, so
        # the defect was latent — which is exactly when it is free to fix.
        #
        # A private overlay (e.g. a strain's new-gene flat inputs) is therefore
        # ADDITIVE to the defaults rather than a replacement for them. Later
        # links win over earlier ones on a key collision.
        chain: list[Path] = []
        if _DEFAULT_OVERRIDES.is_file():
            chain.append(_DEFAULT_OVERRIDES)
        if overrides is not None:
            extra = ([overrides] if isinstance(overrides, (str, os.PathLike))
                     else list(overrides))
            chain.extend(Path(p).resolve() for p in extra)

        for override in chain:
            caller_supplied = override != _DEFAULT_OVERRIDES
            for key, path in self._read_manifest(override, override.parent).items():
                if key in variant_keys:
                    # The variant explicitly generated this key; its file wins.
                    # Surface the suppressed collision rather than hiding it — the
                    # per-key declared-vs-injected check ecoli-sources#12 asked for.
                    warnings.warn(
                        f"parca override for {key!r} suppressed: variant bundle "
                        f"{base_manifest.name} generated this key, so its file "
                        "takes precedence over the whole-file override.",
                        stacklevel=2,
                    )
                    continue
                index[key] = path
                if caller_supplied:
                    self.override_supplied_keys.add(key)

        self._index = index
        # Kept as provenance: the genotype a ParCa build was made from IS its
        # bundle manifest, so downstream steps need to be able to name it
        # rather than only read through it.
        self.base_manifest = base_manifest
        # The full chain, in application order. Provenance has to name every
        # manifest that contributed, not just the last one -- a build resolved
        # through two overrides is a different genotype from one resolved
        # through either alone.
        # ⚠ Nothing in this repo reads it yet: it is recorded for consumers that
        # need the whole chain rather than its last link (a genotype cross-check
        # over the overrides, or a variant-generated-key guard that must see
        # keys contributed by an override rather than only by the base
        # manifest's sidecar). Forward-looking, deliberately, not dead.
        self.override_chain = list(chain)
        # Back-compat: the single-file attribute callers previously read, now
        # the LAST link (the one that wins). ``override_chain`` is the record.
        self.overrides = chain[-1] if chain else None
        if validate:
            self._validate(base_manifest, self.overrides)

    @property
    def externally_supplied_keys(self) -> set[str]:
        """Keys this bundle got from somewhere other than the stock base table.

        The union of what a variant GENERATED (sidecar) and what the OVERRIDE
        CHAIN contributed. A consumer asking "did someone deliberately supply
        this key?" must ask both: the sidecar alone misses every payload
        delivered through ``--bundle-overrides``, which is exactly the entry
        point a guard against silently-ignored inputs needs to cover.

        Note the two halves mean subtly different things — generated-by-a-
        variant vs. supplied-by-an-overlay — and a caller that needs to tell
        them apart should read the two attributes directly.
        """
        return set(self.variant_generated_keys) | set(self.override_supplied_keys)

    @staticmethod
    def _variant_generated_keys(manifest: Path) -> set[str]:
        """Keys a variant bundle GENERATED, read from its ``genotype.json``
        sidecar's ``overridden_keys``.

        Empty for a plain base bundle — it has no sidecar, or was not composed
        from generators — so the override still wins over base exactly as before.
        Provenance rides in the sidecar and not the manifest by design: the
        bundle schema is ``strict="filter"`` and drops unknown manifest columns,
        so this is the only in-band signal of what a variant actually wrote.
        """
        sidecar = manifest.parent / "genotype.json"
        if not sidecar.is_file():
            return set()
        try:
            data = json.loads(sidecar.read_text())
        except (ValueError, OSError):
            return set()
        keys = data.get("overridden_keys", [])
        return set(keys) if isinstance(keys, list) else set()

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
            rows = [
                {"canonical_key": k, "source_path": str(p), "description": "", "schema_name": ""}
                for k, p in self._index.items()
            ]
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


def enumerate_for_dashboard() -> list[dict]:
    """Enumerate the ecoli-sources bundle for the dashboard data-sources panel.

    Returns one dict per bundle file::

        {
          "key":        canonical_key (e.g. "condition__media__MIX0-55"),
          "path":       absolute resolved source path (str),
          "category":   first key segment (e.g. "condition", "metabolism"),
          "kind":       "override" if under flat_overrides/, else "inherited",
          "size_bytes": file size in bytes (0 if missing),
        }

    Classification mirrors ``reports/ecoli_sources_report.py:section_inputs``:
    a key is an *override* when its resolved path is under ``flat_overrides/``
    (v2ecoli's locally-diverged flat files), otherwise it is *inherited* from
    the upstream ecoli-sources bundle.
    """
    bundle = SourceBundle()
    out: list[dict] = []
    for key, path in bundle._index.items():
        path = Path(path)
        is_override = "flat_overrides" in str(path)
        parts = key.split("__")
        category = parts[0] if len(parts) > 1 else "root"
        try:
            size_bytes = path.stat().st_size if path.is_file() else 0
        except OSError:
            size_bytes = 0
        out.append(
            {
                "key": key,
                "path": str(path),
                "category": category,
                "kind": "override" if is_override else "inherited",
                "size_bytes": size_bytes,
            }
        )
    out.sort(key=lambda d: (d["category"], d["key"]))
    return out

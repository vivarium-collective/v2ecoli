"""``ParcaBundleStep`` — wraps a pre-built ParCa cache bundle as a
content-addressed ``sim_data`` :class:`~process_bigraph.artifacts.ArtifactRef`.

This Step does **not** run ParCa. ``bundle_dir`` must already contain a valid
sim-data bundle written ahead of time by ``v2ecoli.core.save_sim_input`` /
``save_cache`` (``sim_data_cache.dill``, ``initial_state.json``, ...). Its
whole job is orchestration plumbing: validate the bundle is genuine, content-
address it, and hand a lightweight :class:`ArtifactRef` reference downstream
(e.g. as a Nextflow-style channel value) — without eagerly loading the
(often hundreds-of-MB) payload itself.
"""

from __future__ import annotations

import hashlib
import os

from process_bigraph import Step
from process_bigraph.artifacts import ArtifactRef, SIM_DATA, write_fingerprint

from v2ecoli.core import _cache_verify_skipped
from v2ecoli.library.cache_version import verify_cache_version


def _sha256_file(path: str, chunk_size: int = 1024 * 1024) -> bytes:
    """Chunked sha256 digest of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b''):
            digest.update(chunk)
    return digest.digest()


def _bundle_hash(bundle_dir: str) -> tuple[str, dict[str, str]]:
    """Content-address a bundle directory.

    F6: the combined hash is ``sha256`` over the **concatenation of SORTED
    per-file digests** — deliberately not XOR (two identical/complementary
    file digests would self-cancel under XOR) and sorted on the digest
    *bytes* themselves (not filenames), so directory-listing order can never
    change the result.

    Returns ``(hash_hex, {filename: digest_hex})`` — the per-file digests are
    kept so a caller can pinpoint which file changed on a mismatch.
    """
    filenames = sorted(
        name for name in os.listdir(bundle_dir)
        if os.path.isfile(os.path.join(bundle_dir, name)))
    if not filenames:
        raise FileNotFoundError(
            f"ParcaBundleStep: bundle_dir {bundle_dir!r} has no files to "
            f"content-address")

    context: dict[str, str] = {}
    digests: list[bytes] = []
    for name in filenames:
        digest = _sha256_file(os.path.join(bundle_dir, name))
        context[name] = digest.hex()
        digests.append(digest)

    combined = hashlib.sha256(b''.join(sorted(digests))).hexdigest()
    return combined, context


class ParcaBundleStep(Step):
    """Emit a ``sim_data`` :class:`ArtifactRef` for an existing ParCa bundle.

    Fixture/pre-cached use only: ``bundle_dir`` must point at a directory a
    prior ParCa run (or a fixture-derived ``save_sim_input`` call) has
    already populated. ``update()`` reuses
    ``v2ecoli.library.cache_version.verify_cache_version`` to check the
    bundle is genuine (subject to the usual ``V2ECOLI_SKIP_CACHE_VERIFY``
    escape hatch for a worktree whose content fingerprint has legitimately
    moved) WITHOUT loading the (often hundreds-of-MB) dill payload itself,
    then content-addresses its files. ``ref.store`` is the bundle
    **directory** — downstream consumers (e.g. the T8/T9 sim) inject it
    directly as ``load_cache_bundle``'s ``cache_dir`` argument, which joins
    filenames onto it and requires a directory, not a file.
    """

    config_schema = {
        'mode': 'string',
        'cpus': 'integer',
        'condition': 'maybe[string]',
        'bundle_dir': 'string',
    }

    def outputs(self):
        return {'sim_data': {'_type': 'string', '_is_file': True}}

    def update(self, state):
        bundle_dir = self.config['bundle_dir']

        # Validate bundle_dir is a genuine ParCa bundle before handing out a
        # reference to it — the same staleness check every other cache_dir
        # consumer (e.g. ecoli_baseline, load_cache_bundle) relies on, but
        # WITHOUT the ~157MB dill load + deep-copy load_cache_bundle would
        # otherwise force: this Step is orchestration plumbing, not a sim
        # consumer, and must stay cheap.
        if not _cache_verify_skipped():
            verify_cache_version(bundle_dir)

        bundle_hash, context = _bundle_hash(bundle_dir)

        # Record the fingerprint in the canonical content-addressed store
        # (`.pbg/artifacts/<hash>/fingerprint`) so a later cache-hit lookup
        # against this same content address can check it without re-hashing
        # the whole bundle.
        write_fingerprint(bundle_hash, bundle_hash)

        ref = ArtifactRef(
            kind=SIM_DATA,
            hash=bundle_hash,
            store=bundle_dir,
            context=context,
        )
        return {'sim_data': ref.to_dict()}

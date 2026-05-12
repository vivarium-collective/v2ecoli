"""Generic XArrayEmitter for v2ecoli.

Vendored from vivarium-collective/vEcoli@b25ca24 (PR #414 head). Re-rooted
onto process_bigraph.emitter.Emitter. vEcoli-specific metadata keys and
validator checks are now config-driven; see v2ecoli.library.emitter_presets
for a builder that reproduces vEcoli's exact behavior.
"""

try:
    import xarray  # noqa: F401
    import zarr    # noqa: F401
    import zarrs   # noqa: F401
except ImportError as e:
    raise ImportError(
        f"v2ecoli.library.xarray_emitter requires the [xarray] extra. "
        f"Install with: pip install 'v2ecoli[xarray]'. (missing: {e.name})"
    ) from e

# Re-export only after the outer class is wired in Task 6.
# from v2ecoli.library.xarray_emitter.emitter import XArrayEmitter

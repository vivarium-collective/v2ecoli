"""Re-export shim: XArrayEmitter from pbg-emitters.

The emitter implementation was extracted to a focused library
(https://github.com/vivarium-collective/pbg-emitters) so it can ship
heavy optional deps (xarray / zarr / zarrs) per-emitter and iterate
independently. v2ecoli imports from here go through this shim so
existing call sites (and external users of
``v2ecoli.library.xarray_emitter``) keep working.

v2ecoli-specific layers stay in this package:
  * ``v2ecoli.library.emitter_presets`` — the xarray emitter preset
    builder that reproduces vEcoli's exact metadata keys / validators.
  * ``v2ecoli.library.xarray_run`` — lineage walk over xarray output.
  * the vEcoli-parity test + golden fixture in ``tests/`` exercise the
    shimmed class against the vendored baseline.
"""

# Re-export the public API from pbg-emitters. If the [xarray] extra
# (and therefore xarray / zarr / zarrs) isn't installed,
# ``from viva_emitters.xarray_emitter import XArrayEmitter`` raises
# ImportError; we wrap that in a friendlier message that points at the
# v2ecoli extra (which now pulls pbg-emitters[xarray]).
try:
    from viva_emitters.xarray_emitter import XArrayEmitter  # noqa: F401
except ImportError as e:
    raise ImportError(
        f"v2ecoli.library.xarray_emitter requires the [xarray] extra. "
        f"Install with: pip install 'v2ecoli[xarray]'. (missing: {e.name})"
    ) from e

__all__ = ["XArrayEmitter"]

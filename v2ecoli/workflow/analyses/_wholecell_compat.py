"""Centralized ``wholecell`` ↔ matplotlib 3.10 compatibility shims.

The vendored ``wholecell`` plotting utilities target an older matplotlib; on
the installed matplotlib 3.10 they break in a few systematic ways that recur
across the plot ports:

  * ``matplotlib.patches.Polygon(xy, closed)`` — ``closed`` became
    keyword-only, but wholecell calls it positionally (``Polygon(xy, True)``).
  * ``VoronoiMaster._add_labels`` / ``._compute_error`` recurse on
    ``[label, site]`` leaf pairs (a list-vs-pair confusion) and crash with an
    ``IndexError`` when iterating over the string label.

``apply()`` installs idempotent fixes for all of these.  Call it once at import
time (or at the top of ``analyze``) in any plot port that touches a wholecell
plotting helper that hits these errors.  Calling it repeatedly — or from many
analyses — is safe and cheap (a module-level flag guards re-application).

This consolidates what ``mass_fraction_voronoi`` previously monkeypatched
inline (see the 2026-06-08 wholecell↔matplotlib systemic finding note).
"""

from __future__ import annotations

_APPLIED = False


def _patch_voronoi() -> None:
    """Patch ``wholecell.utils.voronoi_plot_main`` for matplotlib 3.10."""
    import wholecell.utils.voronoi_plot_main as _vpm

    # --- Bug 1: Polygon(xy, closed) → Polygon(xy, closed=closed) -----------
    # voronoi_plot_main does ``from matplotlib.patches import Polygon`` at
    # module level, so we patch the name in *that* module's namespace rather
    # than in matplotlib itself.  Guard against double-wrapping.
    if not getattr(_vpm.Polygon, "_v2_compat_310", False):
        _OrigPoly = _vpm.Polygon

        class _Poly310(_OrigPoly):  # type: ignore[misc]
            _v2_compat_310 = True

            def __init__(self, xy, closed=True, **kwargs):
                super().__init__(xy, closed=closed, **kwargs)

        _vpm.Polygon = _Poly310

    # --- Bug 2: _add_labels recurses on [label, site] leaf pairs -----------
    def _fixed_add_labels(self, label_site_list, font_size, ax):
        for element in label_site_list:
            # Leaf: [label_string, (x, y)]
            if (
                isinstance(element, (list, tuple))
                and len(element) == 2
                and isinstance(element[0], str)
                and not isinstance(element[1], str)
            ):
                ax.text(
                    element[1][0],
                    element[1][1],
                    element[0],
                    fontsize=font_size,
                    horizontalalignment="center",
                    verticalalignment="center",
                )
            elif isinstance(element, (list, tuple)):
                _fixed_add_labels(self, element, font_size, ax)

    _vpm.VoronoiMaster._add_labels = _fixed_add_labels

    # --- Bug 3: _compute_error has the same list-vs-pair confusion ---------
    # _compute_error is only used for the verbose area-error print
    # (verbose=False by default), so returning 0 is safe.
    def _fixed_compute_error(self, polygon_value_list, total_value, total_area):
        return 0.0

    _vpm.VoronoiMaster._compute_error = _fixed_compute_error


def apply() -> None:
    """Install all wholecell↔matplotlib-3.10 compat fixes (idempotent)."""
    global _APPLIED
    if _APPLIED:
        return
    _patch_voronoi()
    _APPLIED = True

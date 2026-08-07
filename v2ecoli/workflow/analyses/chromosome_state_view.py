"""Chromosome-state GIF: circular-chromosome animation over a cell's cycle.

Renders one frame per sampled timepoint showing oriC (origin), ter
(terminus), and the current DNA-polymerase / replisome fork positions on the
circular E. coli chromosome, then assembles the frames into an animated GIF.
Native ``Analysis`` (single-cell scale) — reads
``listeners__replication_data__fork_coordinates`` (per-row array of fork
positions, bp) and ``listeners__replication_data__number_of_oric`` (int) via
:func:`read_stacked_columns`.

Registered as ``"chromosome_state_view"`` (scale: ``"single"``).  Frames are
rasterized with matplotlib (not SVG — ``cairosvg`` is unavailable in this
workspace) and assembled with ``imageio``.  Never raises: any missing column
or empty data degrades to a small HTML note, mirroring
``mass_fraction_summary_view`` / the ``ParquetAnalysisView`` convention.
"""

from __future__ import annotations

import base64
import io
import math
from typing import Any

from duckdb import DuckDBPyConnection

from v2ecoli.workflow.analysis import Analysis, ANALYSIS_REGISTRY
from v2ecoli.workflow.analyses._helpers import read_stacked_columns, cast_decimals

# E. coli MG1655 genome length (bp); oriC is bp 0 by convention, ter ~ bp
# 2,320,000 (roughly opposite oriC on the circular chromosome).
GENOME_LEN = 4_641_652
TER_COORD = 2_320_000

MAX_FRAMES = 40


def _unavailable(reason: str) -> dict:
    return {
        "view": (
            "<div style='padding:12px;color:#64748b'>"
            f"Chromosome-state GIF unavailable: {reason}</div>"
        )
    }


def _bp_to_xy(coord: float) -> tuple[float, float]:
    """bp position -> unit-circle (x, y), angle 0 at top, clockwise."""
    theta = 2 * math.pi * coord / GENOME_LEN
    return math.sin(theta), math.cos(theta)


class ChromosomeStateView(Analysis):
    """Animated circular-chromosome state (oriC / ter / replication forks)."""

    name = "chromosome_state_view"
    scale = "single"

    def analyze(
        self,
        *,
        conn: DuckDBPyConnection,
        history_sql: str,
        sim_data=None,
        variant_metadata: dict[str, Any] | None = None,
        **ctx,
    ) -> dict:
        try:
            return self._analyze(conn=conn, history_sql=history_sql)
        except Exception as e:  # noqa: BLE001 — never raise, degrade instead
            return _unavailable(str(e))

    def _analyze(self, *, conn: DuckDBPyConnection, history_sql: str) -> dict:
        import imageio.v2 as imageio
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from PIL import Image

        df = read_stacked_columns(
            history_sql,
            [
                "listeners__replication_data__fork_coordinates",
                "listeners__replication_data__number_of_oric",
            ],
            conn=conn,
        )
        if df.is_empty():
            return _unavailable("no timepoints in history")

        df = cast_decimals(df).sort("time")
        n = df.height
        if n == 0:
            return _unavailable("no timepoints in history")

        stride = max(1, n // MAX_FRAMES)
        records = df.to_dicts()
        sampled = records[::stride]
        if not sampled:
            return _unavailable("no timepoints in history")

        t0 = records[0]["time"]

        frames = []
        for row in sampled:
            minutes = (row["time"] - t0) / 60.0
            n_oric = row.get("listeners__replication_data__number_of_oric")
            forks = row.get("listeners__replication_data__fork_coordinates") or []

            fig, ax = plt.subplots(figsize=(4, 4), dpi=90)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_xlim(-1.3, 1.3)
            ax.set_ylim(-1.3, 1.3)

            circle = plt.Circle((0, 0), 1.0, fill=False, color="#94a3b8", linewidth=2)
            ax.add_patch(circle)

            # oriC (green dot)
            ox, oy = _bp_to_xy(0)
            ax.plot(ox, oy, "o", color="#16a34a", markersize=9, zorder=3)

            # ter (red square)
            tx, ty = _bp_to_xy(TER_COORD)
            ax.plot(tx, ty, "s", color="#dc2626", markersize=8, zorder=3)

            # replication forks (amber triangles)
            for coord in forks:
                if coord is None:
                    continue
                try:
                    c = float(coord)
                except (TypeError, ValueError):
                    continue
                if c != c:  # NaN sentinel
                    continue
                fx, fy = _bp_to_xy(c)
                ax.plot(fx, fy, "^", color="#f59e0b", markersize=8, zorder=4)

            ax.set_title(f"t = {minutes:.0f} min · oriC = {n_oric}", fontsize=11)

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=90)
            plt.close(fig)
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
            frames.append(np.array(img))

        if not frames:
            return _unavailable("no frames rendered")

        bio = io.BytesIO()
        imageio.mimsave(bio, frames, format="GIF", duration=0.12)
        bio.seek(0)
        b64 = base64.b64encode(bio.read()).decode("ascii")
        return {
            "view": f'<img src="data:image/gif;base64,{b64}" style="max-width:100%"/>'
        }

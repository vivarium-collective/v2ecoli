"""Visualizations for colony / microfluidic-device runs.

Two products, both consumed by the workbench study Visualize tab when written
into a study's ``charts/`` directory:

1. ``render_device_gif`` — an animated GIF of the running device (cells as
   capsules, lineage-coloured) via viva-munk's ``simulation_to_gif``.
2. ``render_phenotype_figures`` — static distribution figures from a
   ``phenotype_extractor`` result: size-at-division, time between birth and
   division (inter-division time), and added size before division (the adder
   plot).

All figures are tier-agnostic — they read the extractor's output, not the
model. Matplotlib runs headless (Agg).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_BLUE = "#3b7dd8"
_ORANGE = "#e4572e"
_GREEN = "#2a9d5c"


def trajectory_to_gif_history(trajectory: list[dict]) -> list[dict]:
    """Convert sampled trajectory frames into ``simulation_to_gif`` input.

    Each frame's cells must carry ``location`` (and optionally angle/length/
    radius). Cells without a location are skipped. Returns a list of
    ``{'agents': {id: {type, location, length, radius, angle, mass}}, 'time'}``.
    """
    history = []
    for frame in trajectory:
        agents: dict[str, dict] = {}
        for cid, cell in frame.get("cells", {}).items():
            loc = cell.get("location")
            if loc is None:
                continue
            agents[cid] = {
                "type": "segment",
                "location": tuple(loc),
                "length": float(cell.get("length") or 2.0),
                "radius": float(cell.get("radius") or 0.5),
                "angle": float(cell.get("angle") or 0.0),
                "mass": float(cell.get("mass") or 0.0),
            }
        history.append({"agents": agents, "time": float(frame.get("time", 0.0))})
    return history


def render_device_gif(
    trajectory: list[dict],
    out_path: str | Path,
    *,
    env_size: float,
    env_width: float | None = None,
    env_height: float | None = None,
    barriers: list | None = None,
    flow_regions: list | None = None,
    skip_frames: int = 1,
    frame_duration_ms: int = 120,
    color_by_phylogeny: bool = True,
) -> Path:
    """Render the running device to an animated GIF at ``out_path``.

    Cells are drawn as capsules; ``color_by_phylogeny`` hue-shifts daughters
    from their mother (relies on the ``{mother}_{i}`` id convention). ``barriers``
    (mother-machine channel walls) and ``flow_regions`` (the wash-out boundary)
    are drawn when supplied.
    """
    from viva_munk.plots.multibody_plots import simulation_to_gif

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    history = trajectory_to_gif_history(trajectory)
    config = {"env_size": float(env_size), "barriers": barriers or []}
    simulation_to_gif(
        history,
        config,
        agents_key="agents",
        filename=out_path.name,
        out_dir=str(out_path.parent),
        skip_frames=skip_frames,
        frame_duration_ms=frame_duration_ms,
        color_by_phylogeny=color_by_phylogeny,
        env_width=env_width,
        env_height=env_height,
        flow_regions=flow_regions,
        show_time_title=True,
    )
    return out_path


def _clean(values: list) -> list[float]:
    return [float(v) for v in values if v is not None and np.isfinite(v)]


def _hist(ax, data: list[float], *, xlabel: str, title: str, color: str, unit: str = "") -> None:
    data = _clean(data)
    if not data:
        ax.text(0.5, 0.5, "no division events recorded yet", ha="center",
                va="center", transform=ax.transAxes, color="#888", fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(xlabel)
        return
    bins = max(5, min(25, int(np.sqrt(len(data)))))
    ax.hist(data, bins=bins, color=color, edgecolor="white", alpha=0.9)
    mean = float(np.mean(data))
    cv = float(np.std(data) / mean) if mean else 0.0
    ax.axvline(mean, color=_ORANGE, lw=2, ls="--",
               label=f"mean {mean:.2f}{unit}  ·  CV {cv:.2f}  ·  n={len(data)}")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.grid(alpha=0.15)


def _write_meta(png_path: Path, title: str, caption: str) -> None:
    png_path.with_suffix(".meta.json").write_text(
        json.dumps({"title": title, "caption": caption}, indent=2)
    )


def render_phenotype_figures(
    phenotypes: dict[str, Any],
    out_dir: str | Path,
    *,
    label: str,
) -> list[Path]:
    """Write the three requested distribution figures + meta sidecars.

    Returns the list of PNG paths. Figures: size-at-division (length, µm),
    inter-division time (birth→division, minutes), and added size before
    division (adder plot: Δlength vs birth length + Δ distribution).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # 1) Size at division — length at the moment of division (µm)
    lengths = _clean(phenotypes.get("size_at_division", {}).get("length", []))
    p1 = out_dir / "size_at_division.png"
    fig, ax = plt.subplots(figsize=(5.2, 3.6), dpi=110)
    _hist(ax, lengths, xlabel="length at division (µm)",
          title=f"{label} — size at division", color=_BLUE, unit=" µm")
    fig.tight_layout()
    fig.savefig(p1)
    plt.close(fig)
    _write_meta(p1, f"{label}: size at division",
                "Distribution of cell length (µm) at the moment of division.")
    written.append(p1)

    # 2) Time between birth and division — inter-division time (→ minutes)
    interdiv_min = [t / 60.0 for t in _clean(phenotypes.get("interdivision_time", []))]
    p2 = out_dir / "interdivision_time.png"
    fig, ax = plt.subplots(figsize=(5.2, 3.6), dpi=110)
    _hist(ax, interdiv_min, xlabel="time birth → division (min)",
          title=f"{label} — inter-division time", color=_GREEN, unit=" min")
    fig.tight_layout()
    fig.savefig(p2)
    plt.close(fig)
    _write_meta(p2, f"{label}: time between birth and division",
                "Distribution of the interval from a cell's birth to its own division (minutes).")
    written.append(p2)

    # 3) Added size before division — the adder plot (Δlength vs birth length)
    added = phenotypes.get("added_length", []) or []
    births = _clean([a.get("birth_length") for a in added])
    deltas = _clean([a.get("delta_length") for a in added])
    p3 = out_dir / "added_size.png"
    fig, (axs, axh) = plt.subplots(1, 2, figsize=(8.6, 3.6), dpi=110,
                                   gridspec_kw={"width_ratios": [1.3, 1.0]})
    if births and deltas and len(births) == len(deltas):
        axs.scatter(births, deltas, s=22, color=_BLUE, alpha=0.7, edgecolor="white", linewidth=0.4)
        mean_d = float(np.mean(deltas))
        axs.axhline(mean_d, color=_ORANGE, lw=2, ls="--", label=f"mean Δ {mean_d:.2f} µm")
        # adder signature: slope ~0 of Δ vs birth length
        if len(births) >= 3 and np.std(births) > 1e-9:
            slope, intercept = np.polyfit(births, deltas, 1)
            xs = np.linspace(min(births), max(births), 50)
            axs.plot(xs, slope * xs + intercept, color="#555", lw=1.2,
                     label=f"fit slope {slope:.2f}")
        axs.legend(fontsize=8)
        axs.set_xlabel("length at birth (µm)")
        axs.set_ylabel("added length Δ (µm)")
        axs.set_title(f"{label} — adder: Δ vs birth", fontsize=11)
        axs.grid(alpha=0.15)
        _hist(axh, deltas, xlabel="added length Δ (µm)",
              title="added-size distribution", color=_BLUE, unit=" µm")
    else:
        for a in (axs, axh):
            a.text(0.5, 0.5, "no division events recorded yet", ha="center",
                   va="center", transform=a.transAxes, color="#888", fontsize=11)
        axs.set_title(f"{label} — adder: Δ vs birth", fontsize=11)
    fig.tight_layout()
    fig.savefig(p3)
    plt.close(fig)
    _write_meta(p3, f"{label}: added size before division",
                "Adder plot — added length Δ vs length at birth (a flat trend is the adder "
                "signature), with the Δ distribution.")
    written.append(p3)

    return written

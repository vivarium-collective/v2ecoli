"""Generic "Analysis flush" entrypoint for a completed multi-node
process-bigraph composite dispatch (backlog item 88 -- e.g. a colony
composite spread across N Ray-cluster nodes).

Reads whatever ``viva_api.compose.run_pbg``'s generic runner persisted for
the just-completed dispatch (``emitter_history.json``, gathered from the
composite's own in-memory emitter when no file-backed emitter already
shipped its own output -- see that module's ``_persist_emitter_history``;
falls back to ``final_state.json``, the always-present final-snapshot-only
default, when no history was captured) and renders a self-contained HTML
report + ``_manifest.json``, matching the SAME S3-manifest contract every
other analysis kind already writes (``written``/``errors`` -- see
``viva_api.common.handlers.analyses.handle_get_ray_analysis_status``), so a
multi-node composite's analysis is exactly as discoverable through
``GET /analyses/{id}/status`` as a hand-triggered one.

Resolves the right renderer from ``composite_id`` (``_RENDERERS`` below) --
today that's ``ColonyVisualization`` for the colony composite; a future
multi-node composite registers its own the same way. Nothing in this
module's own dispatch logic hardcodes colony.

Usage:
    python scripts/run_multi_node_analysis.py \
        --composite-id v2ecoli.composites.ecoli_colony.ecoli_colony \
        --history-uri s3://bucket/vecoli-output/<experiment_id> \
        --out-uri s3://bucket/vecoli-output/<experiment_id>/analyses/<name> \
        --experiment-id <experiment_id>
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

# Same real gap run_standalone_analysis.py's own comment documents: `scripts`
# (no __init__.py, an implicit namespace package) only resolves when the repo
# root is also on sys.path -- not guaranteed by every invocation shape.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _run_aws(args: list[str], tries: int = 3, backoff_s: float = 5.0) -> None:
    """Run an ``aws`` CLI subprocess, retrying transient failures -- mirrors
    run_standalone_analysis.py's own identical helper (real, previously-hit
    S3 network blips, always transient)."""
    last_err: subprocess.CalledProcessError | None = None
    for attempt in range(1, tries + 1):
        try:
            subprocess.run(
                args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
            )
            return
        except subprocess.CalledProcessError as e:
            last_err = e
            if attempt < tries:
                time.sleep(backoff_s * attempt)
    assert last_err is not None
    raise last_err


def _aws_cp(src: str, dst: str) -> None:
    _run_aws(["aws", "s3", "cp", src, dst])


def _try_download(src: str, dst: Path) -> bool:
    """Best-effort ``aws s3 cp``: returns False (not an error) when the
    object simply doesn't exist -- e.g. a composite whose emitter couldn't be
    gathered generically (see ``run_pbg._persist_emitter_history``'s own
    best-effort contract), or a document with no file-backed emitter at all."""
    try:
        _aws_cp(src, str(dst))
        return dst.exists()
    except subprocess.CalledProcessError:
        return False


def _unpack_entry(entry: Any) -> tuple[float, dict[str, Any]]:
    """One gathered-emitter-result entry is either a ``(time, state)`` tuple
    or a flat dict carrying its own ``time`` key -- the same dual shape
    ``colony_report.py``'s own ``_generate_chromosome_gif`` already handles
    (process-bigraph's ``gather_emitter_results`` output shape), mirrored
    here rather than assumed."""
    if isinstance(entry, list | tuple) and len(entry) == 2:
        t, data = entry
        return float(t), data if isinstance(data, dict) else {}
    if isinstance(entry, dict):
        return float(entry.get("time", 0) or 0), entry
    return 0.0, {}


def _flatten_agent_history(emitter_entries: list[Any]) -> list[dict[str, Any]]:
    """Flatten a colony composite's gathered ``agents`` (cells) snapshots
    into the flat per-tick-per-agent rows ``ColonyVisualization`` expects --
    the same shape ``reports/colony_report.py``'s own live simulation loop
    builds (``agent_id``/``time``/``x``/``y``/``length``/``mass``), but from
    ALREADY-PERSISTED history instead of a live ``sim.state``."""
    rows: list[dict[str, Any]] = []
    for entry in emitter_entries:
        t, data = _unpack_entry(entry)
        agents = data.get("agents") or {}
        if not isinstance(agents, dict):
            continue
        for agent_id, cell in agents.items():
            if not isinstance(cell, dict):
                continue
            loc = cell.get("location") or (0.0, 0.0)
            x = float(loc[0]) if loc else 0.0
            y = float(loc[1]) if loc and len(loc) > 1 else 0.0
            rows.append(
                {
                    "agent_id": agent_id,
                    "time": t,
                    "x": x,
                    "y": y,
                    "length": float(cell.get("length", 0.0) or 0.0),
                    "mass": float(cell.get("mass", 0.0) or 0.0),
                }
            )
    return rows


def _derive_env_size(
    history_rows: list[dict[str, Any]], default: float = 40.0, margin: float = 5.0
) -> float:
    """No original dispatch config is available post-hoc (only the persisted
    trajectory) -- derive a reasonable plot bound from the real observed
    positions instead of requiring the exact original ``env_size``. A
    composite whose analysis needs the precise original value should carry
    it in its own emitted state; this is a sane, honest default, not a
    silent guess presented as authoritative."""
    if not history_rows:
        return default
    extent = max(
        (max(abs(r["x"]), abs(r["y"])) for r in history_rows), default=default / 2
    )
    return extent * 2 + margin


def _render_colony(
    emitter_history: dict[str, Any] | None,
    final_state: dict[str, Any] | None,
    tmp: Path,
) -> str:
    """Render via ``ColonyVisualization`` -- the colony composite's own
    registered renderer. Degrades gracefully (matching that Step's own
    documented contract) when history is unavailable: still produces a
    summary-tables-only report from ``final_state`` rather than failing the
    whole analysis. The chromosome-state GIF is deliberately NOT reproduced
    here -- it was built in ``colony_report.py``'s own interactive wrapper
    from bespoke ``EcoliWCM`` instance introspection during a LIVE run, not
    data a generic post-hoc reader has access to; ``ColonyVisualization``
    already renders correctly with ``chrom_gif_b64`` absent.
    """
    from bigraph_schema import allocate_core

    from v2ecoli.visualizations.colony import ColonyVisualization

    emitter_entries = (emitter_history or {}).get("emitter") or []
    history_rows = _flatten_agent_history(emitter_entries)

    colony_gif_b64 = None
    if history_rows:
        colony_gif_b64 = _colony_gif_b64(emitter_entries, tmp)

    final_cells = (final_state or {}).get("cells") or {}
    metadata: dict[str, Any] = {
        "n_final": len(final_cells) if isinstance(final_cells, dict) else "?",
        "n_emitter_frames": len(emitter_entries),
        "colony_gif_b64": colony_gif_b64,
    }
    if history_rows:
        times = [r["time"] for r in history_rows]
        metadata["duration_min"] = (
            round((max(times) - min(times)) / 60, 1) if times else "?"
        )
        metadata["env_size"] = round(_derive_env_size(history_rows))

    viz = ColonyVisualization(
        config={"title": "E. coli Colony Simulation"}, core=allocate_core()
    )
    result = viz.update({"history": history_rows, "metadata": metadata})
    html: str = result["html"]
    return html


def _colony_gif_b64(emitter_entries: list[Any], tmp: Path) -> str | None:
    """Generate the colony spatial GIF from persisted emitter entries, reusing
    ``viva_munk``'s own real GIF renderer (the same one ``colony_report.py``'s
    interactive wrapper calls) -- best-effort: a rendering failure degrades to
    no GIF (the report still renders with its summary tables), never fails
    the whole analysis."""
    import base64

    try:
        from viva_munk.plots.multibody_plots import simulation_to_gif

        rows = _flatten_agent_history(emitter_entries)
        env_size = _derive_env_size(rows)
        gif_path = tmp / "colony.gif"
        skip = max(1, len(emitter_entries) // 100)
        simulation_to_gif(
            emitter_entries,
            config={"env_size": env_size},
            agents_key="agents",
            filename=gif_path.name,
            out_dir=str(tmp),
            skip_frames=skip,
            show_time_title=True,
            frame_duration_ms=100,
        )
        if not gif_path.exists():
            return None
        return base64.b64encode(gif_path.read_bytes()).decode("ascii")
    except Exception as e:  # noqa: BLE001 -- best-effort; a missing GIF is a degraded report, not a failed one
        print(
            f"run_multi_node_analysis: colony GIF generation failed or unsupported: {type(e).__name__}: {e}"
        )
        return None
    finally:
        # A failed simulation_to_gif call can leave matplotlib Figures open
        # (observed once, rarely, as a flake in repeated-call test runs --
        # this process may render several analyses in sequence). Closing
        # unconditionally is cheap and correct regardless of whether this was
        # the true cause; never lets rendering state leak into the next call.
        try:
            import matplotlib.pyplot as plt

            plt.close("all")
        except Exception:  # noqa: BLE001 -- best-effort cleanup, never itself a reason to fail
            pass


_RENDERERS: dict[
    str, Callable[[dict[str, Any] | None, dict[str, Any] | None, Path], str]
] = {
    "v2ecoli.composites.ecoli_colony.ecoli_colony": _render_colony,
}


def run(
    *, composite_id: str, history_uri: str, out_uri: str, experiment_id: str, tmp: Path
) -> dict[str, Any]:
    renderer = _RENDERERS.get(composite_id)
    written: list[str] = []
    errors: list[dict[str, str]] = []

    if renderer is None:
        errors.append(
            {
                "error": f"no registered multi-node analysis renderer for composite_id {composite_id!r}"
            }
        )
    else:
        history_local = tmp / "emitter_history.json"
        final_state_local = tmp / "final_state.json"
        has_history = _try_download(
            f"{history_uri.rstrip('/')}/emitter_history.json", history_local
        )
        has_final_state = _try_download(
            f"{history_uri.rstrip('/')}/final_state.json", final_state_local
        )

        emitter_history = json.loads(history_local.read_text()) if has_history else None
        final_state = (
            json.loads(final_state_local.read_text()) if has_final_state else None
        )

        if emitter_history is None and final_state is None:
            errors.append(
                {
                    "error": f"neither emitter_history.json nor final_state.json found under {history_uri}"
                }
            )
        else:
            try:
                html = renderer(emitter_history, final_state, tmp)
                html_local = tmp / "report.html"
                html_local.write_text(html)
                dest = f"{out_uri.rstrip('/')}/report.html"
                _aws_cp(str(html_local), dest)
                written.append(dest)
            except Exception as e:  # noqa: BLE001 -- surface any render failure in the manifest, don't crash silently
                errors.append({"error": f"{type(e).__name__}: {e}"})

    status = (
        "done"
        if written and not errors
        else ("failed" if errors and not written else "partial")
    )
    manifest = {
        "analysis_kind": "multi-node-composite",
        "composite_id": composite_id,
        "experiment_id": experiment_id,
        "written": written,
        "errors": errors,
        "status": status,
    }
    manifest_local = tmp / "_manifest.json"
    manifest_local.write_text(json.dumps(manifest, indent=2))
    _aws_cp(str(manifest_local), f"{out_uri.rstrip('/')}/_manifest.json")
    return manifest


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--composite-id", required=True)
    p.add_argument(
        "--history-uri",
        required=True,
        help="s3:// prefix where run_pbg.py staged this dispatch's output",
    )
    p.add_argument(
        "--out-uri",
        required=True,
        help="s3:// prefix to write report.html + _manifest.json",
    )
    p.add_argument("--experiment-id", required=True)
    args = p.parse_args()

    with tempfile.TemporaryDirectory() as td:
        manifest = run(
            composite_id=args.composite_id,
            history_uri=args.history_uri,
            out_uri=args.out_uri,
            experiment_id=args.experiment_id,
            tmp=Path(td),
        )
    print(json.dumps(manifest, indent=2))
    if manifest["status"] == "failed":
        sys.exit(1)


if __name__ == "__main__":
    main()

"""Generic "Analysis flush" entrypoint for a completed multi-node
process-bigraph composite dispatch (backlog item 88 -- e.g. a colony
composite spread across N Ray-cluster nodes).

Downloads whatever ``viva_api.compose.run_pbg``'s generic runner persisted
for the just-completed dispatch (``emitter_history.json``, gathered from the
composite's own in-memory emitter when no file-backed emitter already
shipped its own output via ``_redirect_emitters`` -- see that module's
``_persist_emitter_history``; and/or ``final_state.json``, the
always-present final-snapshot default) into a local directory, then hands it
to v2ecoli's own real, generic, already-tested post-run analysis/
visualization mechanism -- ``v2ecoli.workflow.flush.run_flush`` -- the SAME
mechanism the flush already dispatches for every other composite (cd1_*/
ptools_* baseline analyses). Every step ``run_flush`` discovers is applied
generically (via ``iter_post_sim``); nothing in THIS script's own dispatch
logic hardcodes any one composite (colony included) -- a composite-specific
renderer, if one is ever needed, is a new registered post-sim step (see
``v2ecoli.workflow.post_sim_visualizations.EmitterHistorySummary`` for
the first one, itself fully generic: it renders whatever it finds under
``out_dir`` regardless of which composite produced it), never a per-composite
branch in this script.

Writes ``_manifest.json`` matching the SAME S3-manifest contract every other
analysis kind already writes (``written``/``errors`` -- see
``viva_api.common.handlers.analyses.handle_get_ray_analysis_status``), so a
multi-node composite's analysis is exactly as discoverable through
``GET /analyses/{id}/status`` as a hand-triggered one.

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
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

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
            subprocess.run(args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
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


def run(*, composite_id: str, history_uri: str, out_uri: str, experiment_id: str, tmp: Path) -> dict[str, Any]:
    run_dir = tmp / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    has_history = _try_download(f"{history_uri.rstrip('/')}/emitter_history.json", run_dir / "emitter_history.json")
    has_final_state = _try_download(f"{history_uri.rstrip('/')}/final_state.json", run_dir / "final_state.json")

    written: list[str] = []
    errors: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []

    if not has_history and not has_final_state:
        errors.append(
            {"error": f"neither emitter_history.json nor final_state.json found under {history_uri}"}
        )
    else:
        try:
            import v2ecoli.workflow.post_sim_visualizations  # noqa: F401 -- populates VISUALIZATION_REGISTRY
            from v2ecoli.workflow.flush import run_flush

            result = run_flush(str(run_dir), config={}, ws_root=os.getcwd())
        except Exception as e:  # noqa: BLE001 -- surface any flush failure in the manifest, don't crash silently
            errors.append({"error": f"run_flush failed: {type(e).__name__}: {e}"})
        else:
            # run_flush's own per-step skips (e.g. a parquet-only analysis that
            # doesn't apply to this composite's non-parquet output shape) are
            # expected, benign no-ops for THIS script's purposes -- recorded for
            # debuggability, but deliberately not folded into `errors` (which
            # drives this manifest's own status below).
            skipped.extend(result.get("skipped") or [])
            for entry in result.get("placed") or []:
                local_path = Path(entry["path"])
                dest = f"{out_uri.rstrip('/')}/{local_path.name}"
                try:
                    _aws_cp(str(local_path), dest)
                    written.append(dest)
                except subprocess.CalledProcessError as e:
                    stderr = e.stderr.decode()[:500] if e.stderr else str(e)
                    errors.append({"name": entry.get("name", "?"), "error": f"upload failed: {stderr}"})

    status = "done" if written and not errors else ("failed" if errors and not written else "partial")
    manifest = {
        "analysis_kind": "multi-node-composite",
        "composite_id": composite_id,
        "experiment_id": experiment_id,
        "written": written,
        "errors": errors,
        "skipped": skipped,
        "status": status,
    }
    manifest_local = tmp / "_manifest.json"
    manifest_local.write_text(json.dumps(manifest, indent=2))
    _aws_cp(str(manifest_local), f"{out_uri.rstrip('/')}/_manifest.json")
    return manifest


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--composite-id", required=True)
    p.add_argument("--history-uri", required=True, help="s3:// prefix where run_pbg.py staged this dispatch's output")
    p.add_argument("--out-uri", required=True, help="s3:// prefix to write analysis output + _manifest.json")
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

"""scripts/run_standalone_analysis.py must build real rows from summary.json
files and drive the actual ANALYSIS_REGISTRY -- the standalone K8s analysis
path this replaces silently pulled a nonexistent image for every Ray-backend
simulation (never worked), so this entrypoint carries the whole burden of
proof that analysis genuinely executes for this dispatch shape."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_standalone_analysis import build_multiseed_rows, run


def _fake_aws_cp(monkeypatch, seed_summaries: dict[int, dict], written: dict[str, str]):
    """Stub _aws_cp: 'download' pre-seeded summaries by writing them to the
    requested local dest; 'upload' by recording dest -> local content."""
    import scripts.run_standalone_analysis as mod

    def fake(src: str, dst: str) -> None:
        if src.startswith("s3://"):
            seed = int(src.rsplit("seed_", 1)[1].split("/", 1)[0])
            Path(dst).write_text(json.dumps(seed_summaries[seed]))
        else:
            written[dst] = Path(src).read_text()

    monkeypatch.setattr(mod, "_aws_cp", fake)


def test_build_multiseed_rows_reflects_no_division(tmp_path, monkeypatch):
    written: dict[str, str] = {}
    _fake_aws_cp(monkeypatch, {
        0: {"seed": 0, "dry_mass_fg": 220.5},
        1: {"seed": 1, "dry_mass_fg": 221.1},
    }, written)

    rows = build_multiseed_rows("s3://bucket/exp", 2, tmp_path)

    assert len(rows) == 2
    assert all(r["divided"] is False and r["division_time"] == 0.0 for r in rows)
    assert [r["final_dry_mass"] for r in rows] == [220.5, 221.1]


def test_run_writes_real_doubling_time_result_and_manifest(tmp_path, monkeypatch):
    written: dict[str, str] = {}
    _fake_aws_cp(monkeypatch, {
        0: {"seed": 0, "dry_mass_fg": 200.0},
        1: {"seed": 1, "dry_mass_fg": 240.0},
    }, written)

    manifest = run(
        out_uri="s3://bucket/exp", n_seeds=2,
        modules={"multiseed": {"doubling_time_distribution": {}}},
        analysis_name="test-analysis", tmp=tmp_path,
    )

    assert manifest["status"] == "done"
    assert manifest["written"] == ["s3://bucket/exp/analyses/test-analysis/doubling_time_distribution.json"]
    result = json.loads(written["s3://bucket/exp/analyses/test-analysis/doubling_time_distribution.json"])
    assert result["n_cells"] == 2
    assert result["n_divided"] == 0  # this dispatch never simulates division
    assert result["final_dry_mass_mean"] == 220.0
    manifest_written = json.loads(written["s3://bucket/exp/analyses/test-analysis/_manifest.json"])
    assert manifest_written == manifest


def test_run_records_error_for_unknown_module(tmp_path, monkeypatch):
    written: dict[str, str] = {}
    _fake_aws_cp(monkeypatch, {0: {"seed": 0, "dry_mass_fg": 200.0}}, written)

    manifest = run(
        out_uri="s3://bucket/exp", n_seeds=1,
        modules={"multiseed": {"not_a_real_module": {}}},
        analysis_name="test-analysis", tmp=tmp_path,
    )

    assert manifest["status"] == "failed"
    assert manifest["written"] == []
    assert "not_a_real_module" in manifest["errors"][0]["name"]


def test_run_records_error_for_missing_seed_summary(tmp_path, monkeypatch):
    import scripts.run_standalone_analysis as mod

    def always_fail(src: str, _dst: str) -> None:
        import subprocess
        if src.startswith("s3://"):  # only the seed-summary download should fail
            raise subprocess.CalledProcessError(1, ["aws"], stderr=b"NoSuchKey")

    monkeypatch.setattr(mod, "_aws_cp", always_fail)

    manifest = run(
        out_uri="s3://bucket/exp", n_seeds=1,
        modules={"multiseed": {"doubling_time_distribution": {}}},
        analysis_name="test-analysis", tmp=tmp_path,
    )

    assert manifest["status"] == "failed"
    assert "no summary.json" in manifest["errors"][0]["error"]

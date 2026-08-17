"""A real, third bug in the same family as item 60's other two: `.dockerignore`
blanket-excluded `reports/`, so `reports/_summary/render.py` -- real source
`v2ecoli/workflow/analyses/comparison_matrix.py` imports (`from reports._summary
import render`), reachable via "applicable" module resolution -- never made it
into the deployed image at all. Real crash: ModuleNotFoundError: No module
named 'reports', from inside a real Ray Batch container.

Root cause: `.gitignore` already has the correct pattern (`reports/*` +
`!reports/_summary/`, treating that one subtree as source, not derived report
output) -- `.dockerignore` never got the equivalent exception.

Primary verification was a real `docker build` against a minimal busybox
Dockerfile, confirming `reports/_summary/render.py` lands in the build context
while `reports/assets/` (real generated output) stays correctly excluded. This
test is the fast, CI-friendly regression guard using the same ignore-pattern
semantics (via `pathspec`, already a real dependency) so a future edit to
`.dockerignore` can't silently reintroduce this without failing loudly here --
it does not replace the real docker-build verification, which is the actual
proof this works.
"""

from pathlib import Path

import pathspec

REPO_ROOT = Path(__file__).resolve().parent.parent


def _dockerignore_spec():
    lines = (REPO_ROOT / ".dockerignore").read_text().splitlines()
    # "gitwildmatch" (not the newer "gitignore" factory) is deliberate: it's
    # the one empirically cross-checked against a real `docker build` for
    # this exact pattern set (reports/summaries/* correctly excluded) --
    # pathspec's newer "gitignore" mode diverges on single-level-wildcard
    # subdirectory recursion and would make this test silently wrong.
    return pathspec.PathSpec.from_lines("gitwildmatch", lines)


def test_reports_summary_ships_real_source_analysis_modules_need():
    spec = _dockerignore_spec()
    assert not spec.match_file("reports/_summary/render.py"), (
        "reports/_summary/ must ship in the docker build context -- "
        "v2ecoli/workflow/analyses/comparison_matrix.py imports from it "
        "(from reports._summary import render), reachable via the "
        "'applicable' analysis-module keyword (item 60)"
    )
    assert not spec.match_file("reports/_summary/__init__.py")
    assert not spec.match_file("reports/_summary/aggregate.py")


def test_reports_generated_output_still_excluded():
    """The fix must be scoped to reports/_summary/ only -- not a blanket
    un-ignore of the whole reports/ tree, which would bloat the image with
    real generated report-card/dashboard output."""
    spec = _dockerignore_spec()
    assert spec.match_file("reports/assets/some-generated.html")
    assert spec.match_file("reports/summaries/some-run.html")


def test_every_reports_and_scripts_path_registered_analyses_actually_import_ships():
    """Real, current cross-check (not the fixed list above alone): grep every
    registered analysis module (v2ecoli/workflow/analyses/*.py) for its real
    `from reports...`/`from scripts...` imports, resolve each to a concrete
    file path, and confirm none of them are docker-ignored. If a future
    analysis module adds a new such import outside `reports/_summary/`, this
    fails loudly instead of silently shipping a fourth bug in this family."""
    import re

    spec = _dockerignore_spec()
    analyses_dir = REPO_ROOT / "v2ecoli" / "workflow" / "analyses"
    import_lines = set()
    for py_file in analyses_dir.glob("*.py"):
        import_lines.update(re.findall(r"^from ((?:reports|scripts)\.\S+) import", py_file.read_text(), re.MULTILINE))

    assert import_lines, "expected at least one reports./scripts. import to exist (sanity check)"
    for module_path in import_lines:
        rel = module_path.replace(".", "/")
        # Resolve against the real filesystem -- module_path may name a leaf
        # module (`scripts/compare_matched_trajectories.py`) or a package
        # (`reports/_summary/`, an __init__.py directory); don't guess.
        if (REPO_ROOT / f"{rel}.py").is_file():
            real_path = f"{rel}.py"
        elif (REPO_ROOT / rel).is_dir():
            real_path = f"{rel}/__init__.py"
        else:
            raise AssertionError(f"{module_path} (from a real import) resolves to neither {rel}.py nor {rel}/ on disk")
        assert not spec.match_file(real_path), (
            f"{module_path} is imported by a registered analysis module but "
            f"{real_path} is docker-ignored -- this is exactly item 60's third bug"
        )

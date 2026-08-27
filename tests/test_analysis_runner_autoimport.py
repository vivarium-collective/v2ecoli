"""``run_analyses`` must self-register the built-in analyses.

``run_analyses`` resolves every requested analysis name against
``ANALYSIS_REGISTRY`` and skips (with an ``"unknown analysis"`` warning) any
name that isn't there. Those names only register as a side effect of importing
the analysis modules, and nothing in the workflow run path imports the
``v2ecoli.workflow.analyses`` suite. So before this fix a bare
``python -m v2ecoli.workflow.run`` found the registry empty and silently
dropped EVERY declared built-in analysis. ``_register_builtin_analyses``
(called at the top of ``run_analyses``) closes that gap.

Verified in a fresh subprocess so the registry genuinely starts unpopulated —
importing analysis modules is a cached, process-wide side effect, so an
in-process test would be contaminated by any earlier test that imported them.
"""
import subprocess
import sys
import textwrap


def test_builtin_analyses_absent_until_registered():
    """In a clean interpreter, importing the base ``analysis`` module does NOT
    register the built-in suite; ``_register_builtin_analyses`` does."""
    code = textwrap.dedent(
        """
        from v2ecoli.workflow.analysis import ANALYSIS_REGISTRY
        # Base module alone: the built-in suite is NOT yet registered.
        assert "cd1_transcriptomics" not in ANALYSIS_REGISTRY, (
            "precondition failed: something imported the analyses suite before "
            "_register_builtin_analyses was called")
        from v2ecoli.workflow.analysis_runner import _register_builtin_analyses
        _register_builtin_analyses()
        assert "cd1_transcriptomics" in ANALYSIS_REGISTRY
        assert "ptools_rna" in ANALYSIS_REGISTRY
        print("REGISTERED_OK")
        """
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, f"stdout={r.stdout!r}\nstderr={r.stderr!r}"
    assert "REGISTERED_OK" in r.stdout


def test_register_builtin_analyses_is_idempotent():
    """Calling it twice is safe (the underlying import is cached)."""
    code = textwrap.dedent(
        """
        from v2ecoli.workflow.analysis_runner import _register_builtin_analyses
        from v2ecoli.workflow.analysis import ANALYSIS_REGISTRY
        _register_builtin_analyses()
        n1 = len(ANALYSIS_REGISTRY)
        _register_builtin_analyses()
        n2 = len(ANALYSIS_REGISTRY)
        assert n1 == n2, (n1, n2)
        print("IDEMPOTENT_OK")
        """
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, f"stdout={r.stdout!r}\nstderr={r.stderr!r}"
    assert "IDEMPOTENT_OK" in r.stdout

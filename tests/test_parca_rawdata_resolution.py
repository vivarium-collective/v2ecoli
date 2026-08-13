"""raw_data resolution for the ParCa InitializeStep (workbench-run guard).

Regression coverage for vivarium-workbench #752: running the registered
``parca`` composite through the workbench's generic runner fires
``InitializeStep`` with ``raw_data`` materialised to an empty ``dict`` (the
registered document carries ``raw_data=None`` — it was only ever runnable via
the ``v2ecoli-parca`` CLI, which injects a real ``KnowledgeBaseEcoli``). The
first attribute access ``raw_data.operons_on`` then raised
``AttributeError: 'dict' object has no attribute 'operons_on'``.

``_resolve_raw_data`` closes that gap: a valid KB is used as-is, anything else
triggers a real KB load so the composite is self-sufficient from the workbench.
These tests patch ``KnowledgeBaseEcoli`` so they stay fast and hermetic (a real
load reads the flat-file TSVs).
"""

from unittest.mock import MagicMock, patch

import pytest

from v2ecoli.processes.parca.steps import step_01_initialize as s1


def test_valid_raw_data_passes_through_without_constructing_a_kb():
    """A real KB (has ``.operons_on``) is returned untouched — no rebuild."""
    kb = MagicMock()
    kb.operons_on = True
    with patch.object(s1, "KnowledgeBaseEcoli") as mock_kb:
        out = s1._resolve_raw_data({"raw_data": kb})
    assert out is kb
    mock_kb.assert_not_called()


@pytest.mark.parametrize("bad", [None, {}, {"operons_on_typo": True}])
def test_invalid_raw_data_autoloads_a_real_kb(bad):
    """None / a plain dict (the workbench-materialised value) triggers a load."""
    sentinel = MagicMock()
    sentinel.operons_on = True
    with patch.object(s1, "KnowledgeBaseEcoli", return_value=sentinel) as mock_kb:
        out = s1._resolve_raw_data({"raw_data": bad})
    assert out is sentinel
    mock_kb.assert_called_once()
    # Production genotype defaults (mirror v2ecoli/cli/parca.py): operons on.
    assert mock_kb.call_args.kwargs.get("operons_on") is True


def test_declared_bundle_manifest_is_honored_on_autoload():
    """A declared bundle_manifest flows into the SourceBundle used for the load."""
    sentinel = MagicMock()
    sentinel.operons_on = True
    with patch.object(s1, "KnowledgeBaseEcoli", return_value=sentinel), \
         patch.object(s1, "SourceBundle") as mock_bundle:
        s1._resolve_raw_data(
            {"raw_data": {}, "bundle_manifest": "/some/manifest.json"})
    mock_bundle.assert_called_once()
    assert mock_bundle.call_args.kwargs.get("base_manifest") == "/some/manifest.json"

"""Panels in the ensemble/gov-cloud report (scripts/comparison_report_card.py):
the converted-process panel must show each converted process's RESULTING
process-bigraph schema (port types), and the v2ecoli full-config panel must
render from a build-config dict in EVERY mode (so a 'basal' run that injected
processes no longer reads as bare 'basal')."""
from scripts.comparison_report_card import (
    converted_processes_section, v2_config_section, vecoli_config_section)


def _build_with_injection():
    return {
        "engine": "v2ecoli", "condition": "basal", "seed": 0,
        "time_step": 1.0, "n_processes": 2,
        "options": {"overrides": {"injected_processes": {
            "fork_repo": "/abs/vEcoli",
            "add_processes": ["ecoli-mock-secretion"],
            "swap_processes": {},
        }}},
        "processes": [
            {"name": "ecoli-metabolism", "address": "local:ecoli-metabolism",
             "type": "process", "config_keys": ["media_id"], "interface": None},
            {"name": "ecoli-mock-secretion",
             "address": "local:ecoli-mock-secretion", "type": "process",
             "config_keys": ["rate"],
             "interface": {
                 "inputs": {"bulk": "bulk_array", "timestep": "float"},
                 "outputs": {"bulk": "bulk_array"}}},
        ],
        "topology": {
            "ecoli-metabolism": {"inputs": {}, "outputs": {}},
            "ecoli-mock-secretion": {"inputs": {"bulk": ["bulk"]},
                                     "outputs": {"bulk": ["bulk"]}}},
    }


def test_converted_panel_shows_resulting_schema():
    sec = converted_processes_section("basal", _build_with_injection())
    assert sec is not None and sec["kind"] == "content"
    html = sec["html"]
    assert "ecoli-mock-secretion" in html
    assert "vivarium-1.0" in html
    # the RESULTING process-bigraph schema (port names + types) is rendered
    assert "bulk" in html and "bulk_array" in html
    assert "timestep" in html and "float" in html


def test_converted_panel_none_without_injection():
    assert converted_processes_section("basal", {"processes": []}) is None


def test_v2_config_panel_renders_full_process_set():
    sec = v2_config_section("basal", _build_with_injection())
    assert sec is not None and sec["kind"] == "content"
    html = sec["html"]
    # the FULL config — not just the bare 'basal' label — lists the real
    # process set including the injected process
    assert "ecoli-metabolism" in html
    assert "ecoli-mock-secretion" in html


def test_v2_config_panel_none_without_build():
    assert v2_config_section("basal", None) is None


def test_vecoli_config_panel_renders_actual_process_set():
    ve_build = {
        "engine": "vecoli", "source": "EcoliSim.config",
        "condition": "basal", "seed": 0, "time_step": 1.0,
        "media_id": "minimal", "n_processes": 2,
        "processes": ["ecoli-metabolism", "ecoli-transcript-initiation"],
        "exclude_processes": ["monomer_counts_listener"]}
    sec = vecoli_config_section("basal", ve_build)
    assert sec is not None and sec["kind"] == "content"
    html = sec["html"]
    assert "ecoli-metabolism" in html
    # vEcoli ran its own (default) process set — the mock is NOT here
    assert "mock-secretion" not in html
    assert "vEcoli" in sec["title"]


def test_vecoli_config_panel_none_without_build():
    assert vecoli_config_section("basal", None) is None

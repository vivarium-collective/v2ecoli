import json


def test_trajectory_to_gif_history_skips_cells_without_location():
    from v2ecoli.colony_bench.viz import trajectory_to_gif_history
    traj = [
        {"time": 0.0, "cells": {
            "a": {"location": (1.0, 2.0), "length": 2.0, "radius": 0.5, "angle": 0.1, "mass": 0.04},
            "b": {"length": 2.0},  # no location -> skipped
        }},
    ]
    hist = trajectory_to_gif_history(traj)
    assert len(hist) == 1
    agents = hist[0]["agents"]
    assert "a" in agents and "b" not in agents
    assert agents["a"]["type"] == "segment"
    assert agents["a"]["location"] == (1.0, 2.0)


def test_render_phenotype_figures_writes_three_pngs(tmp_path):
    from v2ecoli.colony_bench.viz import render_phenotype_figures
    phenotypes = {
        "size_at_division": {"length": [3.9, 4.1, 4.0, 3.8], "mass": [0.08, 0.08, 0.08, 0.08]},
        "interdivision_time": [1200.0, 1300.0, 1250.0, 1280.0],
        "added_length": [
            {"birth_length": 2.0, "delta_length": 1.9},
            {"birth_length": 2.1, "delta_length": 2.0},
            {"birth_length": 1.9, "delta_length": 2.1},
        ],
    }
    paths = render_phenotype_figures(phenotypes, tmp_path, label="mother machine")
    names = {p.name for p in paths}
    assert names == {"size_at_division.png", "interdivision_time.png", "added_size.png"}
    for p in paths:
        assert p.exists() and p.stat().st_size > 0
        meta = json.loads(p.with_suffix(".meta.json").read_text(encoding="utf-8"))
        assert "title" in meta and "caption" in meta


def test_render_phenotype_figures_handles_empty(tmp_path):
    from v2ecoli.colony_bench.viz import render_phenotype_figures
    empty = {"size_at_division": {"length": [], "mass": []},
             "interdivision_time": [], "added_length": []}
    paths = render_phenotype_figures(empty, tmp_path, label="daughter machine")
    assert len(paths) == 3
    assert all(p.exists() for p in paths)

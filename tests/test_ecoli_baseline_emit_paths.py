"""``filter_listener_paths`` — the emit-path allowlist that lets a study emit
only the listener leaves it needs (mass, a few array listeners) instead of the
full ~400-column listener dump. Pure function, no ParCa cache required."""
from v2ecoli.composites.ecoli_baseline import filter_listener_paths

ALL = [
    "listeners.mass.cell_mass",
    "listeners.mass.dry_mass",
    "listeners.mass.growth",
    "listeners.monomer_counts",
    "listeners.rna_counts.mRNA_cistron_counts",
    "listeners.fba_results.base_reaction_fluxes",
    "listeners.rna_synth_prob.actual_rna_synth_prob",
    "listeners.enzyme_kinetics.metabolite_counts_final",
]


def test_empty_emit_paths_keeps_everything():
    assert filter_listener_paths(ALL, None) == ALL
    assert filter_listener_paths(ALL, []) == ALL


def test_prefix_match_keeps_all_leaves_under_a_namespace():
    # "listeners.mass" keeps all three mass leaves, drops the rest.
    kept = filter_listener_paths(ALL, [["listeners", "mass"]])
    assert kept == [
        "listeners.mass.cell_mass",
        "listeners.mass.dry_mass",
        "listeners.mass.growth",
    ]


def test_accepts_tuple_and_dotted_string_forms_together():
    kept = filter_listener_paths(ALL, [
        ["listeners", "mass"],
        "listeners.monomer_counts",
        ["listeners", "rna_counts", "mRNA_cistron_counts"],
        "listeners.fba_results.base_reaction_fluxes",
    ])
    assert "listeners.mass.cell_mass" in kept
    assert "listeners.monomer_counts" in kept
    assert "listeners.rna_counts.mRNA_cistron_counts" in kept
    assert "listeners.fba_results.base_reaction_fluxes" in kept
    # everything not declared is dropped
    assert "listeners.rna_synth_prob.actual_rna_synth_prob" not in kept
    assert "listeners.enzyme_kinetics.metabolite_counts_final" not in kept
    assert len(kept) == 6


def test_exact_leaf_match_does_not_over_capture_a_sibling_prefix():
    # "listeners.monomer_counts" must NOT match a hypothetical
    # "listeners.monomer_counts_extra" (prefix guard uses "." boundary).
    paths = ["listeners.monomer_counts", "listeners.monomer_counts_extra.x"]
    kept = filter_listener_paths(paths, ["listeners.monomer_counts"])
    assert kept == ["listeners.monomer_counts"]


def test_no_match_yields_empty():
    assert filter_listener_paths(ALL, ["listeners.does_not_exist"]) == []

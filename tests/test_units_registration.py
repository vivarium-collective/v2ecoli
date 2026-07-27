def test_v2ecoli_registers_units_resolver():
    import v2ecoli.visualizations  # noqa: F401  (import triggers registration)
    from viva_superpowers.visualization import Visualization
    assert Visualization.units_resolver is not None
    assert Visualization.units_resolver("listeners.mass.cell_mass") == "fg"

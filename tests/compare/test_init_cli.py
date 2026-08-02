import yaml
from scripts._compare.scaffold import scaffold_investigation


def test_scaffold_writes_comparison_block(tmp_path):
    p = scaffold_investigation(
        name="wcm-cmp", reference_repo="/abs/vEcoli",
        configs=["basal", "with_aa", "configs/redux.json"], out_root=tmp_path)
    data = yaml.safe_load(p.read_text())
    comp = data["comparison"]
    assert comp["candidate"] == "v2ecoli"
    assert comp["reference"] == {"repo": "/abs/vEcoli", "kind": "vecoli"}
    names = [c["name"] for c in comp["configs"]]
    assert names == ["basal", "with_aa", "redux"]     # path → basename stem
    redux = [c for c in comp["configs"] if c["name"] == "redux"][0]
    assert redux["config"] == "configs/redux.json"

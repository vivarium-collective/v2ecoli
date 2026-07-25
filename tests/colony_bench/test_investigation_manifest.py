import yaml, pathlib

INV = pathlib.Path("workspace/investigations/colonies/investigation.yaml")

def test_manifest_lists_part_b_studies():
    data = yaml.safe_load(INV.read_text())
    studies = data["studies"]
    for s in ["colonies-01-hpc-readiness", "colonies-02-parallel-multigen-perf",
              "colonies-03-wcm-rss-leak", "colonies-04-device-phenotype-harness"]:
        assert s in studies, f"missing {s}"
    assert "phenotype" in data["title"].lower() or "phenotype" in data["question"].lower()

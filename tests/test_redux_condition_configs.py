import json
from scripts.gen_redux_condition_configs import build_redux_config

def test_build_redux_config_merges_swap_block_and_condition():
    cond = {"experiment_id": "cond_acetate", "condition": "acetate"}
    basal_redux = {
        "experiment_id": "metabolism_redux_basal", "condition": "basal",
        "swap_processes": {"ecoli-metabolism": "ecoli-metabolism-redux"},
        "exclude_processes": ["exchange_data"],
        "flow": {"ecoli-metabolism-redux": [["ecoli-chromosome-structure"]]},
        "strip_pint_ports": {"ecoli-metabolism-redux": ["listeners"]},
        "attach_pint_ports": {"ecoli-metabolism-redux": {"boundary": "mM"}},
        "output_ports": {"ecoli-metabolism-redux": ["bulk", "environment"]},
    }
    out = build_redux_config(cond, basal_redux)
    # condition comes from the cond config; swap block from basal redux
    assert out["condition"] == "acetate"
    assert out["swap_processes"] == {"ecoli-metabolism": "ecoli-metabolism-redux"}
    assert out["flow"]["ecoli-metabolism-redux"] == [["ecoli-chromosome-structure"]]
    assert out["strip_pint_ports"] and out["attach_pint_ports"] and out["output_ports"]
    assert out["experiment_id"] == "metabolism_redux_acetate"

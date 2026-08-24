"""Bake / re-key the model-side omics fixtures by EcoCyc id.

Two operations, neither of which re-runs a simulation:

``--rekey-proteome``
    Add a ``by_id`` map (EcoCyc monomer id) to the committed
    ``model_proteome.json`` **without changing a single value**. The fixture is
    keyed by gene symbol today, and symbol is not a safe join key across sources
    (symbol spaces differ between databases and are not injective across them),
    so a cross-source consumer currently has no id-keyed join available at all.
    The monomer order in sim_data is what ties the two together: position i of
    ``monomer_counts`` is both ``ids[i]`` and ``symbols[i]``, so the existing
    ``by_symbol`` values can be re-filed under their ids exactly. Verified
    bijective for this build (4309 monomers, 4309 unique ids, 4309 unique
    symbols) — the script asserts it rather than assuming it.

``--transcriptome``
    Write ``model_transcriptome.json`` keyed by EcoCyc **gene** id. By default
    the vector is taken from the pinned reference
    (``population_phenotype_basal_reference.json``), which already carries this
    run's ensemble-mean transcriptome — so the committed transcriptome and the
    committed proteome describe the *same* ensemble. Passing ``--from-sweep``
    instead derives it from a sweep through the run-keyed cache.

Why not re-bake from a sweep? The committed fixtures come from a 20-cell
(4 seeds x 8 gens, gen>=3) ensemble that is no longer on disk. Re-baking from a
different sweep would silently change every value the basal card grades, which
is a separate decision from adding an id key.

Run:
    python scripts/bake_model_omics.py --rekey-proteome --transcriptome
    python scripts/bake_model_omics.py --transcriptome --from-sweep out/<sweep>
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # scripts/ for _provenance

from _provenance import provenance_stamp  # noqa: E402

FIXTURES = REPO / "tests/fixtures/population_phenotype_basal"
REFERENCE = REPO / "tests/fixtures/population_phenotype_basal_reference.json"
_PARCA_STATE = REPO / "out/sim_data_full/parca_state.pkl.gz"
GEN_LB = 3


def _load_parca_state():
    from v2ecoli.processes.parca.data_loader import load_parca_state
    if not _PARCA_STATE.exists():
        raise SystemExit(
            f"need {_PARCA_STATE} for the id vocabulary (build it with "
            f"scripts/build_cache.py / the full-mode ParCa)")
    return load_parca_state(str(_PARCA_STATE))


def _labels(state):
    from v2ecoli.library.gene_meta import omics_labels
    return omics_labels(state)


def _keyed(vector, keys):
    """``({key: value}, n_unmapped, n_collisions)``; collisions are summed."""
    out, n_unmapped, n_collisions = {}, 0, 0
    for k, v in zip(keys, vector):
        if not k or k == "None":
            n_unmapped += 1
            continue
        if k in out:
            n_collisions += 1
        out[k] = out.get(k, 0.0) + float(v)
    return out, n_unmapped, n_collisions


def rekey_proteome(labels) -> dict:
    """Add ``by_id`` to model_proteome.json, preserving every value exactly."""
    path = FIXTURES / "model_proteome.json"
    pro = json.load(open(path, encoding="utf-8"))
    by_symbol = pro["by_symbol"]
    ids = [str(m) for m in labels["proteome"]["ids"]]
    symbols = [str(s) for s in labels["proteome"]["symbols"]]

    # The re-key is only exact if symbol -> id is 1:1 over the monomer order.
    live = [(i, s) for i, s in zip(ids, symbols) if s and s != "None"]
    if len({s for _, s in live}) != len(live):
        raise SystemExit(
            "symbol -> monomer-id is not 1:1 for this build, so by_symbol cannot "
            "be re-keyed exactly; re-bake from the sweep instead")
    missing = [s for _, s in live if s not in by_symbol]
    if missing:
        raise SystemExit(
            f"{len(missing)} symbols in sim_data are absent from the committed "
            f"by_symbol (e.g. {missing[:5]}) — the fixture and this sim_data "
            f"disagree; refusing to re-key")

    by_id = {i: by_symbol[s] for i, s in live}
    # Same values, new key: assert it rather than trusting it.
    assert sorted(by_id.values()) == sorted(by_symbol[s] for _, s in live)

    pro["by_id"] = by_id
    pro["id_key"] = "EcoCyc monomer id"
    pro["keying"] = {
        "n_monomers": len(ids),
        "n_by_id": len(by_id),
        "n_by_symbol": len(by_symbol),
        "n_id_unmapped": len(ids) - len(live),
        "n_symbol_collisions": 0,
        "note": ("by_id is a re-key of the existing by_symbol values (identical "
                 "numbers, EcoCyc monomer id instead of gene symbol), not a "
                 "re-bake — the source ensemble is unchanged."),
    }
    path.write_text(json.dumps(pro, indent=2))
    return pro


def transcriptome_from_reference(labels) -> dict:
    """Key the pinned reference's transcriptome vector by EcoCyc gene id."""
    ref = json.load(open(REFERENCE, encoding="utf-8"))
    crit = ref["axes"]["omics.transcriptome"]["criterion"]
    tx = labels["transcriptome"]
    if list(crit["ids"]) != [str(c) for c in tx["ids"]]:
        raise SystemExit(
            "the pinned reference's transcriptome ids do not match this "
            "sim_data's mRNA cistron order — the reference was pinned against a "
            "different ParCa state, so its vector cannot be keyed with these "
            "gene ids")
    ensemble = ref["stimulus"]["ensemble"]
    m = re.search(r"->\s*(\d+)\s*cells", ensemble)
    if not m:
        raise SystemExit(f"cannot read n_cells from reference ensemble: {ensemble!r}")
    return _node(crit["ref_vector"], tx, int(m.group(1)),
                 "pinned reference vector (per-cell time-mean "
                 "mRNA_cistron_counts, gen>=%d), keyed by EcoCyc gene id" % GEN_LB,
                 {"kind": "pinned_reference", "reference": REFERENCE.name,
                  "ensemble": ensemble,
                  "blessed_model_ref": ref["stimulus"].get("blessed_model_ref")})


def transcriptome_from_sweep(sweep_dir: Path, labels) -> dict:
    """Key a sweep's transcriptome vector (via the run-keyed cache) by gene id."""
    from v2ecoli.library.sim_vector_cache import load_or_extract
    env = load_or_extract(str(sweep_dir), GEN_LB)
    vec = (env["vectors"].get("omics") or {}).get("transcriptome")
    if not vec:
        raise SystemExit(f"no transcriptome vector extracted from {sweep_dir}")
    return _node(vec["vector"], labels["transcriptome"], vec["n_cells"],
                 "sweep; per-cell time-mean mRNA_cistron_counts, gen>=%d" % GEN_LB,
                 {"kind": "sweep", "sweep_dir": str(sweep_dir),
                  "experiment_id": env["run"]["experiment_id"],
                  "run_commit": env["provenance"]["run_commit"]})


def _node(vector, tx_labels, n_cells, method, source) -> dict:
    gene_ids = [str(g) for g in tx_labels["gene_ids"]]
    if len(gene_ids) != len(vector):
        raise SystemExit(
            f"gene-id/count width mismatch: {len(gene_ids)} vs {len(vector)}")
    by_gene, n_unmapped, n_collisions = _keyed(vector, gene_ids)
    return {"n_cells": n_cells, "gen_lb": GEN_LB, "method": method,
            "units": "counts/cell", "id_key": "EcoCyc gene id",
            "source": source, "by_gene_id": by_gene,
            "keying": {"n_cistrons": len(vector), "n_by_gene_id": len(by_gene),
                       "n_unmapped": n_unmapped, "n_collisions": n_collisions}}


def backfill_model_ref() -> list[str]:
    """Record the fixtures' model ref in their provenance.

    The five fixtures carry a ``reconstructed: true`` stamp whose producing
    commit is null. The identity was never actually lost, though — the sibling
    pinned reference records ``stimulus.blessed_model_ref`` for the same
    ensemble. Copying it across makes the fixtures self-describing without
    claiming a clean bake: ``reconstructed`` stays true, and ``code.commit``
    stays null, because the ref identifies the *model* the ensemble came from,
    not the tree that ran the bake.
    """
    ref = json.load(open(REFERENCE, encoding="utf-8"))
    model_ref = ref["stimulus"].get("blessed_model_ref")
    ensemble = ref["stimulus"].get("ensemble")
    if not model_ref:
        raise SystemExit(f"{REFERENCE.name} carries no stimulus.blessed_model_ref")
    touched = []
    for path in sorted(FIXTURES.glob("model_*.json")):
        blob = json.load(open(path, encoding="utf-8"))
        prov = blob.get("provenance")
        if not isinstance(prov, dict) or prov.get("blessed_model_ref"):
            continue
        prov["blessed_model_ref"] = model_ref
        prov["blessed_model_ref_source"] = (
            f"{REFERENCE.name} stimulus.blessed_model_ref ({ensemble})")
        path.write_text(json.dumps(blob, indent=2))
        touched.append(path.name)
    return touched


def main(rekey: bool, transcriptome: bool, from_sweep: str | None,
         backfill: bool = False) -> None:
    if not (rekey or transcriptome or backfill):
        raise SystemExit(
            "nothing to do: pass --rekey-proteome / --transcriptome / --backfill-model-ref")
    if backfill:
        touched = backfill_model_ref()
        print(f"backfilled blessed_model_ref into {len(touched)} fixtures: "
              + ", ".join(touched) if touched else "blessed_model_ref already present")
        if not (rekey or transcriptome):
            return
    labels = _labels(_load_parca_state())

    if rekey:
        pro = rekey_proteome(labels)
        k = pro["keying"]
        print(f"model_proteome.json: re-keyed — {k['n_by_id']} by_id / "
              f"{k['n_by_symbol']} by_symbol over {k['n_monomers']} monomers "
              f"({k['n_id_unmapped']} unmapped, {k['n_symbol_collisions']} "
              f"symbol collisions); values unchanged")

    if transcriptome:
        if from_sweep:
            node = transcriptome_from_sweep(Path(from_sweep), labels)
        else:
            node = transcriptome_from_reference(labels)
        node["provenance"] = provenance_stamp(
            REPO, config="v2ecoli/configs/population_phenotype_basal.json",
            sweep={"sweep_dir": node["source"].get("sweep_dir"),
                   "parca_inputs_hash": None,
                   "n_cells": node["n_cells"], "gen_lb": GEN_LB},
            bake_script="scripts/bake_model_omics.py --transcriptome")
        (FIXTURES / "model_transcriptome.json").write_text(json.dumps(node, indent=2))
        k = node["keying"]
        print(f"model_transcriptome.json: {node['n_cells']} cells, "
              f"{k['n_by_gene_id']} gene ids over {k['n_cistrons']} mRNA cistrons "
              f"({k['n_unmapped']} unmapped, {k['n_collisions']} collisions), "
              f"source {node['source']['kind']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rekey-proteome", action="store_true",
                    help="add by_id to model_proteome.json (values unchanged)")
    ap.add_argument("--transcriptome", action="store_true",
                    help="write model_transcriptome.json keyed by EcoCyc gene id")
    ap.add_argument("--from-sweep", metavar="DIR", default=None,
                    help="derive the transcriptome from this sweep (via the "
                         "run-keyed cache) instead of the pinned reference")
    ap.add_argument("--backfill-model-ref", action="store_true",
                    help="record the pinned reference's blessed_model_ref in "
                         "each fixture's provenance (provenance only, no values)")
    a = ap.parse_args()
    main(rekey=a.rekey_proteome, transcriptome=a.transcriptome,
         from_sweep=a.from_sweep, backfill=a.backfill_model_ref)

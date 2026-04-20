"""Build the plasmid-enabled online-sim cache (`out/cache_plasmid/`).

Combines:
  - the in-tree plasmid-aware ParCa fixture
    (``models/parca/parca_state.pkl.gz``, hydrated to a SimulationDataEcoli),
  - a known-good baseline ``simData.cPickle`` (e.g. from a vEcoli ParCa
    run; the fast-mode v2ecoli pickle on its own has initial-state
    issues that crash the online sim on the first step),
into a patched sim_data, then runs ``save_cache(..., has_plasmid=True)``.

Usage::

    uv run python scripts/build_plasmid_cache.py
    uv run python scripts/build_plasmid_cache.py \\
        --baseline out/workflow/simData.cPickle \\
        --cache-dir out/cache_plasmid

The baseline is searched in this order if ``--baseline`` is omitted:
  1. ``out/workflow/simData.cPickle``
  2. ``out/kb/baseline_simData.cPickle``
  3. ``$VECOLI_REPO/out/test_installation/parca/kb/simData.cPickle``
     (where ``VECOLI_REPO`` defaults to ``../vEcoli``)

If none exists, the script prints how to generate one via vEcoli's
``runscripts/parca`` (~30-70 min) and exits non-zero.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

PLASMID_REPLICATION_ATTRS = (
    'plasmid_sequence', 'plasmid_sequence_rc', 'plasmid_length',
    'plasmid_A_count', 'plasmid_T_count', 'plasmid_G_count', 'plasmid_C_count',
    'plasmid_ori_coordinate', 'plasmid_forward_sequence',
    'plasmid_forward_complement_sequence', 'plasmid_replichore_lengths',
    'plasmid_replication_sequences',
)
PLASMID_UNIQUE_MOLECULES = (
    'full_plasmid', 'plasmid_domain', 'oriV', 'plasmid_active_replisome',
)


def _find_baseline(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            sys.exit(f"--baseline {p} does not exist")
        return p
    candidates = [
        REPO_ROOT / 'out' / 'workflow' / 'simData.cPickle',
        REPO_ROOT / 'out' / 'kb' / 'baseline_simData.cPickle',
        Path(os.environ.get('VECOLI_REPO', REPO_ROOT.parent / 'vEcoli'))
            / 'out' / 'test_installation' / 'parca' / 'kb' / 'simData.cPickle',
    ]
    for p in candidates:
        if p.exists():
            print(f"  using baseline: {p}")
            return p
    print("ERROR: no baseline simData.cPickle found. Tried:", file=sys.stderr)
    for p in candidates:
        print(f"  - {p}", file=sys.stderr)
    print(
        "\nGenerate one with vEcoli's ParCa (this takes ~30-70 minutes):\n"
        "    cd $VECOLI_REPO   # or wherever vEcoli is checked out\n"
        "    uv run python -m runscripts.parca \\\n"
        "        --config configs/templates/parca_standalone.json \\\n"
        "        --outdir out/parca\n"
        "    cp out/parca/kb/simData.cPickle "
        f"{REPO_ROOT}/out/workflow/simData.cPickle\n",
        file=sys.stderr,
    )
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--baseline', default=None,
        help='Path to a known-good simData.cPickle (vEcoli ParCa output).')
    parser.add_argument('--parca-fixture', default='models/parca/parca_state.pkl.gz',
        help='Path to the in-tree plasmid-aware ParCa fixture.')
    parser.add_argument('--patched-out', default='out/kb/simData_plasmid_patched.cPickle',
        help='Where to write the patched sim_data pickle.')
    parser.add_argument('--cache-dir', default='out/cache_plasmid',
        help='Output directory for the online-sim cache.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--mechanistic-replisome', action='store_true',
        help='Build a cache that gates chromosome AND plasmid replication '
             'on full replisome subunit availability (DnaG, pol III core, '
             'β-clamp, DnaB, HolA at 2-per-oriC etc.). Off by default — '
             'use to reproduce the vEcoli DnaG-depletion lineage-collapse '
             'investigation in v2ecoli.')
    parser.add_argument('--dnag-lexa-scale', type=float, default=1.0,
        help='Scale the LexA delta_prob entry on TU00352[c] (the only '
             'dnaG-containing TU kept by ParCa) by this factor. Default '
             '1.0 = unchanged ParCa fit (full LexA repression on the '
             'housekeeping dnaG promoter, biologically wrong since rpsUp1 '
             'is not LexA-regulated in vivo). 0.0 = LexA removed entirely. '
             'Fractional values (e.g. 0.5) attenuate it, consistent with '
             'a "1-of-3 dnaG TUs is LexA-regulated" lumped-TU story: '
             'rpsUp3 is the SOS-inducible LexA target, but ParCa kept '
             'TU00352 (rpsUp1 housekeeping) and inherited rpsUp3\'s '
             'regulation onto it. Use the attenuation to dial the '
             'effective DnaG steady-state up toward the ~10 copies '
             'experimental target without wholly removing the regulation.')
    args = parser.parse_args()

    from v2ecoli.processes.parca.data_loader import (
        hydrate_sim_data_from_state, load_parca_state,
    )

    baseline_path = _find_baseline(args.baseline)
    print(f"  loading plasmid-aware fixture: {args.parca_fixture}")
    plasmid_state = load_parca_state(args.parca_fixture)
    sd_p = hydrate_sim_data_from_state(plasmid_state)
    if not getattr(sd_p.molecule_ids, 'full_plasmid', None):
        sys.exit(f"ERROR: {args.parca_fixture} has no full_plasmid id — "
                 "is the fixture plasmid-aware?")

    print(f"  loading baseline: {baseline_path}")
    with open(baseline_path, 'rb') as f:
        sd_good = pickle.load(f)

    sd_good.molecule_ids.full_plasmid = sd_p.molecule_ids.full_plasmid
    sd_good.molecule_ids.plasmid_ori = sd_p.molecule_ids.plasmid_ori
    for attr in PLASMID_REPLICATION_ATTRS:
        setattr(sd_good.process.replication, attr,
                getattr(sd_p.process.replication, attr))
    for k in PLASMID_UNIQUE_MOLECULES:
        sd_good.internal_state.unique_molecule.unique_molecule_definitions[k] = (
            sd_p.internal_state.unique_molecule.unique_molecule_definitions[k])
        if k not in sd_good.molecule_groups.unique_molecules_domain_index_division:
            sd_good.molecule_groups.unique_molecules_domain_index_division.append(k)
    sd_good.getter._all_plasmid_coordinates = sd_p.getter._all_plasmid_coordinates
    sd_good.getter._all_submass_arrays['full_plasmid'] = (
        sd_p.getter._all_submass_arrays['full_plasmid'])
    sd_good.getter._all_total_masses['full_plasmid'] = (
        sd_p.getter._all_total_masses['full_plasmid'])
    sd_good.getter._all_compartments['full_plasmid'] = (
        sd_good.getter._all_compartments.get('full_chromosome', ['c']))

    if args.dnag_lexa_scale != 1.0:
        import numpy as np
        tr = sd_good.process.transcription_regulation
        rna_ids = list(sd_good.process.transcription.rna_data['id'])
        tu_idx = rna_ids.index('TU00352[c]')
        lexa_idx = tr.tf_ids.index('PC00010')
        I = tr.delta_prob['deltaI']
        J = tr.delta_prob['deltaJ']
        V = tr.delta_prob['deltaV']
        mask = (I == tu_idx) & (J == lexa_idx)
        n_hit = int(mask.sum())
        if n_hit == 0:
            print("  --dnag-lexa-scale: no (TU00352, lexA) delta_prob entry "
                  "found — nothing to patch")
        elif args.dnag_lexa_scale == 0.0:
            keep = ~mask
            tr.delta_prob['deltaI'] = I[keep]
            tr.delta_prob['deltaJ'] = J[keep]
            tr.delta_prob['deltaV'] = V[keep]
            print(f"  --dnag-lexa-scale 0.0: removed {n_hit} "
                  f"(TU00352, lexA) delta_prob entry(ies); LexA no "
                  f"longer represses the housekeeping dnaG promoter.")
        else:
            old = float(V[mask][0])
            V[mask] = old * args.dnag_lexa_scale
            print(f"  --dnag-lexa-scale {args.dnag_lexa_scale}: "
                  f"attenuated (TU00352, lexA) delta_prob from {old:.4g} "
                  f"to {old * args.dnag_lexa_scale:.4g}.")

    os.makedirs(os.path.dirname(args.patched_out), exist_ok=True)
    with open(args.patched_out, 'wb') as f:
        pickle.dump(sd_good, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  wrote patched sim_data: {args.patched_out}")

    from v2ecoli.composite import save_cache
    save_cache(args.patched_out, cache_dir=args.cache_dir,
               seed=args.seed, has_plasmid=True,
               mechanistic_replisome=args.mechanistic_replisome)
    print(f"  cache ready: {args.cache_dir} "
          f"(mechanistic_replisome={args.mechanistic_replisome})")


if __name__ == '__main__':
    main()

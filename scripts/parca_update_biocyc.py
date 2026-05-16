#!/usr/bin/env python
"""
Refresh the 10 BioCyc-sourced TSVs under
``v2ecoli/processes/parca/reconstruction/ecoli/flat/`` from the EcoCyc
webservice.

Hits ``https://websvc.biocyc.org/wc-get?type=<file_id>`` for each file
and overwrites the local copy.  Keeps a cache of per-file fetch status
in ``out/compare/biocyc_meta.json`` so the parca comparison report can
display ``n/10 OK`` without refetching.

Usage::

    python scripts/parca_update_biocyc.py                    # fetch all
    python scripts/parca_update_biocyc.py --only genes,rnas  # subset
    python scripts/parca_update_biocyc.py --timeout 120      # per-request

Adapted from ``vEcoli/reconstruction/ecoli/scripts/update_biocyc_files.py``
and ``v2parca/scripts/compare_parca.py`` (the ``--fetch-biocyc`` path).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests


BIOCYC_FILE_IDS = [
    "complexation_reactions", "dna_sites", "equilibrium_reactions",
    "genes", "metabolic_reactions", "metabolites", "proteins",
    "rnas", "transcription_units", "trna_charging_reactions",
]
BASE_URL = "https://websvc.biocyc.org/wc-get?type="

REPO_ROOT = Path(__file__).resolve().parent.parent
FLAT_DIR = (REPO_ROOT / 'v2ecoli' / 'processes' / 'parca' /
            'reconstruction' / 'ecoli' / 'flat')
META_PATH = REPO_ROOT / 'out' / 'compare' / 'biocyc_meta.json'


def fetch_one(file_id: str, timeout: int) -> dict:
    try:
        r = requests.get(BASE_URL + file_id, timeout=timeout)
        r.raise_for_status()
        out = FLAT_DIR / f'{file_id}.tsv'
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(r.text)
        return {'bytes': len(r.text), 'lines': r.text.count('\n'),
                'status': 'ok'}
    except Exception as e:
        return {'bytes': 0, 'lines': 0, 'status': str(e)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--only', default='',
                   help='Comma-separated file_id subset (default: all 10).')
    p.add_argument('--timeout', type=int, default=60,
                   help='Per-request timeout in seconds (default: 60).')
    args = p.parse_args()

    ids = [s.strip() for s in args.only.split(',') if s.strip()] or BIOCYC_FILE_IDS
    unknown = [f for f in ids if f not in BIOCYC_FILE_IDS]
    if unknown:
        print(f'Unknown file_ids: {unknown}.  Known: {BIOCYC_FILE_IDS}',
              file=sys.stderr)
        return 2

    results = {}
    for fid in ids:
        print(f'  {fid:28s}', end=' ', flush=True)
        r = fetch_one(fid, timeout=args.timeout)
        results[fid] = r
        if r['status'] == 'ok':
            print(f"{r['bytes']:>10,d} bytes, {r['lines']:>5d} lines")
        else:
            print(f"FAILED: {r['status']}")
        time.sleep(1)  # be nice to BioCyc

    meta = {
        'when': time.strftime('%Y-%m-%d %H:%M:%S'),
        'n_files': len(BIOCYC_FILE_IDS),
        'n_fetched': sum(1 for v in results.values() if v['status'] == 'ok'),
        'files': results,
    }
    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Merge with any existing meta so partial runs don't clobber.
    if META_PATH.exists():
        try:
            prior = json.loads(META_PATH.read_text())
            prior_files = prior.get('files', {})
            prior_files.update(results)
            meta['files'] = prior_files
            meta['n_fetched'] = sum(1 for v in prior_files.values()
                                    if v['status'] == 'ok')
        except Exception:
            pass
    META_PATH.write_text(json.dumps(meta, indent=2))
    print(f"\n{meta['n_fetched']}/{meta['n_files']} OK "
          f"(meta: {META_PATH.relative_to(REPO_ROOT)})")
    return 0 if meta['n_fetched'] == len(ids) else 1


if __name__ == '__main__':
    sys.exit(main())

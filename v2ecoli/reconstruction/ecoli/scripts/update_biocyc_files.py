"""
Pulls the latest versions of the raw data files sourced from BioCyc/EcoCyc
from their webservices server and updates the local versions of the files
under reconstruction/ecoli/flat.

Usage:
    python -m v2ecoli.reconstruction.ecoli.scripts.update_biocyc_files
    python -m v2ecoli.reconstruction.ecoli.scripts.update_biocyc_files --preview
"""

import argparse
import os
import time

import requests

BASE_API_URL = "https://websvc.biocyc.org/wc-get?type="
BASE_API_URL_PREVIEW = "https://brg-preview.ai.sri.com/wc-get?type="

BIOCYC_FILE_IDS = [
    "complexation_reactions",
    "dna_sites",
    "equilibrium_reactions",
    "genes",
    "metabolic_reactions",
    "metabolites",
    "proteins",
    "rnas",
    "transcription_units",
    "trna_charging_reactions",
]

FLAT_DIR = os.path.join(
    os.path.dirname(__file__), '..', 'flat')


def update_biocyc_files(preview=False):
    """Fetch latest BioCyc flat files and write to flat/ directory.

    Args:
        preview: If True, use the preview server (requires auth).

    Returns:
        dict: {file_id: n_bytes} for each downloaded file.
    """
    base_url = BASE_API_URL_PREVIEW if preview else BASE_API_URL
    results = {}

    for file_id in BIOCYC_FILE_IDS:
        print(f"  Fetching {file_id}...", end=" ", flush=True)
        if preview:
            response = requests.get(
                base_url + file_id,
                auth=("<USERNAME>", "<PASSWORD>"))
        else:
            response = requests.get(base_url + file_id)
        response.raise_for_status()

        outpath = os.path.join(FLAT_DIR, file_id + ".tsv")
        with open(outpath, "w") as f:
            f.write(response.text)

        n_bytes = len(response.text)
        results[file_id] = n_bytes
        print(f"{n_bytes:,} bytes")
        time.sleep(1)  # Rate limit

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Import latest EcoCyc-sourced flat files from EcoCyc servers")
    parser.add_argument(
        "--preview", action="store_true",
        help="Import files from the preview server (requires auth).")
    args = parser.parse_args()
    update_biocyc_files(preview=args.preview)

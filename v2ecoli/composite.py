"""
Document generation and Composite loading for v2ecoli.

generate_document() — builds the E. coli state from EcoliSim, wraps steps
    as v2-native BigraphStep instances, and saves the document.

load_composite() — loads a saved document and returns a Composite ready to run.
"""

import os
import dill

from bigraph_schema import allocate_core, get_path
from process_bigraph import Composite

from contextlib import chdir

from wholecell.utils.filepath import ROOT_PATH
from ecoli.experiments.ecoli_master_sim import EcoliSim, CONFIG_DIR_PATH

from v2ecoli.types import ECOLI_TYPES


def _build_core():
    """Create and configure a bigraph-schema core with ecoli types."""
    core = allocate_core()
    core.register_types(ECOLI_TYPES)
    return core


def generate_document(outpath='out/ecoli.pickle'):
    """Build the E. coli composite document from EcoliSim.

    Creates the initial state from simData, infers schemas, and saves
    the document as a pickle file.

    Args:
        outpath: Path for the output file.

    Returns:
        The path to the saved file.
    """
    core = _build_core()

    with chdir(ROOT_PATH):
        sim = EcoliSim.from_file(CONFIG_DIR_PATH + "default.json")
        sim.build_ecoli()

    state = sim.generated_initial_state

    # TODO: wrap v1 step/process instances as v2-native BigraphStep/BigraphProcess
    # For now, store the raw initial state
    schema = core.infer(state)
    document = {'schema': schema, 'state': state}

    os.makedirs(os.path.dirname(outpath) or '.', exist_ok=True)
    with open(outpath, 'wb') as f:
        dill.dump(document, f)

    print(f"Saved document to {outpath}")
    return outpath


def load_composite(path='out/ecoli.pickle', core=None):
    """Load a saved document and return a Composite ready to run.

    Args:
        path: Path to the document produced by generate_document.
        core: Pre-configured bigraph-schema core. If None, creates one.

    Returns:
        A Composite ready for .run(interval).
    """
    if core is None:
        core = _build_core()

    with open(path, 'rb') as f:
        document = dill.load(f)

    composite = Composite(document, core=core)
    return composite

"""v2ecoli workflow framework: meta-composite variants × seeds × generations sweeps."""

from v2ecoli.workflow.config import load_config_with_inheritance
from v2ecoli.workflow.variants import expand_branches, BranchSpec
from v2ecoli.workflow.meta_composite import (
    build_meta_composite, register_workflow_processes)
from v2ecoli.workflow.run import run_workflow

__all__ = [
    "load_config_with_inheritance",
    "expand_branches",
    "BranchSpec",
    "build_meta_composite",
    "register_workflow_processes",
    "run_workflow",
]

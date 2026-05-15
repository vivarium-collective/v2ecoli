"""build_core() — wraps process_bigraph.allocate_core().

Imports declared in workspace.yaml are auto-discovered by allocate_core()
once they're pip-installed in the workspace venv. Use the dashboard's
Install button on a Registry catalog entry, or run
`.venv/bin/pip install -e <path>` manually. No manual register_link()
boilerplate needed for standard pbg-* packages.
"""
from process_bigraph import allocate_core


def build_core():
    return allocate_core()

#!/usr/bin/env bash
# Build the 3 Cython extensions vendored under
# v2ecoli/processes/parca/wholecell/utils/ for the current Python.
#
# Run once after a fresh clone or after Python/numpy upgrade.  Output is
# three .so files alongside the .pyx sources (gitignored).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
cd "$REPO_ROOT"
"$PY" -c "import Cython" 2>/dev/null || "$PY" -m pip install cython
"$PY" - <<'PY'
import numpy as np, sys
from Cython.Build import cythonize
from setuptools import Extension
from distutils.core import setup as dsetup
base = 'v2ecoli/processes/parca/wholecell/utils'
mods = ('_build_sequences', 'mc_complexation', '_fastsums')
exts = [Extension(f'v2ecoli.processes.parca.wholecell.utils.{m}',
                  [f'{base}/{m}.pyx'],
                  include_dirs=[np.get_include()],
                  define_macros=[('NPY_NO_DEPRECATED_API','NPY_1_7_API_VERSION')])
        for m in mods]
sys.argv = ['build', 'build_ext', '--inplace']
dsetup(name='v2ecoli-parca-cython', ext_modules=cythonize(exts, language_level=3))
PY
echo "done: $(ls $REPO_ROOT/v2ecoli/processes/parca/wholecell/utils/*.so | wc -l) .so files built"

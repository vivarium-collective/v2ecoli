"""Build the 3 Cython extensions vendored under
v2ecoli/processes/parca/wholecell/utils/.

Package metadata lives in pyproject.toml; this file exists only because
setuptools doesn't yet support declaring Cython ext_modules in the
PEP 621 metadata.
"""

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

_BASE = "v2ecoli/processes/parca/wholecell/utils"
_MODS = ("_build_sequences", "mc_complexation", "_fastsums")

ext_modules = [
    Extension(
        f"v2ecoli.processes.parca.wholecell.utils.{m}",
        [f"{_BASE}/{m}.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
    for m in _MODS
]

setup(ext_modules=cythonize(ext_modules, language_level=3))

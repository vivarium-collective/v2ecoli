FROM python:3.12-slim-bookworm

# System deps required by v2ecoli's scientific Python stack:
#   gfortran / libopenblas-dev / liblapack-dev — NumPy, SciPy, CVXPY compiled extensions
#   cmake — some CVXPY/numba build steps
#   libffi-dev — cffi (required by several packages)
#   git — uv source deps (vEcoli[dev], pbg-superpowers resolved via git)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        gfortran \
        cmake \
        git \
        curl \
        ca-certificates \
        libopenblas-dev \
        liblapack-dev \
        libffi-dev \
        python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv from the official image (same pattern as pbg-template).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    V2ECOLI_OUT=/app/out \
    V2ECOLI_RESULTS=/app/results

WORKDIR /app
COPY . .

# Pin to exactly 3.12.12 — matches requires-python = "==3.12.12" in pyproject.toml.
# The fallback omits --no-install-project in case the project build backend
# requires it (setuptools + cython/numpy are listed in build-system.requires).
RUN uv python install 3.12.12 && \
    (uv sync --python 3.12.12 --no-install-project || uv sync --python 3.12.12)

# Compile the three Cython extensions vendored under
# v2ecoli/processes/parca/wholecell/utils/ (mc_complexation, _build_sequences,
# _fastsums).  These are required by the ParCa pipeline and must be built
# in-place against the installed numpy/Python before the image is sealed.
# We use `uv run --with cython` rather than the helper shell script because
# cython is only in [build-system] (not project deps), so it is absent from
# .venv and `python -m pip` is unavailable in uv-managed environments.
RUN uv run --with cython python -c "\
import numpy as np, sys; \
from Cython.Build import cythonize; \
from setuptools import Extension; \
from distutils.core import setup as dsetup; \
base = 'v2ecoli/processes/parca/wholecell/utils'; \
mods = ('_build_sequences', 'mc_complexation', '_fastsums'); \
exts = [Extension('v2ecoli.processes.parca.wholecell.utils.' + m, [base + '/' + m + '.pyx'], include_dirs=[np.get_include()], define_macros=[('NPY_NO_DEPRECATED_API','NPY_1_7_API_VERSION')]) for m in mods]; \
sys.argv = ['build', 'build_ext', '--inplace']; \
dsetup(name='v2ecoli-parca-cython', ext_modules=cythonize(exts, language_level=3)) \
"

# Pre-create bind-mount targets so Singularity/Apptainer can mount them at
# runtime without needing write access to create them.
RUN mkdir -p /app/results /app/out

# Default entry point — the dispatch layer uses `singularity exec` which
# overrides this; CMD is mainly useful for direct `docker run` or local dev.
CMD ["uv", "run", "v2ecoli-parca", "--help"]

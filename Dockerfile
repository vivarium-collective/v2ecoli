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
RUN PYTHON=/app/.venv/bin/python bash scripts/parca_cython_build.sh

# Pre-create bind-mount targets so Singularity/Apptainer can mount them at
# runtime without needing write access to create them.
RUN mkdir -p /app/results /app/out

# Default entry point — the dispatch layer uses `singularity exec` which
# overrides this; CMD is mainly useful for direct `docker run` or local dev.
CMD ["uv", "run", "v2ecoli-parca", "--help"]

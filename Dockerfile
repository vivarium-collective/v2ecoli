# syntax=docker/dockerfile:1
#
# v2ecoli simulation SOFTWARE image — the simulation code + its locked dependency
# environment ONLY. It deliberately contains NO model/ParCa data: the ParCa cache
# (out/cache) is computed (`scripts/build_cache.py`) or fetched at job time, never baked.
#
# Ray comes from v2ecoli's own lock (2.55.x), so the Ray version matches across every
# node of a cluster. On AWS, the Ray entrypoint + AWS CLI are layered on top of this image
# by sms-cdk/docker/ray.Dockerfile (built with BASE_IMAGE=<this image>).
#
# Build (from the repo root):
#   docker build -t v2ecoli:dev .
FROM python:3.12-bookworm

# uv (pinned binary) for fast, lock-faithful installs.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Build toolchain for the vendored Cython extensions + git for the git-main deps.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app/v2ecoli
COPY . .

# Install v2ecoli + its locked deps into an in-project venv. Include the `ray` and
# emitter extras (the multiseed fan-out + XArray/Parquet output); skip dev tooling and
# the heavy vivarium-dashboard (mirrors CI's `--no-install-package vivarium-dashboard`).
# UV_LINK_MODE=copy makes the venv self-contained (real wheel copies, not links into the
# cache). The cache itself is a BuildKit cache mount below, so it is reused across builds
# but NOT baked into the image layer — this is what keeps the image from ballooning toward
# ~16 GB with a duplicated wheel cache.
ENV UV_LINK_MODE=copy
# v2ecoli pins requires-python == 3.12.12 exactly; let uv fetch that managed interpreter.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv python install 3.12.12
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev --extra ray --extra emitters --no-install-package vivarium-dashboard \
 || uv sync --no-dev --extra ray --no-install-package vivarium-dashboard

# Put the venv on PATH so `python`, `ray`, and the `v2ecoli-*` console scripts resolve —
# both for non-login shells (ENV PATH) and LOGIN shells (the entrypoint runs the workload
# via `bash -lc`, which re-sources /etc/profile and would otherwise drop the venv).
ENV PATH="/app/v2ecoli/.venv/bin:${PATH}"
RUN printf 'export PATH="/app/v2ecoli/.venv/bin:$PATH"\n' > /etc/profile.d/10-v2ecoli.sh

# Sanity: the package imports and Ray is present at the locked version.
RUN python -c "import v2ecoli, ray; print('v2ecoli ok; ray', ray.__version__)"

# syntax=docker/dockerfile:1
#
# v2ecoli simulation SOFTWARE image — the simulation code + its locked dependency
# environment ONLY. It deliberately contains NO model/ParCa data: the ParCa cache
# (out/cache) is computed (`scripts/build_cache.py`) or fetched at job time, never baked.
#
# Ray comes from v2ecoli's own lock (2.55.x), so the Ray version matches across every
# node of a cluster. This image is SELF-CONTAINED for AWS Batch multi-node-parallel runs:
# it bundles the AWS CLI and the generic Batch-MNP Ray entrypoint (docker/ray-batch-
# entrypoint.sh), so no downstream layering is required — sms-api builds it (workload-owned,
# like vEcoli's image) and sms-cdk only provides the MNP job def + build infra.
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

# vEcoli `composite-softfloor` checkout as a SIBLING of /app/v2ecoli, so the
# comparison driver (scripts/run_comparison_ensemble.py --composite vecoli) can
# sys.path-override the installed `ecoli` release with the composite-branch code
# — i.e. run the ORIGINAL vEcoli model as a process-bigraph composite
# (`build_composite_native`) WITH the ppGpp soft-floor, on Ray, no Nextflow.
# Pinned to the soft-floor commit so the image is reproducible.
ARG VECOLI_COMPOSITE_REF=composite-softfloor
RUN git clone --depth 1 --branch "${VECOLI_COMPOSITE_REF}" \
        https://github.com/vivarium-collective/vEcoli.git /app/vEcoli \
 && python -c "import sys; sys.path.insert(0, '/app/vEcoli'); from ecoli.composites.ecoli_composite import build_composite_native; print('vEcoli composite (soft-floor) importable')"

# ─── AWS Batch multi-node-parallel (Ray) runtime layer ───────────────────────
# Make the image self-contained for Ray-on-Batch: the AWS CLI (stages the ParCa
# cache in + the zarr results out, per the entrypoint's RAY_STAGE_*/RAY_OUT_* env)
# and the generic Batch-MNP Ray entrypoint. The image runs as root with WORKDIR
# /app/v2ecoli, so the entrypoint can stage the cache under /app. The entrypoint is
# env-driven and a no-op outside Batch, so it doesn't affect local/dev use.
RUN apt-get update && apt-get install -y --no-install-recommends curl unzip ca-certificates \
 && case "$(uname -m)" in \
      x86_64)  awscli_url="https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" ;; \
      aarch64) awscli_url="https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip" ;; \
      *) echo "unsupported arch: $(uname -m)" >&2; exit 1 ;; \
    esac \
 && curl -fsSL "$awscli_url" -o /tmp/awscliv2.zip \
 && unzip -q /tmp/awscliv2.zip -d /tmp \
 && /tmp/aws/install \
 && rm -rf /tmp/aws /tmp/awscliv2.zip \
 && apt-get clean && rm -rf /var/lib/apt/lists/* \
 && aws --version

COPY docker/ray-batch-entrypoint.sh /opt/ray-batch-entrypoint.sh
RUN chmod +x /opt/ray-batch-entrypoint.sh

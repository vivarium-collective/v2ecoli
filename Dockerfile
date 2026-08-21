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
# the heavy vivarium-workbench (mirrors CI's `--no-install-package vivarium-workbench`).
# UV_LINK_MODE=copy makes the venv self-contained (real wheel copies, not links into the
# cache). The cache itself is a BuildKit cache mount below, so it is reused across builds
# but NOT baked into the image layer — this is what keeps the image from ballooning toward
# ~16 GB with a duplicated wheel cache.
ENV UV_LINK_MODE=copy
# v2ecoli pins requires-python == 3.12.12 exactly; let uv fetch that managed interpreter.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv python install 3.12.12
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev --extra ray --extra emitters --no-install-package vivarium-workbench \
 || uv sync --no-dev --extra ray --no-install-package vivarium-workbench

# Put the venv on PATH so `python`, `ray`, and the `v2ecoli-*` console scripts resolve —
# both for non-login shells (ENV PATH) and LOGIN shells (the entrypoint runs the workload
# via `bash -lc`, which re-sources /etc/profile and would otherwise drop the venv).
ENV PATH="/app/v2ecoli/.venv/bin:${PATH}"
RUN printf 'export PATH="/app/v2ecoli/.venv/bin:$PATH"\n' > /etc/profile.d/10-v2ecoli.sh

# Sanity: the package imports and Ray is present at the locked version.
RUN python -c "import v2ecoli, ray; print('v2ecoli ok; ray', ray.__version__)"

# matplotlib font resolution: the cd1_*/ptools_* analysis plotting stack requests
# "Arial" (seaborn's default sans-serif list puts it first), which this base image
# doesn't have. Live-confirmed impact, THREE times: a 10-seed x 10-generation cd1_*
# analysis job produced zero output after 75+ minutes, log saturated with
# repeated "Font family 'Arial' not found" warnings.
#
# Two prior attempts, both confirmed live to fail, root cause traced precisely
# each time rather than guessed again:
# 1. fonts-liberation (metric-compatible substitute) alone does NOT fix this —
#    matplotlib's font_manager matches by each font FILE's own internal
#    name-table metadata, not fontconfig aliases, and Liberation Sans's own
#    metadata says "Liberation Sans", never "Arial".
# 2. Rewriting DejaVu Sans's own name-table to say "Arial" (via fonttools) DID
#    make matplotlib resolve "Arial" correctly -- but only at BUILD time (the
#    build's own assert confirmed it). At RUNTIME, in the real AWS Batch
#    container, the exact same 998/1000-Arial-warning symptom reproduced
#    unchanged -- meaning matplotlib's font cache/config directory resolution
#    genuinely differs between the Docker build context and the actual job
#    execution context (default resolution depends on $HOME, which the
#    entrypoint's own login-shell invocation -- `bash -lc`, see above -- can
#    plausibly re-derive differently than a plain `RUN` step).
#
# Real fix: stop relying on default HOME-based cache-directory resolution
# entirely. Pin MPLCONFIGDIR to one explicit, fixed, world-writable path via
# ENV, so the exact same directory is used to build the cache now AND to read
# it at container-run time later, regardless of shell type, login vs.
# non-login invocation, or HOME differences.
ENV MPLCONFIGDIR=/opt/mplconfig
RUN mkdir -p /opt/mplconfig /usr/share/fonts/truetype/arial-alias \
 && chmod 777 /opt/mplconfig \
 && python - <<'PYEOF'
import os, shutil
import matplotlib
from fontTools.ttLib import TTFont

src = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans.ttf")
dst = "/usr/share/fonts/truetype/arial-alias/Arial.ttf"
shutil.copy(src, dst)

font = TTFont(dst)
for record in font["name"].names:
    if record.nameID in (1, 4, 6, 16):  # family / full / postscript / typographic-family
        record.string = "Arial"
    elif record.nameID in (2, 17):  # subfamily / typographic-subfamily
        record.string = "Regular"
font.save(dst)
PYEOF
RUN fc-cache -f \
 && python -c "\
import matplotlib.font_manager as fm; \
fm.fontManager.addfont('/usr/share/fonts/truetype/arial-alias/Arial.ttf'); \
fm.fontManager.__init__(); \
names = {f.name for f in fm.fontManager.ttflist}; \
assert 'Arial' in names, f'Arial alias failed to register: {sorted(names)}'; \
print('Arial now resolves via matplotlib font_manager:', 'Arial' in names)"

# PRISTINE upstream CovertLab/vEcoli checkout as a SIBLING of /app/v2ecoli, so
# the comparison driver (scripts/run_comparison_ensemble.py --composite vecoli)
# runs the ORIGINAL, UNMODIFIED vEcoli model as a process-bigraph composite via
# the EXTERNAL wrapper (v2ecoli.library.vecoli_pbg_upstream + upstream_division):
# upstream EcoliSim is imported from here and each vivarium process wrapped
# externally, with ZERO edits to the checkout. V2E_VECOLI_DIR points the wrapper
# at it. The fork to WRAP is spec-driven: docker/build-and-push-ecr.sh reads
# comparison_spec.json's vecoli.{repo,commit} and passes them as these build-args,
# so pointing the comparison at a NEW vEcoli fork is a pure spec edit + rebuild —
# no Dockerfile change. Defaults clone the pristine upstream CovertLab/vEcoli@master.
# VECOLI_UPSTREAM_REF may be a branch OR a commit sha; we fetch+checkout so either
# works (a bare `clone --branch <sha>` does not accept a commit).
ARG VECOLI_UPSTREAM_REPO=https://github.com/CovertLab/vEcoli
ARG VECOLI_UPSTREAM_REF=master
RUN git clone "${VECOLI_UPSTREAM_REPO}" /app/vEcoli \
 && git -C /app/vEcoli checkout "${VECOLI_UPSTREAM_REF}"
# Compile upstream vEcoli's Cython extensions in-place (wholecell.utils.
# _build_sequences / mc_complexation / _fastsums). The image reaches vEcoli via
# V2E_VECOLI_DIR — it is NOT pip-installed — so its Cython must be built here, or
# the on-node ParCa build (build_upstream_parca.py) and every --composite vecoli
# run fail with "No module named 'wholecell.utils._build_sequences'". numpy is in
# the venv; install Cython (build-only) and use the image's build toolchain.
RUN --mount=type=cache,target=/root/.cache/uv \
    (python -c "import Cython" 2>/dev/null || uv pip install "Cython==3.1.2") \
 && cd /app/vEcoli && python setup.py build_ext --inplace
ENV V2E_VECOLI_DIR=/app/vEcoli
# Verify the clone + that the external upstream wrapper imports and can reach
# upstream EcoliSim through its sys.path/Cython-pinning shim (does NOT build a
# composite — that needs an upstream sim_data, staged at job time).
RUN python -c "import os; os.environ.setdefault('V2E_VECOLI_DIR', '/app/vEcoli'); \
from v2ecoli.library.vecoli_pbg_upstream import _ensure_upstream; \
print('upstream EcoliSim importable:', _ensure_upstream()['EcoliSim'].__name__)"

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

COPY docker/batch-array-entrypoint.sh /opt/batch-array-entrypoint.sh
RUN chmod +x /opt/batch-array-entrypoint.sh

COPY docker/batch-container-entrypoint.sh /opt/batch-container-entrypoint.sh
RUN chmod +x /opt/batch-container-entrypoint.sh

# DnaA-oriC stepped-Adair succinate reference run — reproduction guide

Reproduces the 12-generation succinate-media DnaA-oriC dynamics reference:
the `dnaa5_stepped_adair_100_100_50_10_3_3_3_3_unlock60_sustain1_seed{1,4}_12gen`
lineages summarised in
`dnaa5_stepped_adair_100_100_50_10_3_3_3_3_unlock60_sustain1_seeds1and4_report.pdf`.

Pinned to commit `b577de4b` on `feat/aim2-dnaa-oric-box-binding`.

Set this up as a new dashboard study under the theme **cooperativity via
binding affinity** — the stepped Adair K_d ladder implements cooperativity
by lowering the K_d of each successive oriC-low site as more sites become
occupied, rather than through a Hill-style rate law.

---

## What the reference run shows

- Sat-init gate fires when the oriC low-affinity cluster reaches 8/8 bound
  DnaA-ATP sustained for ≥ 1 s.
- Stepped Adair K_d ladder `100, 100, 50, 10, 3, 3, 3, 3` nM across the eight
  low-affinity sites gives the cooperative fill required to drive that
  saturation transition.
- Parent cooperativity requires **instantaneous positive bulk gradient**
  (via `COOP_GRADIENT_GATE`) — the Adair ladder collapses to Langmuir
  K_d_max whenever bulk DnaA-ATP isn't rising.
- Daughter oriCs additionally require **60 s cumulative positive bulk
  gradient** (via `POST_INIT_UNLOCK_S`) before their K_d ladder is
  eligible to engage. Any negative-gradient tick resets the daughter
  counter.
- Result: mostly one initiation per generation (occasional reinitiations),
  τ mean ≈ 68.9 min, cell mass at initiation ~ 743 fg, bulk DnaA-ATP
  sawtooth 10-30 nM.

---

## 1. Check out the pinned commit

Check out `b577de4b` however you prefer (branch, worktree, detached HEAD):

```bash
git checkout b577de4b
uv sync --extra emitters
```

## 2. Wire in the ParCa fixture, simulation cache, and steady-state dill

The steady-state dill used to reproduce the reference PDF is committed at
`out/steady_state_inputs/dnaa5_v1.5_hillKd_h4_K3_seed4_gen5.dill`. The
launch script expects it at
`out/dnaa5_v1.5_hillKd_h4_K3_seed4/gen_dills/gen5.dill`, so drop it in
place.

For a deeper burned-in starting state, use
`out/steady_state_inputs/milestone_fromGen8_seed5_gen5.dill` instead —
~13 generations of stepped-Adair sat-init dynamics pre-loaded, starts
at milestone cell mass with no warmup drift.

```bash
# ParCa fixture must match the runtime cache (SHA a6f1cbb7...)
# If not already present, restore from the fixture kept on main.

# Succinate V=1.5e-3 apo+ATP kinetic cache
mkdir -p out
# Link or copy your local cache to:
#   out/cache_dnaa2_v1.5e-3_kd3nm_apoATP_kinetic

# Steady-state resume dill
mkdir -p out/dnaa5_v1.5_hillKd_h4_K3_seed4/gen_dills
cp out/steady_state_inputs/dnaa5_v1.5_hillKd_h4_K3_seed4_gen5.dill \
   out/dnaa5_v1.5_hillKd_h4_K3_seed4/gen_dills/gen5.dill
```

### Using a different steady-state dill

If you'd rather start from your own burned-in state, edit the
`--resume-dill` flag in `scripts/run_milestone_dnaa5_stepped_adair.sh`
to point at a dill of your choice. The dill must come from a dnaa5-era
run so that the DnaA binding process state is populated; a dill from a
pre-Adair / pre-Hill-K_d checkpoint will not have the required state.

---

## 3. Run

```bash
bash scripts/run_milestone_dnaa5_stepped_adair.sh 1 4
```

Launches `seed=1` and `seed=4` in parallel. Each takes ~60-90 min.
Outputs land in:

- `out/dnaa5_stepped_adair_100_100_50_10_3_3_3_3_unlock60_sustain1_seed1_12gen_parquet/`
- `out/dnaa5_stepped_adair_100_100_50_10_3_3_3_3_unlock60_sustain1_seed4_12gen_parquet/`

Plus matching `gen_dills/` folders and `_run.log` files.

### The 19 env vars in the launch script

```
V2ECOLI_DNAA_ADAIR_KD=1                        # enable Adair binding math
V2ECOLI_DNAA_ADAIR_KDS_NM=100,100,50,10,3,3,3,3 # per-site stepped ladder
V2ECOLI_DNAA_ADAIR_KD_MAX_NM=100               # Langmuir fallback when gated
V2ECOLI_DNAA_ADAIR_KD_MIN_NM=3                 # floor for the ladder
V2ECOLI_DNAA_ADAPTIVE_KHALF=1                  # adaptive K_half (needs GRADIENT_GATE)
V2ECOLI_DNAA_COOP_GRADIENT_GATE=1              # instantaneous rising-bulk gate (parent + daughter)
V2ECOLI_DNAA_COOP_STUCK_GATE=0                 # disable stuck-time gate
V2ECOLI_DNAA_GRADIENT_GATE=1                   # populate rolling window
V2ECOLI_DNAA_GRADIENT_MIN_SLOPE_NM_PER_S=0.05  # threshold for "rising"
V2ECOLI_DNAA_GRADIENT_WINDOW_S=120             # 2 min smoothing window
V2ECOLI_DNAA_HILL_CONC=0                       # Hill-in-conc off
V2ECOLI_DNAA_HILL_KD=0                         # Hill-K_d off (switch to Adair)
V2ECOLI_DNAA_HYDROLYSIS_RATE_PER_MIN=0.025     # DnaA-ATP -> DnaA-ADP
V2ECOLI_DNAA_KHALF_STUCK_THRESHOLD_S=300       # 5 min stuck before adaptive K_half fires
V2ECOLI_DNAA_KINETIC_ORIC_LOW=0                # fast equilibrium (not kinetic)
V2ECOLI_DNAA_POST_INIT_UNLOCK_S=60             # daughter-only 60 s cumulative gate
V2ECOLI_DNAA_RELAX_SNAP=0                      # no snap on relax
V2ECOLI_SATURATION_SUSTAINED_S=1               # fire on 1 s sustained saturation
V2ECOLI_SATURATION_TRIGGERED_INIT=1            # replace mass-clock with sat-init gate
```

Notes on the gates:

- `GRADIENT_GATE=1` populates the bulk-DnaA-ATP rolling window used to
  compute `gradient_rising`. Without it, `gradient_rising` defaults to
  `True` and both `COOP_GRADIENT_GATE` and `ADAPTIVE_KHALF` become no-ops.
- `COOP_GRADIENT_GATE=1` applies to **all** domains (parent + daughter):
  every tick, if bulk isn't rising the Adair ladder collapses to Langmuir
  K_d_max. Parent has to have positive gradient at the moment it wants to
  cooperatively bind.
- `POST_INIT_UNLOCK_S=60` is a **daughter-only** cumulative gate: fresh
  domains that appear after the first tick start with the ladder locked
  (K_d clamped at K_d_max) and must accumulate 60 s of continuous positive
  bulk gradient before the Adair ladder unlocks. Any negative-gradient
  tick resets the counter.

Both gates check the same `gradient_rising` signal; parent uses it
instantaneously, daughter accumulates 60 s.

---

## 4. Verification — expected numbers

Aggregate targets from the reference PDF (seed=4 12-gen, clean single-init gens):

- τ mean ≈ 68.9 min
- oriC low bound DnaA-ATP saturates at 8/8 within a few seconds of firing
- Cell mass at initiation ≈ 700-800 fg (parent oriC)
- Bulk DnaA-ATP sawtooth: 10-30 nM
- Mostly one initiation per generation, with occasional reinitiations

Post-run analysis PDF is generated with `scripts/plot_5panel_bulkgate.py`.

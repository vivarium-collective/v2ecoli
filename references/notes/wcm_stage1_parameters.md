# WCM Stage 1 — Heuristic parameter values

**Source.** Companion parameter table to Fu/Xiao/Jun (2023). Stage 1 values are *heuristic* — anchored where literature exists (Sekimizu 1987, Kawakami 2006, Hansen & Atlung 2018), placeholder where structural detail is still being worked out (RIDA / DDAH / DARS rates are flagged "artificial setup").

**Growth condition.** ABT minimal glycerol, MG1655. Slow growth — non-overlapping cell cycle so synchronisation issues are avoided.

## Full table (verbatim from the PDF)

| # | Parameter | Stage 1 value | Source / note |
|---|---|---|---|
| 1 | cell birth volume | 1 µm³ | |
| 2 | doubling time | 150 min | ABT min-glycerol, non-overlapping cycle |
| 3 | C period | 70 min | |
| 4 | D period | 30 min | |
| 5 | replication speed | 66.31 kbp/min | = genome length / C period |
| 6 | dnaA transcription rate | 1.5 mRNA/min/gene | constitutive at this stage |
| 7 | DnaA translation efficiency | 1 protein/mRNA | Hansen & Atlung 2018 |
| 8 | DnaA degradation rate | 0 | fully stable |
| 9 | DnaA→ATP Kd | 30 nM | Sekimizu 1987, Kawakami 2006 |
| 10 | DnaA→ADP Kd | 100 nM | |
| 11 | DnaA-ATP intrinsic hydrolysis rate | 0.046 /min | Sekimizu 1987 |
| 12 | consensus DnaA-box Kd | 1 nM | ATP and ADP forms |
| 13 | oriC location | 3925744–3925989 | |
| 14 | oriC low-affinity box count | 8 | DnaA-ATP only |
| 15 | oriC low-affinity box Kd | 50 → 1 nM (cooperative) | decreasing with occupancy |
| 16 | Hda hydrolysis rate (RIDA) | 40 /min/fork pair | artificial |
| 17 | datA location | 4392732–4392914 | |
| 18 | datA DnaA-ATP hydrolysis rate (DDAH) | 12 /min/locus | artificial |
| 19 | DARS1 location | 813086–813186 | |
| 20 | DARS1 ADP→ATP conversion rate | 5 /min/locus | artificial |
| 21 | DARS2 location | 2969112–2969367 | |
| 22 | DARS2 ADP→ATP conversion rate | 10 /min/locus | artificial |
| 23 | SeqA sequestration time | 0 min | not considered |

## v2ecoli mapping status

Parameters that map cleanly to a surfaced composite parameter are **wired**. Parameters that require new wiring or a different data path are **TODO** — listed for follow-up. The `v2ecoli-stage1-heuristics` variant in `studies/dnaa-05-itv2-comparison/study.yaml` drives every wired parameter.

| # | Stage 1 parameter | v2ecoli parameter (composite key) | Status |
|---|---|---|---|
| 1 | cell birth volume = 1 µm³ | — (set by media + post-division mass listener) | not driven by composite param |
| 2 | doubling time = 150 min | — (set by media composition) | TODO: needs media-level override |
| 3 | C period = 70 min | — (emerges from `chromosome_replication.replication_speed`) | TODO: not surfaced |
| 4 | D period = 30 min | — (set by `cell_division_listener` config) | TODO: not surfaced |
| 5 | replication speed = 66.31 kbp/min | — | TODO |
| 6 | dnaA transcription rate = 1.5 mRNA/min/gene | — (ParCa-fitted) | needs ParCa override hook |
| 7 | DnaA translation efficiency = 1 | model_settings.translation_efficiency_eg10235_scale (dnaa-01) | wired (default == 1.0) |
| 8 | DnaA degradation rate = 0 | — (DnaA already non-degraded by default) | confirmed already |
| 9 | DnaA→ATP Kd = 30 nM | — (in `MONOMER0-160_RXN` equilibrium) | TODO: not surfaced |
| 10 | DnaA→ADP Kd = 100 nM | — (in `MONOMER0-4565_RXN`) | TODO: not surfaced |
| 11 | DnaA-ATP intrinsic hydrolysis rate = 0.046 /min | `hydrolysis_rate_per_min` | **wired** (default already 0.046) |
| 12 | consensus DnaA-box Kd = 1 nM | `kd_high_nM` | **wired** (default already 1.0) |
| 14 | oriC low-affinity box count = 8 | — (in box catalog data file) | catalog is 7 today; needs +1 entry |
| 15 | oriC low-affinity box Kd = 50→1 nM cooperative | `kd_low_nM` (single value, no cooperativity yet) | partial — set to 50; cooperativity is dnaa-03-EQ-02 |
| 16 | Hda (RIDA) hydrolysis rate = 40 /min/fork pair | `rida_rate_per_min` | **wired** (default 4.6; Stage 1 raises to 40) |
| 17–22 | datA + DARS1 + DARS2 (DDAH, DARS network) | — (no Steps yet) | TODO: dnaa-05/dnaa-06 deferred work |
| 23 | SeqA sequestration time = 0 | `refractory_seconds` | **wired** (default 600; Stage 1 sets to 0) |
| — | initial DnaA pool | `initial_dnaA_count_per_cell` | recommend 300 to anchor to literature band |

## What gets pushed into the variant

The `v2ecoli-stage1-heuristics` variant uses `v2ecoli.composites.baseline_recipes.dnaa_04_with_dnaa_initiation_trigger` (most-complete recipe) with these explicit overrides:

```yaml
hydrolysis_rate_per_min:  0.046    # row 11
rida_rate_per_min:        40.0     # row 16 (Hda)
atp_fraction_clamp_low:   null     # disable clamp — let RIDA + DDAH + DARS do the work
atp_fraction_clamp_high:  null
kd_high_nM:               1.0      # row 12
kd_low_nM:                50.0     # row 15 (highest of the cooperative range)
enable_oric_binding:      true
enable_dnaap_binding:     true
oric_high_threshold:      0.8      # 80 % of 8 oriC sites occupied → fire
refractory_seconds:       0.0      # row 23 (SeqA off at Stage 1)
initial_dnaA_count_per_cell: 300   # anchor to Schmidt 2016 / paper c_I
mechanism:                rida-v0  # so the v2ecoli RIDA Step is active
```

Rows 1–5 and 17–22 are unwired — they live behind v2ecoli internals (media composition, replication-speed constant in `chromosome_replication`, ParCa cache, and the not-yet-implemented DDAH/DARS Steps). They're listed in the `model_settings` block of `dnaa-05` as `gate: required-before-run` so a future iteration of the variant can pick them up once the wiring lands.

## Why this exists

The user's ask: *"set as many of our simulation variables as you can to these values."* That's what this variant does — every parameter the composite recipe currently surfaces is pinned. The unwired rows are catalogued here so the gap is visible (and so the dashboard's expert-input UI can show them as "awaiting future wiring" rather than silently dropping them).

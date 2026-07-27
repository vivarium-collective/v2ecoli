# SUMMARY — Hardening Plan A: root-cause v2ecoli #143 (O₂/CO₂ exchange deficit)

Branch: `harden/143-o2-rootcause` (based on origin/main). Commits are the record of progress.

## Verdict
**Not (ii) instrumentation. Primarily (i) real** — the low O₂ is genuine wcEcoli-FBA behavior —
**with the #143 −40 % O₂ v2-vs-v1 delta being an (iii)-flavored averaging-window fragility**, not
a localized respiratory defect. #143 is closable as "real, understood."

## How I got there (each step committed)
1. **Env fix (non-destructive).** This origin/main worktree needs `bigraph_schema.contract`
   (`ProcessContract`), absent from the shared venv's older bigraph-schema. Installed the locked
   commit (`4b208e13`, bigraph-schema 1.4.3) into `.deps/` and shadow it via
   `PYTHONPATH=.deps:.` — the shared venv (used by the driving session's canonical checkout) is
   untouched. Helper: `.run-env.sh`.
2. **Pipeline map + instrumentation ruled out** (`scripts/probe_o2_fba_vs_listener.py`): O₂/CO₂
   unconstrained in media; listener O₂ = net O₂ consumed by internal reactions (mass-balanced);
   report-card extraction symmetric v1↔v2.
3. **Mechanism** (`scripts/probe_o2_longrun.py`): O₂/CO₂ follow a **bimodal within-generation
   trajectory** — respiratory burst early (O₂ −1.3, CO₂ 0), then ~90 % non-respiring (O₂ −0.005,
   CO₂ +1.9). O₂ front-loaded, CO₂ back-loaded. Time-average = the low O₂:glucose ~0.09.
4. **The delta** = generation-length sensitivity: mean O₂ swings −36 % between a 1600- and
   2520-tick generation while mean CO₂ moves <1 %. v2's longer/stuck generations (#142) dilute
   its mean O₂ ~40 % below v1's → the headline. CO₂ (back-loaded) diverges less (−20 %).
5. Corroborated by the ketchup FBA-bridge (O₂ movable 3× with the cell viable → weakly-determined
   respiratory branch); reduced-cost probe (`scripts/probe_o2_degeneracy.py`).

## Deliverables
- `workspace/studies/showcase-6-equivalence-large/143-rootcause-findings.md` — full evidence trail.
- `.../143-evidence/o2co2_trajectory_seed0.txt` — the bimodal trace.
- `.../143-issue-comment.md` — paste-ready comment for issue #143.
- `showcase-6/study.yaml` — `rootcause_143` block.
- `ketchup-baseline-comparison/investigation.yaml` — open O₂:glucose decision **RESOLVED**
  (real FBA-regime behavior, not an exchange-reporting artifact).
- Probes under `scripts/probe_o2_*.py`.

## No code fix applied — by design
The listener and the report-card comparison are **correct**; there is no bug to patch. The
exchange-flux axis is real but *fragile* (front-loaded flux averaged over ensembles with
different generation-length distributions). Recommended (optional) hardening, not done here to
keep the exchange-fingerprint regression untouched: grade O₂/CO₂ over a matched averaging window
(or exclude duration-capped cells). A physiological O₂:glucose would require a modeling change
(kinetic O₂/terminal-oxidase target or pinned respiratory branch), out of scope for a root-cause.

## Notes for the driving session
- Run anything from this worktree with:
  `PYTHONPATH=~/code/v2e-h143/.deps:~/code/v2e-h143 ~/code/v2ecoli/.venv/bin/python ...`
- `out/cache` and `out/cache_full` are symlinks to the canonical ParCa caches (not rebuilt).
- `.deps/` is git-ignored (local dependency shadow).

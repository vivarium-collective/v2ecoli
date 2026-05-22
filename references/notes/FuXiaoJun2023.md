# FuXiaoJun2023 — Initiator-titration model v2

**Citation.** Fu, H., Xiao, F., Jun, S. (2023). *Bacterial Replication Initiation as Precision Control by Protein Counting*. PRX Life **1**, 013011. doi:[10.1103/PRXLife.1.013011](https://doi.org/10.1103/PRXLife.1.013011)

**PDF.** `references/papers/FuXiaoJun2023_PRXLife.pdf` (sha256 `9155b5a1113d2d7241fd590ac36830c97db3eb7b608db645fbcae779d66a40a0`).

## Big idea

A mean-field extension of the 1991 Hansen/Christensen/Atlung initiator-titration model. The cell triggers replication initiation by counting DnaA copies against ~300 chromosomal high-affinity DnaA boxes (titration step 1) followed by ~10 low-affinity *oriC* boxes (titration step 2). The two-step Poisson process compresses intrinsic noise from the standard 1/√N down to 1/N — explaining the observed precision of initiation timing in real cells.

Two questions answered:
1. **Why does *E. coli* maintain ~10× more DnaA than needed for one initiation?** Because the surplus titrates the chromosomal binding sites; only the residual after titration reaches *oriC*. Titration is the protein-counting device.
2. **Why does DnaA exist in active (DnaA-ATP) and inactive (DnaA-ADP) forms when only ATP is competent for initiation?** Because the replication-dependent DnaA-ATP → DnaA-ADP conversion by **RIDA** stabilises the cycle. Without RIDA, the protocell exhibits oscillatory initiation instability in fast-growth regimes.

## Model state (mean-field)

| Symbol | Meaning |
|---|---|
| `V(t)` | cell volume, exponential growth: `V = V₀ exp(λt)`, λ = ln2/τ |
| `I_T(t)` | DnaA-ATP total copy number |
| `I_D(t)` | DnaA-ADP total copy number |
| `B(t)` | total binding-site count (chromosomal + per-oriC) |
| `ρ_i(t)` | fractional progress of replication-fork generation *i* (0 = at ori, 1 = at ter) |
| `d` | number of active replication generations |

## Dynamics (Appendix D)

```
dI_T/dt = λ·(I_T + I_D) − ν·I_T               # balanced synthesis as DnaA-ATP, minus intrinsic ATPase
dI_D/dt =                   ν·I_T              # ADP accumulates from ATPase
dρ_i/dt = 1/C   for each ρ_i < 1               # forks travel ori → ter in time C
```

Plus the binding-site map `B[ρ]` from Eq. 6:
```
B(t) = N_B · [1 + Σ_{i=1..d} ρ_i · 2^(d-i)] + 2^d · n_B
```

## Initiation rule

When `I_T ≥ B` AND all `n_B` oriC sites are saturable by DnaA-ATP (assumption: only DnaA-ATP can bind oriC):
- A new replication generation prepends: `ρ → (0, ρ_1, ..., ρ_d)`, `d → d+1`
- Newly bound DnaA-ATP on the duplicated oriC boxes is displaced and hydrolysed by **RIDA** to DnaA-ADP (replication-dependent extra hydrolysis flux on active replisomes).

## Canonical parameter values (paper Fig. 5–6, *E. coli* wild-type)

| Symbol | Value | Source |
|---|---|---|
| `τ` (mass-doubling time) | 60 min in slow growth; 25 min in fast | Helmstetter–Cooper |
| `C` (replication duration) | 40 min | refs [37], [38] |
| `D` (term → division) | 20 min | (not used in this protocell variant) |
| `N_B` (chromosomal sites) | 300 | refs [15], [26], [27] |
| `n_B` (oriC sites) | 10 | refs [15]–[17], [22] |
| `c_I` ([DnaA] concentration) | 1 µM (≈ 600 nM total DnaA in 1 µm³ cell) | refs [33], [34] |
| `ν` (intrinsic ATPase rate) | 1/(15 min) ≈ 0.046 min⁻¹ | ref [13] |
| `K_D` (chromosomal box) | ≈ 1 nM | refs [15], [26], [45] |
| `K_D` (oriC weak box) | ≈ 100 nM | refs [15], [26], [46] |

## Predictions to test against v2ecoli

1. **Initiation mass** at slow growth (τ > C): `v_i = (1/c_I)·(N_B + n_B)`, independent of growth rate when `n ≡ ⌊C/τ⌋ = 0`. For fast growth (C/τ > 1), `v_i = (1/c_I)·(α·N_B + n_B)` with `α = 1/2ⁿ + (2 − (n+2)/2ⁿ)·τ/C`.
2. **Synchrony** between *ori* pairs: CV_int ~ 1/N (where N = total DnaA at initiation), much better than the 1/√N Poisson floor.
3. **Adder behaviour**: added cell size between consecutive initiations is independent of size at initiation.
4. **Δ4 mutant (no extrinsic conversion)** shows oscillatory initiation instability in fast-growth regimes — a direct test of whether RIDA is the stabiliser in v2ecoli's mechanism.

## Implementation in this workspace

- **Process**: `v2ecoli.processes.initiator_titration_v2.InitiatorTitrationV2`
- **Composite recipe**: `v2ecoli.composites.initiator_titration_v2.standalone` — single-cell run of the mean-field model, for side-by-side comparison with the v2ecoli DnaA mechanism.

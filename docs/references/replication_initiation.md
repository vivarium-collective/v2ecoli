# Replication Initiation — Molecular Reference

This file is a structured Markdown summary of
[`replication_initiation_molecular_info.pdf`](replication_initiation_molecular_info.pdf),
the curated reference for the next iteration of the chromosome-replication
model. The PDF is the canonical source — this file mirrors its facts in a
diff-able form so they can be cited by code, tests, and reports.

The Python module [`v2ecoli/data/replication_initiation/molecular_reference.py`](../../v2ecoli/data/replication_initiation/molecular_reference.py)
encodes these facts as importable constants, and
[`tests/test_replication_initiation_reference.py`](../../tests/test_replication_initiation_reference.py)
asserts that the module agrees with this document.

If a fact below changes, update the PDF, this file, the data module, and
re-run the reference tests in the same PR.

---

## oriC

- **Length:** 462 bp.
- **Two functional regions:**
  - **DUE** (duplex unwinding element) — site of local DNA unwinding for
    replisome loading.
  - **DOR** (DnaA-oligomerization region) — platform for DnaA binding and
    oligomerization that drives DUE opening.

### DnaA boxes at oriC (11 total)

| Box   | Class         | Sequence    | Affinity / nucleotide preference            |
|-------|---------------|-------------|---------------------------------------------|
| R1    | high-affinity | TTATCCACA   | Kd ≈ 1 nM. Binds both DnaA-ATP and DnaA-ADP.|
| R2    | high-affinity | TTATACACA   | Kd ≈ 1 nM (slightly lower than R1/R4). Both nucleotide forms.|
| R4    | high-affinity | TTATCCACA   | Kd ≈ 1 nM. Both nucleotide forms.           |
| R5M   | low-affinity  | (variant)   | Kd > 100 nM. Prefers DnaA-ATP; cooperative. |
| τ2    | low-affinity  | (variant)   | Kd > 100 nM. Prefers DnaA-ATP; cooperative. |
| I1    | low-affinity  | (variant)   | Kd > 100 nM. Prefers DnaA-ATP; cooperative. |
| I2    | low-affinity  | (variant)   | Kd > 100 nM. Prefers DnaA-ATP; cooperative. |
| I3    | low-affinity  | (variant)   | Kd > 100 nM. Prefers DnaA-ATP; cooperative. |
| C1    | low-affinity  | (variant)   | Kd > 100 nM. Prefers DnaA-ATP; cooperative. |
| C2    | low-affinity  | (variant)   | Kd > 100 nM. Prefers DnaA-ATP; cooperative. |
| C3    | low-affinity  | (variant)   | Kd > 100 nM. Prefers DnaA-ATP; cooperative. |

R1 and R4 are the highest-affinity consensus boxes. R2 differs by one base
position from the consensus.

### IHF-binding sites at oriC

| Site | Location relative to DnaA boxes              | Notes                                                |
|------|----------------------------------------------|------------------------------------------------------|
| IBS1 | Between R1 and R5M                           | Primary site. Bound when R1 is occupied by DnaA.     |
| IBS2 | Overlaps with R1                             | Secondary.                                           |

### Initiation mechanism

1. Pre-initiation: R1, R2, R4 occupied by DnaA (ATP **or** ADP); IHF on IBS1.
2. DnaA-ATP cooperatively assembles on the right half of the DOR in
   ordered occupancy: **C1 → I3 → C2 → C3**, anchored by R4-bound DnaA.
3. IHF-induced bending promotes a second DnaA-ATP filament on the left
   half of the DOR, anchored by R1-bound DnaA. R2-bound DnaA assists.
4. The DnaA oligomers generate torsional strain that locally unwinds the
   adjacent DUE.
5. The opened DUE provides the single-stranded substrate for DnaB
   helicase loading and replisome assembly.

### Post-initiation: SeqA sequestration

- Dam methyltransferase methylates adenine within **GATC** sites on both
  strands. After initiation, oriC is transiently **hemimethylated**.
- SeqA binds the >10 GATC sites in newly replicated oriC and forms
  multimers, sequestering the locus and preventing DnaA rebinding.
- Sequestration window: **~10 minutes** (~1/3 of a doubling time at rapid
  growth).

### Key references
- Katayama et al. 2017, *Frontiers in Microbiology* 8:2496.
- Kasho, Ozaki, Katayama 2023, *Int. J. Mol. Sci.* 24(14):11572.

---

## dnaA promoter

- **Length:** 448 bp.
- Two promoters separated by ~80 bp:
  - **p1** — basal transcription level.
  - **p2** — ≈ **3× stronger than p1**.

### DnaA boxes in the dnaA promoter region

| Box | Class                   | Notes                                                       |
|-----|-------------------------|-------------------------------------------------------------|
| 1   | high-affinity (consensus TTATCCACA) | Both DnaA-ATP and DnaA-ADP.                       |
| 2   | high-affinity (1 bp off consensus)  | Both DnaA-ATP and DnaA-ADP.                       |
| 3   | very low affinity                   | DnaA-ATP/ADP.                                     |
| 4   | DnaA-ATP-preferential               | Overlaps with **box a**.                          |
| a   | (overlaps box 4)                    | —                                                 |
| b   | DnaA-ATP-preferential, higher affinity than c | —                                       |
| c   | DnaA-ATP-preferential               | —                                                 |

DnaA binding to these sites yields complex autoregulation. The dominant
finding is repression of dnaA from its own promoter; one recent study
reports both positive and negative regulation at p2.

The region is GATC-rich; SeqA binds hemimethylated GATC sites after
replication and modulates dnaA expression for a short window, helping
buffer the transient gene-dosage increase from passage of the replisome.

### Key references
- Speck, Weigel, Messer 1999, *EMBO J.* 18(21):6169–6176.
- Saggioro, Olliver, Sclavi 2013, *Biochem. J.* 449(2):333–341.

---

## DnaA box consensus

- **Consensus:** `TTWTNCACA` (W = A|T; N = any).
- Highest-affinity variant: `TTATCCACA`, Kd ≈ 1 nM.
- Position-3 preference: **A > T**.
- Position-5 preference: **C > A ≥ G > T**.
- Effective affinity is context-dependent: cooperative binding raises the
  apparent affinity of clustered boxes (e.g. the mioC promoter has 5–8
  boxes, only one matching consensus).
- Relaxed motif from the datA / DARS1 / DARS2 bioinformatic study:
  `HHMTHCWVH` (H = A|C|T; M = A|C; W = A|T; V = A|C|G).

### Key references
- Schaper & Messer 1995, *J. Biol. Chem.* 270(29):17622–17626.
- Roth & Messer 1998, *Mol. Microbiol.* 28(2):395–401.
- Hansen et al. 2006, *J. Mol. Biol.* 355(1):85–95.
- Olivi et al. 2025, *Nat. Commun.* 16(1):7813.

---

## RIDA — Regulatory Inactivation of DnaA

- Triggered by the **DNA-loaded β-clamp** of the DNA polymerase III
  holoenzyme (DnaN / `dnaN` gene product).
- The clamp interacts with **ADP-bound Hda** (an AAA+ protein with an
  N-terminal clamp-binding motif) to form a clamp–Hda complex.
- The complex catalytically stimulates hydrolysis of DnaA-ATP → DnaA-ADP.
- Couples ongoing replication to DnaA inactivation, preventing
  reinitiation within the same cell cycle.

### Key references
- Katayama et al. 2017, *Frontiers in Microbiology* 8:2496.
- Riber et al. 2016, *Frontiers in Mol. Biosci.* 3:29.

---

## DDAH — datA-Dependent DnaA-ATP Hydrolysis

- Backup pathway to RIDA; depends on the **datA** locus.
- **datA position:** 94.7 min on the *E. coli* chromosome (near oriC).
- **datA length:** 363 bp.
- Minimal functional region contains **1 IBS** and **4 DnaA boxes**.
- Essential boxes: **2, 3, 7**. Box 4 is stimulatory.
- Mechanism: IHF binding induces a DNA architecture (likely loop
  formation) that catalytically promotes hydrolysis of DnaA-bound ATP.
- Deletion of datA causes untimely initiation in growing cells.

### Key references
- Katayama et al. 2017, *Frontiers in Microbiology* 8:2496.
- Hansen & Atlung 2018, *Frontiers in Microbiology* 9:319.

---

## DARS1 & DARS2 — DnaA-Reactivating Sequences

Both loci convert inactive **DnaA-ADP → apo-DnaA**, which then re-binds
ATP to regenerate active DnaA-ATP.

| Locus  | Length | Common core boxes | Additional features                                  |
|--------|--------|-------------------|------------------------------------------------------|
| DARS1  | 632 bp | I, II, III        | —                                                    |
| DARS2  | 737 bp | I, II, III        | Box IV, Box V, IBS1-2 (IHF), FBS1, FBS2-3 (Fis).     |

- **DARS2 is dominant in vivo.**
- DARS2 activity is regulated through cell-cycle-timed binding of IHF
  (IBS1-2) and Fis (FBS2-3).

### Key references
- Fujimitsu, Senriuchi, Katayama 2009, *Genes & Development* 23(10):1221–1233.
- Kasho et al. 2014, *Nucleic Acids Res.* 42(21):13134–13149.

---

## Status vs the current v2ecoli model

The current `v2ecoli/processes/chromosome_replication.py` initiates
replication on a **mass threshold** (`M_cell / n_oriC ≥ M_critical(τ)`)
and does **not** model the DnaA cycle, oriC binding state, RIDA, DDAH,
DARS1/2, or SeqA sequestration. DnaA boxes exist as structural objects
(`v2ecoli.types.schema_types.DNAA_BOX_ARRAY`) but their binding state
does not gate initiation. See the draft PR for the phased plan to close
this gap.

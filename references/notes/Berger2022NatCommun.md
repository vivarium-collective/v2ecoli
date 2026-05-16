# Berger2022NatCommun — Robust replication initiation from coupled homeostatic mechanisms

**Berger & ten Wolde, 2022, Nat Commun 13:6556.**
[https://www.nature.com/articles/s41467-022-33886-6](https://www.nature.com/articles/s41467-022-33886-6)

## Why it matters here

Mathematical / modeling paper analyzing how DnaA-driven initiation can be
robust under noise. Argues that pure titration vs DnaA-ATP-fraction sensing
are different homeostatic mechanisms that COMBINE to give the observed
robustness.

Directly relevant to dnaa-04: their predictions about initiation-mass
distribution and 1/N box-count scaling are the modeling counterpart to
our `reduced-box-count-shortens-inter-initiation` test.

## Key concepts to absorb

- Coupled homeostasis (titration + ATP-fraction)
- Initiation mass as a control variable
- Sensitivity to fluctuations in DnaA synthesis

## Behaviors / claims supported

- claim: `dnaa.initiator.titration-model`
- dnaa-04 `reduced-box-count-shortens-inter-initiation`, `initiation-mass-mean-matches-heuristic`

# Experimental claim datasets

Curated experimental response curves this investigation grades against. Each is
paired with its claim + provenance in `../../investigation.yaml` (`inputs.datasets`)
and cited via `../references/papers.bib`.

## To add

### `basan2015_acetate_vs_growth.csv` (placeholder — not yet extracted)

The aerobic acetate-overflow curve from Basan et al. 2015 (Nature 528:99–104):
acetate secretion rate vs growth rate, showing the linear rise above a critical
growth-rate threshold. Expected columns:

```
growth_rate_per_h, acetate_secretion_mmol_gDCW_h, sigma   # sigma optional (error bar)
```

Extract from the paper's main figure / supplement. The `curve_response` criterion
(see the overflow study) grades the model's swept-GUR acetate response against
these points: the onset growth rate + the overflow slope.

## Future perturbations and paired data bundles

Discrete carbon-source swaps and metabolic gene knockouts will each need a paired
data bundle (a ParCa condition cache / knockout override) alongside their
experimental claim. The dataset entry and the bundle reference are two halves of
one perturbation unit and should be added together so they stay coupled.

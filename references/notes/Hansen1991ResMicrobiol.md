# Hansen1991ResMicrobiol — Classic initiator-titration model

**Hansen, Christensen, Atlung, 1991, Res Microbiol 142(2-3):161-167.**
[https://www.sciencedirect.com/science/article/pii/0923250891900256](https://www.sciencedirect.com/science/article/pii/0923250891900256)

## Why it matters here

The original initiator-titration model paper. Predicts that initiation
timing is set by DnaA accumulation to a threshold that scales with the
number of DnaA-binding sites on the chromosome (titration). The 1/N
scaling prediction in dnaa-04's `reduced-box-count-shortens-inter-initiation`
test comes from this paper.

## Key predictions

- Inter-initiation time ∝ chromosomal DnaA-box count.
- Initiation mass per origin is approximately constant across growth rates.

## Behaviors / claims supported

- claim: `dnaa.initiator.titration-model`
- claim: `dnaa.titration.initiation-timing`
- dnaa-04 `reduced-box-count-shortens-inter-initiation` (1/N scaling test)
- dnaa-03 `chromosomal-occupancy-monotonic-increasing`

# v2ecoli Consensus Elongation Model


**Vision**: To resolve structural divergence between SteadyStateElongation and KineticTrnaCharging, the v2ecoli consensus elongation model will unify kinetic tRNA charging (from trna_charging_final) with ppGpp regulation (from main branch) into a single, mechanistically detailed translation model that captures translation bottlenecks, nutrient stress response, and metabolic integration.

---

## Core Features

### 1. **Individual tRNA Species** (86 tracked)

Adopting the species-level tracking from trna_charging_final, the consensus model will maintain each tRNA species separately with its own charged and uncharged states, replacing the main branch's amino-acid pooling approach. This integration will enable mechanistic modeling of aminoacyl-tRNA synthetase kinetics with species-specific catalytic rates (k_cat) and Michaelis constants (K_M) already parameterized in trna_charging_final. Rare codon penalties emerge from Michaelis-Menten saturation kinetics: when a particular tRNA species becomes depleted due to biased codon usage or high translation demand, the kinetic equations reduce elongation rate at codons requiring that tRNA; similarly, when amino acid substrate becomes scarce, saturation of the tRNA charging reaction itself limits aminoacylation capacity.

### 2. **Codon-Aware Translation**

Inheriting the codon-aware translation framework from trna_charging_final, the consensus model will translate codons (not amino acid sequences) with explicit tRNA-codon pairing that respects wobble pairing rules already implemented. Each codon's elongation rate will be kinetically determined by the availability of its cognate charged tRNA species. This will mean the same amino acid can have different translation speeds depending on which tRNA species is used. When a tRNA becomes scarce, ribosomes will stall at codons requiring that tRNA until charging replenishes the pool, leveraging the mechanistic rare-codon stalling capability from trna_charging_final.

### 3. **ppGpp Stringent Response**

Integrating the stringent response mechanism from the main branch, the consensus model will incorporate RelA detection of uncharged tRNA and ppGpp synthesis, along with SpoT-mediated degradation. ppGpp will inhibit transcription of rRNA and tRNA operons, reducing ribosomal biogenesis when the cell can't afford protein synthesis—a regulation already implemented in main. This will create the self-regulating feedback loop from main: low amino acids → slow tRNA charging → uncharged tRNA accumulation → ppGpp surge → fewer new ribosomes → reduced translation capacity → lower amino acid demand → pools recover → ppGpp degradation resumes.

### 4. **Amino Acid Dynamics with Feedback**

Bringing together synthesis, import, and export mechanisms from the main branch, the consensus model will capture amino acid dynamics through homeostatic FBA with ppGpp-modulated growth-rate-dependent targets (already in main), transporter-mediated import/export with Michaelis-Menten kinetics (already parameterized in main), and luxury catabolism at high pools. **Critically, elongation's kinetic constraints will feed back to metabolism** (via aa_count_diff from SteadyStateElongationModel) to dynamically adjust homeostatic targets based on the mismatch between what kinetics allow and what translation demands. This closed-loop coupling allows the model to capture how media richness sustains translation, how starvation triggers stringent response, and how the cell recovers from depletion.

### 5. **Integrated ODE System**

Unifying the tRNA charging ODE framework from trna_charging_final with the amino acid and ppGpp dynamics from main, the consensus model will couple kinetic equations for tRNA charging dynamics, amino acid pools, and ppGpp concentration simultaneously, all in physiological concentrations. The reconciliation framework from trna_charging_final will map the continuous ODE solution back to discrete codon positions: when the ODE predicts slow elongation (low tRNA availability), a binary search will find the ribosomal position matching that kinetic prediction, effectively "freezing" the ribosome until conditions improve. This unification will bridge the continuous kinetics of trna_charging_final with the metabolic integration of main.

---

## What It Will Unite

By integrating the species-level tRNA tracking and codon-aware kinetics from trna_charging_final with the ppGpp regulation, amino acid synthesis kinetics, and metabolic coupling from main, the consensus model will combine capabilities that were previously split: **rare codon penalties** and **ribosomal stalling** (from trna_charging_final), **ppGpp regulation** and **starvation recovery** (from main), with unified transport kinetics and metabolic-translation coupling. This convergence will achieve mechanistic detail of trna_charging_final while preserving the computational tractability and robust metabolic integration of the main branch.

---

## Key Implementation Details

**ParCa will integrate**: tRNA synthetase kinetics (k_cat, K_M) already parameterized in trna_charging_final, genome-wide codon frequencies from trna_charging_final with tRNA-codon mapping, transporter kinetics (Vmax, K_M) from main branch, ppGpp kinetics (RelA/SpoT rates) already in main, and growth-rate-dependent biomass targets from main's homeostatic FBA.

**Processes to unify**: polypeptide_elongation.py (merge trna_charging_final's ODE system with main's ppGpp coupling), metabolism.py (incorporate trna_charging_final's kinetic ODE alongside main's homeostatic FBA), enzyme_kinetics.py (add trna_charging_final's aaRS kinetics to main), amino_acid_transport.py (integrate main's import/export with trna_charging_final's ODE), ppgpp_regulation.py (adopt main's RelA/SpoT mechanism), transcription (integrate main's ppGpp feedback). **Integration note**: KineticTrnaChargingModel currently lacks the aa_count_diff feedback mechanism from SteadyStateElongationModel; this must be added to close the loop between elongation kinetics and metabolism's homeostatic targets.

**Validation plan**: Compare consensus (merged) vs. main (steady-state base) vs. trna_charging_final (kinetic base) vs. v1ecoli across 5 generations × 3 seeds in 4 media conditions. Success criteria: growth ±5–10%, tRNA charging ≥85%, ppGpp stringent response functional, rare codon penalties 20–70% slowdown emergent.

---

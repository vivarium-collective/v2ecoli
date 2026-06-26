# v2ecoli Consensus Elongation Model


**Vision**: To resolve structural divergence between SteadyStateElongation and KineticTrnaCharging, the v2ecoli consensus elongation model will unify kinetic tRNA charging (from trna_charging_final) with ppGpp regulation and amino acid synthesis, import and export kinetics(from main branch) into a single, mechanistically detailed translation model that captures translation bottlenecks, nutrient stress response, and metabolic integration.

---

## Core Features

### 1. **Individual tRNA Species** (86 tracked)

Adopting the species-level tracking from trna_charging_final, the consensus model will maintain each tRNA species separately with its own charged and uncharged states, replacing the main branch's amino-acid pooling approach. This integration will enable mechanistic modeling of aminoacyl-tRNA synthetase kinetics with species-specific catalytic rates (k_cat) and Michaelis constants (K_M) already parameterized in trna_charging_final. Rare codon penalties emerge from Michaelis-Menten saturation kinetics: when a particular tRNA species becomes depleted due to biased codon usage or high translation demand, the kinetic equations reduce elongation rate at codons requiring that tRNA; similarly, when amino acid substrate becomes scarce, saturation of the tRNA charging reaction itself limits aminoacylation capacity.

### 2. **Codon-Aware Translation**

Inheriting the codon-aware translation framework from trna_charging_final, the consensus model will translate codons (not amino acid sequences) with explicit tRNA-codon pairing that respects wobble pairing rules already implemented. Each codon's elongation rate will be kinetically determined by the availability of its cognate charged tRNA species. This will mean the same amino acid can have different translation speeds depending on which tRNA species is used. When a tRNA becomes scarce, ribosomes will stall at codons requiring that tRNA until charging replenishes the pool, leveraging the mechanistic rare-codon stalling capability from trna_charging_final.

### 3. **ppGpp Stringent Response**

Integrating the stringent response mechanism from the main branch, the consensus model will incorporate RelA detection of uncharged tRNA and ppGpp synthesis, along with SpoT-mediated degradation. ppGpp will inhibit transcription of rRNA and tRNA operons, reducing ribosomal biogenesis when the cell can't afford protein synthesis—a regulation already implemented in main. This will create the self-regulating feedback loop from main: low amino acids → slow tRNA charging → uncharged tRNA accumulation → ppGpp surge → fewer new ribosomes → reduced translation capacity → lower amino acid demand → pools recover → ppGpp degradation resumes.

### 4. **Amino Acid Dynamics with Feedback**

Bringing together synthesis, import, and export mechanisms from the main branch, the consensus model will capture amino acid dynamics through homeostatic FBA with ppGpp-modulated growth-rate-dependent targets (already in main), transporter-mediated import/export with Michaelis-Menten kinetics (already parameterized in main), and maintenance of excess pools via homeostatic targets. **Critically, elongation's kinetic constraints will feed back to metabolism** (via aa_count_diff from SteadyStateElongationModel) to dynamically adjust homeostatic targets based on the mismatch between what kinetics allow and what translation demands. This closed-loop coupling allows the model to capture how media richness sustains translation (via growth-rate-dependent targets) and how starvation triggers stringent response (via ppGpp synthesis in response to uncharged tRNA).

### 5. **Integrated ODE System**

Unifying the tRNA charging ODE framework from trna_charging_final with synthesis, import, and export kinetics from main, the consensus model will couple kinetic equations for tRNA charging dynamics, amino acid synthesis/import/export rates, and ppGpp simultaneously. **ODE design choice**: all ODE variables (free tRNAs, charged tRNAs, amino acids, ppGpp) will be tracked as **molecule counts**, not concentrations. This preserves KineticTrnaCharging's native unit system and makes the reconciliation framework (which requires discrete codon-reading events) work cleanly. K_M parameters are stored as concentrations (`__per_L`) but converted to counts at ODE startup via `K_M_counts = K_M_per_L × cell_volume`; concentrations are reconverted only at interfaces (metabolic coupling, output). **Critical implementation**: merge the ODE systems from SteadyStateElongationModel (calculate_steady_state_trna_charging) and KineticTrnaChargingModel (run_model), retaining KineticTrnaCharging's detailed codon-reading kinetics while integrating SteadyStateElongation's metabolic coupling and ppGpp feedback mechanisms. The reconciliation framework from trna_charging_final will map the continuous ODE solution back to discrete codon positions: when the ODE predicts slow elongation (low tRNA availability), a binary search will find the ribosomal position matching that kinetic prediction, effectively "freezing" the ribosome until conditions improve.

---

## What It Will Unite

By integrating the species-level tRNA tracking and codon-aware kinetics from trna_charging_final with the ppGpp regulation, amino acid synthesis kinetics, and metabolic coupling from main, the consensus model will combine capabilities that were previously split: **rare codon penalties** and **ribosomal stalling** (from trna_charging_final), **ppGpp regulation** and **starvation recovery** (from main), with unified transport kinetics and metabolic-translation coupling. This convergence will achieve mechanistic detail of trna_charging_final while preserving the computational tractability and robust metabolic integration of the main branch.

---

## Key Implementation Details

**ParCa will integrate**: tRNA synthetase kinetics (k_cat, K_M) already parameterized in trna_charging_final, genome-wide codon frequencies from trna_charging_final with tRNA-codon mapping, transporter kinetics (Vmax, K_M) from main branch, ppGpp kinetics (RelA/SpoT rates) already in main, and growth-rate-dependent biomass targets from main's homeostatic FBA.

**Processes to unify**: 
- **polypeptide_elongation.py**: Merge ODE systems from SteadyStateElongationModel.calculate_steady_state_trna_charging and KineticTrnaChargingModel.run_model. Use **molecule counts** as ODE variables (not concentrations) to preserve reconciliation framework compatibility. K_M parameters stored as `__per_L`, converted to counts at ODE startup. Retain KineticTrnaCharging's codon-reading kinetics while adding SteadyStateElongation's ppGpp coupling and aa_count_diff feedback mechanism. Convert back to concentrations only at metabolism interface.
- **metabolism.py**: Incorporate trna_charging_final's kinetic ODE alongside main's homeostatic FBA, using aa_count_diff to dynamically adjust homeostatic targets
- **enzyme_kinetics.py**: Add trna_charging_final's aaRS kinetics to main
- **amino_acid_transport.py**: Integrate main's import/export with trna_charging_final's ODE
- **ppgpp_regulation.py**: Adopt main's RelA/SpoT mechanism
- **transcription**: Integrate main's ppGpp feedback on rRNA/tRNA operons

**Critical integration notes**: 
- ODE merging is non-trivial; KineticTrnaChargingModel.run_model() produces codon-reading rates and reconciliation constraints that must drive SteadyStateElongation's coupling to metabolism
- KineticTrnaChargingModel currently lacks the aa_count_diff feedback mechanism from SteadyStateElongationModel; this must be added to close the loop between elongation kinetics and metabolism's homeostatic targets

**Validation plan**: Compare consensus (merged) vs. main (steady-state base) vs. trna_charging_final (kinetic base) vs. v1ecoli across 5 generations × 3 seeds in 4 media conditions. Success criteria: growth ±5–10%, tRNA charging ≥85%, ppGpp stringent response functional, rare codon penalties 20–70% slowdown emergent.

---

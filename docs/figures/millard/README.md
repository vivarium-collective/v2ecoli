# Millard kinetic-metabolism figures

Generated from real short runs of the built composites
(`./.venv/bin/python scripts/make_millard_figures.py`, `out/cache` present).

- **01_metabolism_swap_growth.png** — `baseline_millard`, 60 steps (1 s/step):
  cell_mass trajectory with central-carbon metabolism running as the Millard
  kinetic ODE (not FBA). Mass is FLAT at this length (1268.7→1268.8 fg,
  +0.02 fg / +0.002%) — growth is not resolvable over 60 s (cell cycle
  ~2700 s); the figure shows the WCM runs stably under the metabolism swap.
- **02_central_fluxes.png** — `baseline_millard`, 60 steps: four key Millard
  central fluxes (PTS_4 glucose uptake, CYTBO O2 respiration, PGI glycolysis,
  PYK pyruvate kinase). Fluxes are live and non-trivial (e.g. PTS_4 ramps
  0.070→0.229; PGI dips then recovers), confirming the kinetic ODE is active.
- **03_env_responsiveness.png** — step-level glucose sweep via
  `MillardPDMPMetabolism.update(external_concentrations={"GLCx": <mM>})`:
  glucose-uptake flux |PTS_4| vs external glucose, 0.05→20 mM. Textbook
  saturation (0.389→0.822 mM/s) — the kinetics respond to the environment.
- **04_reactor_o2_consumption.png** — `reactor_bird_coupled_millard`, 60 steps
  (cells_per_agent=1e11, gas_flow 2 L/min): reactor.dissolved_o2 vs
  o2_saturation (left) and agent O2 exchange counts/step (right). The Millard
  cell consumes reactor O2 end-to-end — DO drops 10.2→6.4 mg/L below the
  ~10.2 mg/L saturation while O2 exchange stays negative (uptake, growing to
  -2.7e7 counts/step).

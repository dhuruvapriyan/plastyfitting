# CICR Plasticity Fitting — Full Pipeline

## Overview

The goal is to fit parameters of the **CICR-extended synaptic plasticity model** (`GluSynapse_CICR.mod`) against experimental STDP ratios, using bluecellulab NEURON simulations as the source of calcium traces and JAX ODEs for fast parameter optimization.

Two repositories collaborate:

| Repo | Role |
|---|---|
| `plastyfire` | NEURON/bluecellulab simulations → calcium traces |
| `plastyfitting` | JAX ODE fitting → optimized CICR parameters |

---

## Stage 1 — Prepare the Circuit (`plastyfire`)

**What:** Build the SONATA circuit with custom per-synapse thresholds (θ_d, θ_p) and synapse properties (ρ₀, Use_d/p, gmax_d/p) baked into `dhuruva_modified_edges.h5`.

**Key file:** `/lustre06/project/6077694/dhuruva/plastyfire/data/dhuruva_circuit_config.json`
- Points to `dhuruva_modified_edges.h5` for all synapse properties (rho0, Use, gmax)
- θ_d / θ_p are injected separately at runtime by `simulator_edges.py` from the same file

---

## Stage 2 — Run STDP Induction Simulations (`plastyfire`)

**What:** For each pre→post pair and each STDP protocol (e.g. 10Hz_10ms, 10Hz_5ms, …), run a full NEURON pairing simulation and record spine calcium traces.

**Entry point:**
```bash
python submit_edges_sims.py --params chindemi --freq 10Hz --skip-existing --execution-mode slurm
```

**What it does internally (`plastyfire/simulator_edges.py → _run_prefire_edges_process`):**
1. Loads `prefire_simulation_config.json` + `prefire_prespikes.h5` from the workdir
2. Instantiates the post-synaptic cell + synapses via bluecellulab
3. Sets global GluSynapse HOC params (τ_effca, γ_d, γ_p) from the named preset
4. Injects θ_d / θ_p per synapse from `dhuruva_modified_edges.h5`
5. Sets per-synapse RNG seeds to match Neurodamus Random123 seeding
6. **Records per synapse:** `cai_CR`, `shaft_cai`, `ica_NMDA`, `ica_VDCC`, `rho_GB`
7. Runs CVODE simulation (with optional fast-forward)

**Outputs written to `bluecellulab_results/` inside each workdir:**

| File | Contents |
|---|---|
| `rho.h5` | SONATA-format initial/final ρ per synapse |
| `rho_timeseries.npy` | Full ρ(t) traces, shape `(T, 1+n_syn)` |
| `rho_global_ids.npy` | Global circuit IDs for each synapse column |
| `simulation_traces.pkl` | **JAX-ready calcium traces** (see format below) |

**`simulation_traces.pkl` format** (new dict-keyed format, matches `plastyfitting/_load_pkl()`):
```python
{
  "t":          float32 (T,)               # uniform 0.025 ms grid
  "cai_CR":     {global_id: float32 (T,)}  # spine calcium (mM)
  "shaft_cai":  {global_id: float32 (T,)}  # shaft calcium (mM)
  "ica_NMDA":   {global_id: float32 (T,)}  # NMDA Ca current (nA)
  "ica_VDCC":   {global_id: float32 (T,)}  # VDCC Ca current (nA)
  "rho_GB":     {global_id: float32 (T,)}  # synaptic weight trace
  "prespikes":  float32 (n_spikes,)        # pre-synaptic spike times (ms)
  "global_ids": list[int]
}
```

**Workdir structure** (one per pair × protocol):
```
refitting_results/fitting/n100/seed19091997/L5TTPC_L5TTPC_STDP/
  simulations/<pre_gid>-<post_gid>/<protocol>/
    prefire_simulation_config.json
    prefire_prespikes.h5
    bluecellulab_results/
      rho.h5
      rho_timeseries.npy
      rho_global_ids.npy
      simulation_traces.pkl   ← new
```

---

## Stage 3 — Validate Calcium Traces (`plastyfitting`)

**What:** Load `simulation_traces.pkl`, run the JAX CICR ODE forward with Chindemi baseline parameters, and verify that the simulated spine calcium and effcai match the NEURON traces.

**Key scripts:**
- `scripts/inspect/` — visualize traces, compare NEURON vs JAX calcium
- `scripts/validate/` — numerical validation of ODE fidelity

**Core logic (`plastyfitting/cicr_common.py`):**
- `_load_pkl()` reads the `.pkl` and returns arrays JAX can consume directly
- `compute_effcai_piecewise_linear_jax()` / `_jax_peak_effcai_zoh()` replicate the NEURON effcai ODE exactly via JAX `lax.scan`

---

## Stage 4 — Fit CICR Parameters (`plastyfitting`)

**What:** Use JAX + optax/scipy to minimize the loss between predicted EPSP ratios and experimental targets, by adjusting CICR model parameters.

**Entry point:**
```bash
python scripts/fit/get_cpre_cpost.py
```

**Experimental targets** (in `plastyfitting/cicr_common.py`):
```python
EXPERIMENTAL_TARGETS = {
    "10Hz_10ms": 1.2013,
    "10Hz_-10ms": 0.7312,
    ...
}
```

**What is being fit:**
- CICR model parameters: IP3R trigger amplitude, RyR amplification, SERCA pump rate, etc. (see `plastyfitting/models/cicr_er_ip3_simple.py`)
- The fitted parameters replace the baseline Chindemi values to better reproduce the STDP curves

**How EPSP ratio is predicted:**
1. Load calcium traces from `simulation_traces.pkl`
2. Compute peak effcai (C_pre, C_post) using the JAX ZOH ODE
3. Feed into the CICR model → predicted Δρ per synapse
4. Average Δρ → predicted EPSP ratio
5. Compare against `EXPERIMENTAL_TARGETS` → loss

---

## Data Paths

| Variable | Path |
|---|---|
| `BASE_DIR` | `/project/rrg-emuller/dhuruva/plastyfitting/` |
| `L5_TRACE_DIR` | `BASE_DIR/trace_results/CHINDEMI_PARAMS` |
| `THRESHOLD_TRACE_DIR` | `L5_TRACE_DIR/threshold_traces_out` |
| Circuit config | `/lustre06/project/6077694/dhuruva/plastyfire/data/dhuruva_circuit_config.json` |
| Edges file | `/lustre06/project/6077694/dhuruva/plastyfire/data/dhuruva_modified_edges.h5` |

---

## Quick Start

```bash
# 1. Run bluecellulab simulations (plastyfire repo)
cd /home/dhuruva/projects/ctb-emuller/dhuruva/plastyfire
python submit_edges_sims.py --params chindemi --freq 10Hz --skip-existing --execution-mode slurm

# 2. Install plastyfitting package
cd /home/dhuruva/projects/ctb-emuller/dhuruva/plastyfitting
pip install -e .

# 3. Validate JAX traces against NEURON output
python scripts/validate/<validation_script>.py

# 4. Fit CICR parameters
python scripts/fit/get_cpre_cpost.py
```

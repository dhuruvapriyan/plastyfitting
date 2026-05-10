# plastyfitting

JAX-accelerated parameter fitting for **CICR-extended calcium-based synaptic plasticity models**.

This repo fits the parameters of a Calcium-Induced Calcium Release (CICR) model that extends the [Chindemi et al. 2022](https://www.nature.com/articles/s41467-022-30214-w) Graupner-Brunel STDP synapse. It consumes calcium traces produced by [plastyfire](../plastyfire) (NEURON simulations) and optimizes CICR dynamics using fast JAX ODE integration.

## Repository Structure

```
plastyfitting/
├── plastyfitting/           # installable core package
│   ├── cicr_common.py           # base CICRModel class, JAX utilities, data loading
│   ├── cicr_common_weighted.py  # weighted NMDA/VDCC Ca variant
│   └── models/                  # model implementations
│       ├── cicr_er_ip3_simple.py    # canonical active CICR model (IP3R + RyR + SERCA)
│       ├── cicr_minimal*.py         # minimal CICR priming variants
│       ├── gb_only*.py              # GB baseline models (no CICR)
│       ├── gb_dual_effcai*.py       # dual NMDA/VDCC effcai models
│       ├── gb_vdcc_only*.py         # VDCC-only variants
│       ├── btsp_model.py            # BTSP phenomenological model
│       └── hong_ross_model.py       # Hong & Ross ER priming model
│
├── mod/                     # all NEURON .mod files
├── neuron_sims/             # NEURON runner/submission scripts
├── scripts/
│   ├── fit/                 # fitting pipeline entry points
│   ├── validate/            # NEURON vs JAX comparisons
│   ├── inspect/             # debugging and inspection tools
│   └── plot/                # visualization scripts
├── toy_model/               # standalone analytical scripts
├── biodata/                 # experimental STDP data (Chindemi 2022 Zenodo)
├── papers/                  # literature notes
└── archive/                 # stale/experimental scripts (kept for reference)
```

## Relationship to plastyfire

`plastyfire` runs expensive NEURON simulations to produce calcium trace `.pkl` files.
`plastyfitting` reads those traces and fits the CICR extension parameters using JAX.

## Installation

```bash
pip install -e .
```

## Usage

Run a fitting job:
```bash
python scripts/fit/get_cpre_cpost.py
```

Validate JAX against NEURON:
```bash
python scripts/validate/validate_jax_vs_neuron.py
```

#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

# Add the directory containing cicr_common to the path if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cicr_common import _load_pkl, compute_effcai_piecewise_linear_jax, BASE_DIR

def compute_and_compare_cpre_cpost(pair_name="180164-197248", protocol="10Hz_10ms", pathway="L5TTPC", params_file=None):
    if pathway == "L5TTPC":
        trace_dir = Path(BASE_DIR) / "trace_results/Chindemi_params"
    else:
        trace_dir = Path(BASE_DIR) / "trace_results/L23PC_Chindemi_params"
        
    pkl_path = trace_dir / pair_name / protocol / "simulation_traces.pkl"
    if not pkl_path.exists():
        print(f"Error: Could not find simulation output at {pkl_path}")
        return
        
    # Load the pickle data leveraging cicr_common
    data = _load_pkl(str(pkl_path))
    
    if "cai_pre" not in data or "cai_post" not in data:
        print(f"Error: Threshold traces not found for pair {pair_name}. Make sure they've been generated and saved via c_pre_finder.")
        return
        
    cai_pre = data["cai_pre"]
    t_pre = data["t_pre"]
    cai_post = data["cai_post"]
    t_post = data["t_post"]
    
    # These are the static Cpre and Cpost saved by the original NEURON simulation
    c_pre_mod = data["c_pre"]
    c_post_mod = data["c_post"]
    
    # Default tau_eff value from the model
    tau_eff = 278.318
    
    # If params file passed in, load tau_eff
    if params_file:
        import json
        if os.path.isfile(params_file):
            with open(params_file) as f: ep = json.load(f)
        else:
            ep = json.loads(params_file)
        if "best_parameters" in ep:
            ep = {k: (v["value"] if isinstance(v, dict) else v) for k, v in ep["best_parameters"].items()}
        if "tau_effca" in ep:
             tau_eff = float(ep["tau_effca"])
        elif "tau_eff" in ep:
             tau_eff = float(ep["tau_eff"])
        elif "tau_effca_GB_GluSynapse" in ep:
             tau_eff = float(ep["tau_effca_GB_GluSynapse"])
             
    n_syn = len(c_pre_mod)
    base_gid = 348477224
    
    # --- Test 1: JAX Baseline Cpre/Cpost (no CICR, pure VDCC/NMDA) ---
    print("\n--- Test 1: JAX Baseline Cpre/Cpost (CICR OFF) ---")
    print(f"Computing with params: {{'tau_effca_GB_GluSynapse': {tau_eff}}}")
    print(f"Presynaptic mapping of length {n_syn} created")
    
    cpre_dict_jax = {}
    cpost_dict_jax = {}
    
    for i in range(n_syn):
        effcai_pre = compute_effcai_piecewise_linear_jax(jnp.array(cai_pre[i]), jnp.array(t_pre), tau_effca=tau_eff)
        effcai_post = compute_effcai_piecewise_linear_jax(jnp.array(cai_post[i]), jnp.array(t_post), tau_effca=tau_eff)
        
        cpre_dict_jax[base_gid + i] = float(jnp.max(effcai_pre))
        cpost_dict_jax[base_gid + i] = float(jnp.max(effcai_post))

    print(f"-> JAX Baseline Cpre:  {cpre_dict_jax}")
    print(f"-> JAX Baseline Cpost: {cpost_dict_jax}")
    
    # --- Test 2: MOD-saved Cpre/Cpost (from NEURON simulation) ---
    print("\n--- Test 2: MOD-Saved Cpre/Cpost (from NEURON pkl) ---")
    
    cpre_dict_mod = {}
    cpost_dict_mod = {}
    
    for i in range(n_syn):
        cpre_dict_mod[base_gid + i] = float(c_pre_mod[i])
        cpost_dict_mod[base_gid + i] = float(c_post_mod[i])
        
    print(f"-> MOD Saved Cpre:  {cpre_dict_mod}")
    print(f"-> MOD Saved Cpost: {cpost_dict_mod}")
    
    # Note: In the new priming CICR model, Cpre/Cpost are computed from
    # baseline VDCC/NMDA traces only (no CICR during test pulses).
    # CICR effects are handled inside the main simulation scan_step.
    # Test 1 values ARE the correct Cpre/Cpost for the new model.
    
    return cpre_dict_jax, cpost_dict_jax

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute proper Cpre and Cpost dynamically using JAX.")
    parser.add_argument("--pair", type=str, default="180164-197248", help="Pair name (e.g. 180164-197248)")
    parser.add_argument("--protocol", type=str, default="10Hz_10ms", help="Protocol name (e.g. 10Hz_10ms)")
    parser.add_argument("--pathway", type=str, default="L5TTPC", choices=["L5TTPC", "L23PC"], help="Pathway (L5TTPC or L23PC)")
    parser.add_argument("--eval", type=str, default=None, help="JSON file or string of params to evaluate")
    
    args = parser.parse_args()
    compute_and_compare_cpre_cpost(args.pair, args.protocol, args.pathway, args.eval)

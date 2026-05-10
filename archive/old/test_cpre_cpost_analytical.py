#!/usr/bin/env python3
"""
Validation script to check Cpre and Cpost computed analytically from 
raw cai_CR traces vs the expected parameters.

User specified tau_effca_GB = 278.318
"""

import os
import glob
import pickle
import numpy as np
import argparse

def compute_effcai_piecewise_linear(cai_trace, t, tau_effca=278.318, min_ca=70e-6, effcai0=0.0):
    """
    Compute effective calcium using an exact piecewise-linear analytical integrator.
    Since NEURON uses CVODE (variable micro-steps), the simulated trace is a continuous
    integration. Our extracted `cai_trace` is sampled at discrete time steps.
    Integrating it using simple Euler introduces discretization error (~1%).
    Using exact piecewise-linear integration perfectly integrates the sampled points,
    yielding a drastically closer match to CVODE's internal result.
    """
    n_points = len(t)
    effcai = np.zeros(n_points)
    effcai[0] = effcai0
    
    for i in range(n_points - 1):
        dt = t[i + 1] - t[i]
        if dt <= 0:
            effcai[i + 1] = effcai[i]
            continue
            
        f0 = cai_trace[i] - min_ca
        f1 = cai_trace[i+1] - min_ca
        a = (f1 - f0) / dt
        
        decay = np.exp(-dt / tau_effca)
        
        # Exact integral of (f0 + a*s) * exp(-(dt-s)/tau) from 0 to dt
        term1 = f0 * tau_effca * (1 - decay)
        term2 = a * (tau_effca * dt - (tau_effca**2) * (1 - decay))
        
        effcai[i + 1] = effcai[i] * decay + term1 + term2
    
    return effcai

def validate_traces(traces_dir, tau_eff=278.318):
    pkl_files = glob.glob(os.path.join(traces_dir, "*_threshold_traces.pkl"))
    if not pkl_files:
        print(f"No threshold traces found in {traces_dir}")
        return

    print(f"Validating {len(pkl_files)} pairs with tau_effca = {tau_eff} ms...\n")
    
    all_cpre = []
    all_cpost = []

    for path in sorted(pkl_files):
        pair_name = os.path.basename(path).replace("_threshold_traces.pkl", "")
        with open(path, "rb") as f:
            data = pickle.load(f)
            
        pre_data = data["pre"]
        post_data = data["post"]
        
        pre_t = pre_data["t"]
        post_t = post_data["t"]
        
        # Calculate max effcai for each synapse in the pair
        cpre_syns = []
        cpost_syns = []
        
        for syn_idx in pre_data["cai_CR"].keys():
            cai_pre = pre_data["cai_CR"][syn_idx]
            effcai_pre = compute_effcai_piecewise_linear(cai_pre, pre_t, tau_effca=tau_eff)
            cpre_syns.append(np.max(effcai_pre))
            
            cai_post = post_data["cai_CR"][syn_idx]
            effcai_post = compute_effcai_piecewise_linear(cai_post, post_t, tau_effca=tau_eff)
            cpost_syns.append(np.max(effcai_post))
            
        mean_cpre = np.mean(cpre_syns)
        mean_cpost = np.mean(cpost_syns)
        
        all_cpre.extend(cpre_syns)
        all_cpost.extend(cpost_syns)
        
        sim_cpre_dict = pre_data.get("c_pre", {})
        sim_cpost_dict = post_data.get("c_post", {})
        
        # We only print the first few pairs to avoid spamming the console
        if len(all_cpre) <= 20: 
            print(f"Pair {pair_name}:")
            for i, syn_idx in enumerate(pre_data["cai_CR"].keys()):
                scpre = sim_cpre_dict.get(syn_idx, 0.0)
                scpost = sim_cpost_dict.get(syn_idx, 0.0)
                acpre = cpre_syns[i]
                acpost = cpost_syns[i]
                
                print(f"  Synapse {syn_idx}:")
                print(f"    Cpre:  Analytic = {acpre:.6f}, Sim = {scpre:.6f}, Diff = {abs(acpre - scpre):.6e}")
                print(f"    Cpost: Analytic = {acpost:.6f}, Sim = {scpost:.6f}, Diff = {abs(acpost - scpost):.6e}")
            print("-" * 40)

    print("\nGlobal Statistics:")
    print(f"Total synapses analyzed: {len(all_cpre)}")
    print(f"Global Mean Cpre:  {np.mean(all_cpre):.6f} ± {np.std(all_cpre):.6f}")
    print(f"Global Mean Cpost: {np.mean(all_cpost):.6f} ± {np.std(all_cpost):.6f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("traces_dir", help="Directory containing the extracted threshold traces")
    parser.add_argument("--tau", type=float, default=278.318, help="Tau effca value to use")
    
    args = parser.parse_args()
    validate_traces(args.traces_dir, args.tau)

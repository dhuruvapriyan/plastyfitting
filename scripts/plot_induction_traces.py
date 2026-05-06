#!/usr/bin/env python3
"""
Load simulation_traces.pkl from a workdir and plot cai_CR, effcai_GB, rho_GB.

Edit DEFAULT_PKL / DEFAULT_VERSION below and run:
    python tests/plot_induction_traces.py
"""

import sys
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_PKL     = "/home/dhuruva/projects/ctb-emuller/dhuruva/plastyfire/trace_results/DHURUVA_PARAMS_V17/180164-197248/10Hz_10ms/simulation_traces.pkl"
DEFAULT_VERSION = "v17"


def _build_theta_map(results):
    """Build {global_syn_id: (theta_d, theta_p)} from syn_props stored in the pkl.

    syn_props lists are ordered the same as syn_idx / cell.synapses iteration.
    We recover the global_syn_id <-> list-index correspondence by matching
    syn_props["Cpre"][i]  ==  c_pre[global_syn_id]  (exact float equality —
    both originate from the same recorder.max() call stored here).
    """
    syn_props = results.get("syn_props", {})
    c_pre_dict = results.get("c_pre", {})
    if (not syn_props or "theta_d_GB" not in syn_props
            or "theta_p_GB" not in syn_props or not c_pre_dict):
        return {}

    cpre_to_gid = {v: k for k, v in c_pre_dict.items()}
    theta_map = {}
    for i, cpre_val in enumerate(syn_props.get("Cpre", [])):
        gid = cpre_to_gid.get(cpre_val)
        if gid is not None:
            theta_map[gid] = (
                syn_props["theta_d_GB"][i],
                syn_props["theta_p_GB"][i],
            )
    return theta_map


def plot_traces(results, version, workdir, output_filename):
    t = results["t"]           # ms
    v = results["v"]
    prespikes  = np.asarray(results.get("prespikes",  []))
    postspikes = np.asarray(results.get("postspikes", []))

    panel_keys = [
        ("cai_CR",    r"$[Ca^{2+}]_{CR}$ (mM)"),
        ("effcai_GB", r"$\mathrm{effcai}_{GB}$ (mM)"),
        ("rho_GB",    r"$\rho_{GB}$"),
    ]

    # Collect syn_ids from whichever key is available first
    syn_ids = []
    for key, _ in panel_keys:
        if key in results and results[key]:
            syn_ids = sorted(results[key].keys())
            break

    theta_map = _build_theta_map(results)

    base, ext = os.path.splitext(output_filename)

    for syn_id in syn_ids:
        fig, axes = plt.subplots(1 + len(panel_keys), 1,
                                 figsize=(14, 3 * (1 + len(panel_keys))),
                                 sharex=True)

        td, tp = theta_map.get(syn_id, (None, None))
        theta_str = (f"  θ_d={td:.4f}  θ_p={tp:.4f}" if td is not None else "")
        fig.suptitle(f"{version}  |  {os.path.basename(workdir)}  |  syn {syn_id}{theta_str}", fontsize=11)

        # soma voltage
        axes[0].plot(t / 1e3, v, color="k", lw=0.8)
        axes[0].set_ylabel("Vm (mV)", fontsize=9)
        axes[0].spines["top"].set_visible(False)
        axes[0].spines["right"].set_visible(False)

        colors = ["#d04040", "#4070c0", "#40a840"]
        for ax, (key, ylabel), color in zip(axes[1:], panel_keys, colors):
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_ylabel(ylabel, fontsize=9)

            trace = results.get(key, {}).get(syn_id)
            if trace is None:
                ax.text(0.5, 0.5, "N/A", transform=ax.transAxes,
                        ha="center", va="center", color="grey", fontsize=8)
                continue

            arr = np.asarray(trace, dtype=float)
            if len(arr) != len(t):
                t_syn = np.linspace(t[0], t[-1], len(arr))
                arr = np.interp(t, t_syn, arr)

            ax.plot(t / 1e3, arr, color=color, lw=0.8)

            # overlay theta_d / theta_p on the effcai_GB panel
            if key == "effcai_GB" and td is not None:
                ax.axhline(td, color="orange",  ls="--", lw=1.2, label=f"θ_d = {td:.4f}")
                ax.axhline(tp, color="mediumpurple", ls="--", lw=1.2, label=f"θ_p = {tp:.4f}")
                ax.legend(fontsize=7, loc="upper right")

        axes[-1].set_xlabel("Time (s)", fontsize=9)
        axes[-1].set_xlim(t[0] / 1e3, 40)
        plt.tight_layout()

        out = f"{base}_syn{syn_id}{ext}"
        plt.savefig(out, dpi=150)
        print(f"Saved: {out}")
        plt.close(fig)


DEFAULT_WORKDIR = "/lustre06/project/6077694/dhuruva/plastyfire/refitting_results/fitting/n100/seed19091997/L5TTPC_L5TTPC_STDP/simulations/180164-197248/10Hz_5ms"
DEFAULT_PARAMS  = "v17"


def main():
    pkl_path = DEFAULT_PKL
    version  = DEFAULT_VERSION
    freq_dt  = os.path.basename(os.path.dirname(pkl_path))
    output   = f"induction_traces_{version}_{freq_dt}.png"

    print(f"Loading: {pkl_path}")
    with open(pkl_path, "rb") as f:
        results = pickle.load(f)

    # Print theta_d / theta_p per synapse
    theta_map = _build_theta_map(results)
    c_pre_dict  = results.get("c_pre",  {})
    c_post_dict = results.get("c_post", {})
    syn_props   = results.get("syn_props", {})
    locs = {}
    if "loc" in syn_props and "Cpre" in syn_props:
        cpre_to_gid = {v: k for k, v in c_pre_dict.items()}
        for i, cpre_val in enumerate(syn_props["Cpre"]):
            gid = cpre_to_gid.get(cpre_val)
            if gid is not None:
                locs[gid] = syn_props["loc"][i]

    print("=" * 70)
    print(f"{'syn_id':>12}  {'loc':>6}  {'cpre':>10}  {'cpost':>10}  {'theta_d':>10}  {'theta_p':>10}")
    print("-" * 70)
    for syn_id in sorted(theta_map.keys()):
        td, tp = theta_map[syn_id]
        cp  = c_pre_dict.get(syn_id, float("nan"))
        cpo = c_post_dict.get(syn_id, float("nan"))
        loc = locs.get(syn_id, "?")
        print(f"{syn_id:>12}  {loc:>6}  {cp:>10.6f}  {cpo:>10.6f}  {td:>10.6f}  {tp:>10.6f}")
    print("=" * 70)

    plot_traces(results, version, os.path.dirname(pkl_path), output)


if __name__ == "__main__":
    main()

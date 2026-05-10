#!/usr/bin/env python3
"""
JAX-model debug plotter for gb_vdcc_only.

Reproduces the EXACT computation from GBVdccOnlyModel in pure numpy:
  1. Load cai_VDCC_CR threshold traces → compute c_pre_vdcc / c_post_vdcc
     via peak_effcai_zoh  (identical to _jax_peak_effcai_zoh)
  2. Compute theta_d, theta_p (basal/apical split)
  3. Integrate effcai and rho over the full simulation trace
     (matches scan_step in gb_vdcc_only.py exactly)
  4. Plot one figure per synapse:
       - Soma voltage
       - cai_VDCC_CR
       - effcai  +  theta_d / theta_p lines
       - rho

Usage
-----
    python scripts/plot_vdcc_jax_debug.py \\
        --pkl trace_results/CHINDEMI_PARAMS/180164-197248/10Hz_10ms/simulation_traces.pkl \\
        --params best_params_gb_vdcc_only_20260307_104913.json \\
        --save --out-dir plots/debug/
"""

import argparse
import json
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Constants matching gb_vdcc_only.py ────────────────────────────────────────
_MIN_CA   = 70e-6   # mM   (resting calcium)
_TAU_IND  = 70.0    # s    (tau_ind_GB → denominator = 70 000 ms)
_RHO_STAR = 0.5

THRESHOLD_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "trace_results", "CHINDEMI_PARAMS", "threshold_traces_out",
)


# ══════════════════════════════════════════════════════════════════════════════
# Exact numpy replicas of the JAX functions
# ══════════════════════════════════════════════════════════════════════════════

def peak_effcai_zoh(cai_trace, tau, dt, min_ca=_MIN_CA):
    """
    Numpy replica of _jax_peak_effcai_zoh.

    ZOH: effcai[n+1] = effcai[n]*exp(-dt/tau) + max(ca-min_ca,0)*tau*(1-exp(-dt/tau))

    Returns the peak value over the trace.
    """
    decay  = np.exp(-dt / tau)
    factor = tau * (1.0 - decay)
    eff    = 0.0
    peak   = 0.0
    for ca in cai_trace[1:]:
        ca_ext = max(float(ca) - min_ca, 0.0)   # clamp ≥ 0 (same as jnp.maximum)
        eff    = eff * decay + ca_ext * factor
        if eff > peak:
            peak = eff
    return peak


def integrate_effcai_rho(cai_vdcc_trace, t_ms, theta_d, theta_p, params, rho0=0.0):
    """
    Numpy replica of scan_step in GBVdccOnlyModel.

    NOTE: the scan_step does NOT clamp (ca - min_ca) ≥ 0 — effcai can briefly
    go slightly negative between spikes. This matches the JAX code exactly.
    """
    tau     = params["tau_eff"]
    gamma_d = params["gamma_d"]
    gamma_p = params["gamma_p"]

    n      = len(cai_vdcc_trace)
    dt_arr = np.diff(t_ms, prepend=t_ms[0])
    dt_arr = np.where(dt_arr <= 0, 1e-6, dt_arr)   # matches JAX jnp.where

    effcai = np.zeros(n)
    rho    = np.zeros(n)
    dep    = np.zeros(n)
    pot    = np.zeros(n)
    rho[0] = rho0

    for i in range(1, n):
        dt     = float(dt_arr[i])
        ca     = float(cai_vdcc_trace[i])
        decay  = np.exp(-dt / tau)
        factor = tau * (1.0 - decay)

        eff_new    = effcai[i-1] * decay + (ca - _MIN_CA) * factor   # no clamp
        effcai[i]  = eff_new

        d = 1.0 if eff_new > theta_d else 0.0
        p = 1.0 if eff_new > theta_p else 0.0
        dep[i] = d
        pot[i] = p

        drho = (
            -rho[i-1] * (1.0 - rho[i-1]) * (_RHO_STAR - rho[i-1])
            + p * gamma_p * (1.0 - rho[i-1])
            - d * gamma_d * rho[i-1]
        ) / 70000.0                                                   # 70 s * 1000 ms/s
        rho[i] = np.clip(rho[i-1] + dt * drho, 0.0, 1.0)

    return effcai, rho, dep, pot


# ══════════════════════════════════════════════════════════════════════════════
# Threshold computation
# ══════════════════════════════════════════════════════════════════════════════

def load_threshold_pkl(pair_name, threshold_dir=THRESHOLD_DIR):
    path = os.path.join(threshold_dir, f"{pair_name}_threshold_traces.pkl")
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def compute_thresholds(thresh_data, gids, is_apical_map, params):
    """Compute per-synapse c_pre, c_post, theta_d, theta_p."""
    tau      = params["tau_eff"]
    pre_sec  = thresh_data["pre"]
    post_sec = thresh_data["post"]

    t_pre  = np.asarray(pre_sec["t_recorded"])   # ms
    t_post = np.asarray(post_sec["t_recorded"])  # ms
    dt_pre  = float(t_pre[1]  - t_pre[0])
    dt_post = float(t_post[1] - t_post[0])

    results = {}
    for gid in gids:
        apical = is_apical_map.get(gid, False)

        cai_pre  = np.asarray(pre_sec["cai_VDCC_CR"][gid])
        cai_post = np.asarray(post_sec["cai_VDCC_CR"][gid])

        c_pre  = peak_effcai_zoh(cai_pre,  tau, dt_pre)
        c_post = peak_effcai_zoh(cai_post, tau, dt_post)

        if apical:
            theta_d = params["d_v0a"] * c_pre + params["d_v1a"] * c_post
            theta_p = params["p_v0a"] * c_pre + params["p_v1a"] * c_post
        else:
            theta_d = params["d_v0"] * c_pre + params["d_v1"] * c_post
            theta_p = params["p_v0"] * c_pre + params["p_v1"] * c_post

        results[gid] = dict(c_pre=c_pre, c_post=c_post,
                            theta_d=theta_d, theta_p=theta_p, apical=apical)
    return results


def print_threshold_table(pair_name, proto, thresh_vals):
    print(f"\n  {'─'*88}")
    print(f"  {pair_name}  |  {proto}")
    print(f"  {'GID':>12}  {'loc':>6}  {'c_pre_vdcc':>14}  {'c_post_vdcc':>14}  "
          f"{'theta_d':>14}  {'theta_p':>14}  {'rho_final':>10}")
    print(f"  {'─'*88}")
    for gid, v in thresh_vals.items():
        loc = "apical" if v["apical"] else "basal"
        rf  = v.get("rho_final", float("nan"))
        print(f"  {gid:>12}  {loc:>6}  {v['c_pre']:>14.6f}  {v['c_post']:>14.6f}  "
              f"{v['theta_d']:>14.6f}  {v['theta_p']:>14.6f}  {rf:>10.4f}")
    print(f"  {'─'*88}")
    print(f"  (c_pre / c_post / theta in mM·ms)")


# ══════════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════════

def plot_synapse_debug(gid, t_ms, v, prespikes_ms, postspikes_ms,
                       cai_vdcc, effcai, rho, dep, pot,
                       thresh, params, pair_name, proto, save, out_dir):

    theta_d = thresh["theta_d"]
    theta_p = thresh["theta_p"]
    rho_f   = rho[-1]
    loc     = "apical" if thresh["apical"] else "basal"
    t_s     = t_ms / 1e3   # ms → s (same as plot_induction_traces.py)

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    title = (
        f"{pair_name}  |  {proto}  |  gid {gid} ({loc})\n"
        f"τ_eff={params['tau_eff']:.1f} ms   "
        f"γ_d={params['gamma_d']:.1f}   γ_p={params['gamma_p']:.1f}   "
        f"c_pre={thresh['c_pre']:.6f} mM·ms   "
        f"c_post={thresh['c_post']:.6f} mM·ms\n"
        f"θ_d={theta_d:.6f} mM·ms   θ_p={theta_p:.6f} mM·ms   "
        f"ρ_final={rho_f:.4f}"
    )
    fig.suptitle(title, fontsize=9, y=0.995)

    # ── 0: soma voltage ───────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(t_s, v, color="k", lw=0.6)
    ax.set_ylabel("Vm (mV)", fontsize=9)
    for sp in prespikes_ms / 1e3:
        ax.axvline(sp, color="tomato", lw=0.5, alpha=0.6)
    for sp in postspikes_ms / 1e3:
        ax.axvline(sp, color="forestgreen", lw=0.5, alpha=0.6, ls="--")

    # ── 1: cai_VDCC_CR ────────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(t_s, cai_vdcc * 1e6, color="steelblue", lw=0.6)
    ax.axhline(_MIN_CA * 1e6, color="gray", lw=0.6, ls=":",
               label=f"min_ca = {_MIN_CA*1e6:.0f} nM")
    ax.set_ylabel("cai_VDCC [nM]", fontsize=9)
    ax.legend(fontsize=7, loc="upper right")

    # ── 2: effcai + thresholds ────────────────────────────────────────────────
    ax = axes[2]
    ax.plot(t_s, effcai, color="darkorange", lw=0.7, label="effcai", zorder=3)
    ax.axhline(theta_d, color="crimson",      lw=1.2, ls="--",
               label=f"θ_d = {theta_d:.6f} mM·ms")
    ax.axhline(theta_p, color="mediumpurple", lw=1.2, ls="--",
               label=f"θ_p = {theta_p:.6f} mM·ms")
    ax.axhline(0.0, color="silver", lw=0.5, ls=":")
    # Background shading: dep and pot active (uses axes-fraction y so always full height)
    ax.fill_between(t_s, 0, 1,
                    where=dep > 0.5, alpha=0.10, color="crimson",
                    transform=ax.get_xaxis_transform(), label="dep active")
    ax.fill_between(t_s, 0, 1,
                    where=pot > 0.5, alpha=0.10, color="mediumpurple",
                    transform=ax.get_xaxis_transform(), label="pot active")
    ax.set_ylabel("effcai [mM·ms]", fontsize=9)
    ax.legend(fontsize=7, loc="upper right")

    # ── 3: rho ────────────────────────────────────────────────────────────────
    ax = axes[3]
    ax.plot(t_s, rho, color="#40a840", lw=0.8,
            label=f"ρ_GB  (final = {rho_f:.4f})")
    ax.axhline(0.5, color="gray", lw=0.7, ls="--", label="ρ = 0.5")
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("ρ_GB", fontsize=9)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.legend(fontsize=7, loc="upper right")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # x-limits: start just before first pre-spike, end just after last
    if len(prespikes_ms):
        t_start = max(t_s[0],  prespikes_ms[0]  / 1e3 - 2.0)
        t_end   = min(t_s[-1], prespikes_ms[-1] / 1e3 + 4.0)
        for ax in axes:
            ax.set_xlim(t_start, t_end)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save:
        fname = os.path.join(out_dir, f"vdcc_debug_{pair_name}_{proto}_gid{gid}.png")
        fig.savefig(fname, dpi=150)
        print(f"  Saved → {fname}")
        plt.close(fig)
    else:
        plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="JAX-model debug plotter for gb_vdcc_only")
    parser.add_argument("--pkl", required=True,
                        help="Path to simulation_traces.pkl")
    parser.add_argument("--params", required=True,
                        help="Path to best_params JSON file")
    parser.add_argument("--threshold-dir", default=THRESHOLD_DIR,
                        help="Directory containing *_threshold_traces.pkl files")
    parser.add_argument("--save", action="store_true",
                        help="Save PNGs instead of displaying")
    parser.add_argument("--out-dir", default=".", help="Output directory")
    args = parser.parse_args()

    # ── load params ───────────────────────────────────────────────────────────
    with open(args.params) as f:
        params = json.load(f)
    print(f"\nParams: {args.params}")
    for k, v in params.items():
        print(f"  {k:>12} = {v:.6f}")

    # ── load simulation pkl ───────────────────────────────────────────────────
    with open(args.pkl, "rb") as f:
        sim = pickle.load(f)

    t_ms = np.asarray(sim["t"], dtype=np.float64)
    # if t is in seconds, convert to ms
    if t_ms[-1] < 1000.0:
        t_ms = t_ms * 1e3

    v          = np.asarray(sim["v"], dtype=np.float64)
    prespikes  = np.asarray(sim.get("prespikes",  []), dtype=np.float64)
    postspikes = np.asarray(sim.get("postspikes", []), dtype=np.float64)
    # convert spike times to ms if needed
    if len(prespikes) and prespikes[-1] < 1000.0:
        prespikes  = prespikes  * 1e3
        postspikes = postspikes * 1e3

    cai_vdcc_dict = sim.get("cai_VDCC_CR", {})
    rho_gb_dict   = sim.get("rho_GB", {})
    syn_props     = sim.get("syn_props", {})
    gids          = list(cai_vdcc_dict.keys())

    # is_apical from syn_props["loc"]
    loc_list = syn_props.get("loc", [])
    is_apical_map = {
        gid: (loc_list[i] == "apical" if i < len(loc_list) else False)
        for i, gid in enumerate(gids)
    }

    # rho0 per synapse (first time-point of rho_GB trace)
    rho0_map = {
        gid: float(np.asarray(rho_gb_dict[gid])[0]) if gid in rho_gb_dict else 0.0
        for gid in gids
    }

    # ── derive pair name and protocol from path ─────────────────────────────
    pkl_path  = os.path.abspath(args.pkl)
    proto     = os.path.basename(os.path.dirname(pkl_path))
    pair_name = os.path.basename(os.path.dirname(os.path.dirname(pkl_path)))
    print(f"\nPair: {pair_name}  |  Protocol: {proto}")

    # ── load threshold pkl ────────────────────────────────────────────────────
    thresh_data = load_threshold_pkl(pair_name, args.threshold_dir)
    if thresh_data is None:
        raise FileNotFoundError(
            f"No threshold pkl found for '{pair_name}' in {args.threshold_dir}"
        )

    thresh_vals = compute_thresholds(thresh_data, gids, is_apical_map, params)

    if args.save:
        os.makedirs(args.out_dir, exist_ok=True)

    # ── per-synapse integration + plot ────────────────────────────────────────
    for gid in gids:
        cai_vdcc = np.asarray(cai_vdcc_dict[gid], dtype=np.float64)
        rho0     = rho0_map[gid]
        tv       = thresh_vals[gid]

        effcai, rho, dep, pot = integrate_effcai_rho(
            cai_vdcc, t_ms, tv["theta_d"], tv["theta_p"], params, rho0=rho0
        )
        tv["rho_final"] = float(rho[-1])

        plot_synapse_debug(
            gid, t_ms, v, prespikes, postspikes,
            cai_vdcc, effcai, rho, dep, pot,
            tv, params, pair_name, proto,
            args.save, args.out_dir,
        )

    # ── summary table ─────────────────────────────────────────────────────────
    print_threshold_table(pair_name, proto, thresh_vals)


if __name__ == "__main__":
    main()

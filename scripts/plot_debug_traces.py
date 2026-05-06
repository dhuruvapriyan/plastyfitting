#!/usr/bin/env python3
"""
Debug trace plotter for the Simplified IP3R+RyR CICR model.

Runs _debug_sim on selected protocols and plots all state variables
(cai_raw, IP3, Ca_ER, ca_cicr, h_ref, effcai, rho) as stacked subplots.

Usage:
  python scripts/plot_debug_traces.py --params best_params_simplified_ip3r_trigger_+_ryr_amplifier_20260301_211912.json
  python scripts/plot_debug_traces.py  # uses DEFAULT_PARAMS
"""

import os, sys, json, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from cicr_common import preload_all_data, _load_pkl, _load_basis
from cicr_er_ip3_simple import SimpleCICRModel, _debug_sim, CAI_REST
from plot_cicr_diagnostic import compute_single

# Diagnostic protocols: (freq_delay, description)
DIAG_PROTOCOLS = [
    ("10Hz_10ms",  "10Hz +10ms (pre→post, standard LTP)"),
    ("10Hz_-10ms", "10Hz −10ms (post→pre, standard LTD)"),
    ("2Hz_5ms",    "2Hz +5ms (low freq control)"),
    ("5Hz_5ms",    "5Hz +5ms (moderate freq)"),
]


def load_best_params(json_path, model):
    """Load params from JSON, falling back to model defaults."""
    dp = dict(model.DEFAULT_PARAMS)
    if json_path and os.path.isfile(json_path):
        with open(json_path) as f:
            data = json.load(f)
        if "best_parameters" in data:
            data = {k: (v["value"] if isinstance(v, dict) else v)
                    for k, v in data["best_parameters"].items()}
        dp.update({k: float(v) for k, v in data.items() if k in dp})
    return dp


def run_debug_traces(dp, protocol_data, pair_idx=0, syn_idx=0):
    """Run _debug_sim on each protocol, return dict of {proto: result_dict}."""
    results = {}
    for proto, desc in DIAG_PROTOCOLS:
        pairs = protocol_data.get(proto)
        if not pairs:
            print(f"  [SKIP] {proto}: no data loaded")
            continue
        pidx = min(pair_idx, len(pairs) - 1)
        n_syn = pairs[pidx]["cai"].shape[0]
        sidx = min(syn_idx, n_syn - 1)

        res = compute_single(pairs[pidx], sidx, dp, debug_sim_fn=_debug_sim)
        res["desc"] = desc
        res["pair_idx"] = pidx
        res["syn_idx"] = sidx
        results[proto] = res
        print(f"  [{proto}] pair={pidx} syn={sidx}  "
              f"ca_cicr_max={res['ca_cicr'].max():.4g}  "
              f"Ca_ER_max={res['ca_er'].max():.4g}  "
              f"effcai_ci_max={res['effcai_ci'].max():.4g}  "
              f"rho_final={res['rho_ci'][-1]:.4f}")
    return results


def plot_traces(results, output, dp):
    """Plot 7-row stacked subplots for each protocol."""
    protos = [p for p, _ in DIAG_PROTOCOLS if p in results]
    if not protos:
        print("No protocols to plot!"); return

    colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e']
    fig, axes = plt.subplots(7, 1, figsize=(14, 24), sharex=True)

    for i, proto in enumerate(protos):
        d = results[proto]
        t_s = d["t"] / 1000.0  # ms → s
        c = colors[i % len(colors)]
        label = proto

        # Row 0: cai_raw (input calcium from VDCC/NMDA)
        axes[0].plot(t_s, d["cai_raw"] * 1e3, color=c, alpha=0.8, label=label)

        # Row 1: IP3 concentration
        axes[1].plot(t_s, d["priming"], color=c, alpha=0.8, label=label)

        # Row 2: Ca_ER (ER luminal calcium)
        axes[2].plot(t_s, d["ca_er"], color=c, alpha=0.8, label=label)

        # Row 3: ca_cicr (cytosolic CICR release)
        axes[3].plot(t_s, d["ca_cicr"] * 1e3, color=c, alpha=0.8, label=label)

        # Row 4: h_ref (refractory inactivation gate)
        axes[4].plot(t_s, d["h_ref"], color=c, alpha=0.8, label=label)

        # Row 5: effcai (with and without CICR)
        axes[5].plot(t_s, d["effcai_ci"], color=c, alpha=0.8, label=f"{label} (w/ CICR)")
        axes[5].plot(t_s, d["effcai_no"], color=c, alpha=0.4, ls='--', label=f"{label} (no CICR)")

        # Row 6: rho (weight)
        axes[6].plot(t_s, d["rho_ci"], color=c, alpha=0.8, lw=1.5, label=f"{label} (w/ CICR)")
        axes[6].plot(t_s, d["rho_no"], color=c, alpha=0.4, ls='--', label=f"{label} (no CICR)")

    # Labels and formatting
    axes[0].set_ylabel(r'$Ca_{raw}$ (µM)')
    axes[0].set_title('Input Calcium (VDCC/NMDA)', fontsize=12)

    axes[1].set_ylabel('IP3 (mM)')
    axes[1].set_title('IP3 Concentration (mGluR trigger)', fontsize=12)

    axes[2].set_ylabel(r'$Ca_{ER}$ (mM)')
    axes[2].set_title('ER Luminal Calcium (priming state)', fontsize=12)

    axes[3].set_ylabel(r'$ca_{cicr}$ (µM)')
    axes[3].set_title('Cytosolic CICR Release', fontsize=12)

    axes[4].set_ylabel(r'$h_{ref}$')
    axes[4].set_ylim(-0.05, 1.05)
    axes[4].set_title('Refractory Inactivation Gate (h_ref)', fontsize=12)

    # Thresholds on effcai panel
    d0 = results[protos[0]]
    axes[5].axhline(d0["theta_d"], color='orange', ls=':', lw=1.5,
                    label=f'θd={d0["theta_d"]:.4f}')
    axes[5].axhline(d0["theta_p"], color='green', ls=':', lw=1.5,
                    label=f'θp={d0["theta_p"]:.4f}')
    axes[5].set_ylabel('effcai')
    axes[5].set_title('Effective Calcium Integrator', fontsize=12)

    axes[6].axhline(0.5, color='gray', ls=':', alpha=0.5)
    axes[6].set_ylim(-0.05, 1.05)
    axes[6].set_ylabel(r'$\rho$')
    axes[6].set_title('Synaptic Weight (Graupner-Brunel)', fontsize=12)
    axes[6].set_xlabel('Time (s)')

    # Auto x-limit: find last significant calcium activity
    t0 = d0["t"] / 1000.0
    cai0 = d0["cai_raw"]
    above = cai0 > cai0.max() * 0.01
    if np.any(above):
        xlim = min(t0[np.where(above)[0][-1]] * 1.5, t0[-1])
    else:
        xlim = t0[-1]

    # Check for explosions — switch to log scale if needed
    max_cicr = max(results[p]["ca_cicr"].max() for p in protos)
    max_er = max(results[p]["ca_er"].max() for p in protos)
    max_effcai = max(results[p]["effcai_ci"].max() for p in protos)

    if max_cicr > 1.0:  # > 1 mM is clearly exploding
        axes[3].set_yscale('symlog', linthresh=1e-3)
        axes[3].set_title('Cytosolic CICR Release [LOG — EXPLOSION DETECTED]',
                          fontsize=12, color='red')
    if max_er > 10.0:
        axes[2].set_yscale('symlog', linthresh=0.1)
        axes[2].set_title('ER Luminal Calcium [LOG — EXPLOSION DETECTED]',
                          fontsize=12, color='red')
    if max_effcai > 100.0:
        axes[5].set_yscale('symlog', linthresh=0.1)
        axes[5].set_title('Effective Calcium [LOG — EXPLOSION DETECTED]',
                          fontsize=12, color='red')

    for ax in axes:
        ax.set_xlim(0, xlim)
        ax.legend(loc='upper right', fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    # Parameter summary text
    param_text = (
        f"V_IP3R={dp['V_IP3R']:.2f}  V_RyR={dp['V_RyR']:.2f}  "
        f"V_SERCA={dp['V_SERCA']:.2f}  V_leak={dp['V_leak']:.2e}\n"
        f"δ_IP3={dp['delta_IP3']:.2f}  τ_IP3={dp['tau_IP3']:.0f}ms  "
        f"τ_ext={dp['tau_extrusion']:.0f}ms  τ_eff={dp['tau_eff']:.0f}ms\n"
        f"τ_ref={dp.get('tau_ref', 665):.0f}ms  K_h_ref={dp.get('K_h_ref', 0.0005):.4f}mM\n"
        f"γd={dp.get('gamma_d_GB_GluSynapse', 150):.1f}  "
        f"γp={dp.get('gamma_p_GB_GluSynapse', 200):.1f}  "
        f"Ca_ER_0={0.4:.1f}mM  K_RyR=0.5µM"
    )
    fig.text(0.02, 0.995, param_text, fontsize=8, family='monospace',
             va='top', ha='left', transform=fig.transFigure,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved → {output}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Debug trace plotter for SimpleCICR")
    parser.add_argument("--params", type=str, default=None,
                        help="Path to best params JSON file")
    parser.add_argument("--pair", type=int, default=0,
                        help="Pair index to plot (default: 0)")
    parser.add_argument("--syn", type=int, default=0,
                        help="Synapse index to plot (default: 0)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output PNG path (auto-generated if not given)")
    parser.add_argument("--max-pairs", type=int, default=5,
                        help="Max pairs to load per protocol")
    args = parser.parse_args()

    model = SimpleCICRModel()
    dp = load_best_params(args.params, model)

    protos = [p for p, _ in DIAG_PROTOCOLS]
    print(f"Loading data for protocols: {protos}")
    protocol_data = preload_all_data(max_pairs=args.max_pairs, protocols=protos)
    for p in protos:
        n = len(protocol_data.get(p, []))
        print(f"  {p}: {n} pairs loaded")

    print(f"\nRunning debug sims (pair={args.pair}, syn={args.syn}):")
    results = run_debug_traces(dp, protocol_data,
                               pair_idx=args.pair, syn_idx=args.syn)

    if args.output:
        out = args.output
    else:
        tag = "default" if not args.params else os.path.splitext(os.path.basename(args.params))[0]
        out = f"debug_traces_{tag}_p{args.pair}_s{args.syn}.png"

    plot_traces(results, out, dp)


if __name__ == "__main__":
    main()

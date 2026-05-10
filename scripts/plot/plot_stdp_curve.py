#!/usr/bin/env python3
"""
Plot STDP curves: model predictions (JAX forward pass) vs in vitro data.

Uses the same JAX forward pass (basis data + singleton means) that CMA-ES
optimized, so predictions match the log file exactly.

Usage:
  python scripts/plot_stdp_curve.py --params best_params_*.json
  python scripts/plot_stdp_curve.py --params best_params_*.json --freq 10 --max-pairs 50
"""

import os, sys, json, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from plastyfitting.cicr_common import preload_all_data, EXPERIMENTAL_TARGETS, EXPERIMENTAL_ERRORS
from plastyfitting.models.cicr_er_ip3_simple import SimpleCICRModel

# ── In vitro data (Markram et al. 1997, 10 Hz) ──────────────────────────
INVITRO_10HZ = {
    "dt":   [-10,    5,      10],
    "mean": [0.7922, 1.2038, 1.2013],
    "sem":  [0.0259, 0.0644, 0.0626],
}


def load_best_x(json_path, model):
    """Load best params JSON and return the x-vector in FIT_PARAMS order."""
    dp = dict(model.DEFAULT_PARAMS)
    if json_path and os.path.isfile(json_path):
        with open(json_path) as f:
            data = json.load(f)
        if "best_parameters" in data:
            data = {k: (v["value"] if isinstance(v, dict) else v)
                    for k, v in data["best_parameters"].items()}
        dp.update({k: float(v) for k, v in data.items() if k in dp})
    return np.array([dp[n] for n in model.PARAM_NAMES])


def parse_delay(proto):
    """Extract numeric delay from protocol string like '10Hz_-10ms' -> -10."""
    parts = proto.split("_")
    delay_str = parts[1].replace("ms", "")
    return int(delay_str)


def plot_stdp(proto_preds, freq, output, invitro=None):
    """Plot STDP curve: model predictions vs in vitro data."""
    plt.rcParams.update({
        'font.size': 14, 'axes.titlesize': 18, 'axes.labelsize': 16,
        'figure.dpi': 150, 'legend.fontsize': 12,
    })

    fig, ax = plt.subplots(figsize=(8, 6))

    # ── Model predictions ──
    delays = []
    means = []
    for proto, pred in sorted(proto_preds.items(), key=lambda x: parse_delay(x[0])):
        delays.append(parse_delay(proto))
        means.append(pred)

    delays = np.array(delays)
    means = np.array(means)

    ax.plot(delays, means, 'o-', color='#1f77b4', markersize=8, lw=2,
            label='in silico (CICR)', zorder=3)

    # ── In vitro data ──
    if invitro is not None:
        iv_dt = np.array(invitro["dt"])
        iv_mean = np.array(invitro["mean"])
        iv_sem = np.array(invitro["sem"])
        ax.errorbar(iv_dt, iv_mean, yerr=iv_sem, fmt='o-', color='#ff7f0e',
                    capsize=4, capthick=1.5, markersize=8, lw=2,
                    label='in vitro (Markram 1997)', zorder=3)

    # ── Reference lines ──
    ax.axhline(1.0, color='gray', ls='--', alpha=0.5, lw=1)
    ax.axvline(0.0, color='gray', ls='--', alpha=0.5, lw=1)

    ax.set_xlabel(r'$\Delta t$ (ms)')
    ax.set_ylabel('EPSP ratio')
    ax.set_title(f'Frequency = {freq} Hz (CMA-ES run 9)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(output, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"\nSaved -> {output}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot STDP curves from fitted CICR model")
    parser.add_argument("--params", type=str, required=True,
                        help="Path to best params JSON file")
    parser.add_argument("--freq", type=int, default=10,
                        help="Frequency to plot (default: 10)")
    parser.add_argument("--max-pairs", type=int, default=100,
                        help="Max pairs per protocol (default: 100)")
    parser.add_argument("--dt-step", type=int, default=10,
                        help="Trace downsampling step (default: 10)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output PNG path (auto-generated if not given)")
    args = parser.parse_args()

    model = SimpleCICRModel()
    x_best = load_best_x(args.params, model)

    # Discover all protocols for this frequency
    all_delays = ["-10ms", "5ms", "10ms"]
    protocols = [f"{args.freq}Hz_{d}" for d in all_delays]

    # Build targets dict -- use real targets if available, dummy otherwise
    targets = {}
    for p in protocols:
        targets[p] = EXPERIMENTAL_TARGETS.get(p, 1.0)  # dummy target for forward only

    print(f"Loading data for {args.freq}Hz protocols...")
    protocol_data = preload_all_data(max_pairs=args.max_pairs, protocols=protocols)
    for p in protocols:
        n = len(protocol_data.get(p, []))
        print(f"  {p}: {n} pairs loaded")

    # Setup JAX forward pass (same as CMA-ES used)
    print(f"Setting up JAX forward pass (dt_step={args.dt_step})...")
    model.setup_jax(protocol_data, targets, dt_step=args.dt_step)

    # Run forward prediction
    preds = np.array(model.forward_batch(jnp.array([x_best])))[0]

    proto_preds = {}
    print(f"\nPredictions:")
    for p, pred in zip(model.proto_names, preds):
        proto_preds[p] = float(pred)
        exp = EXPERIMENTAL_TARGETS.get(p, None)
        exp_str = f"  (exp: {exp:.4f})" if exp else ""
        print(f"  {p:<15s} = {pred:.4f}{exp_str}")

    if not proto_preds:
        print("No results to plot!")
        return

    invitro = INVITRO_10HZ if args.freq == 10 else None

    if args.output:
        out = args.output
    else:
        tag = os.path.splitext(os.path.basename(args.params))[0]
        out = f"stdp_curve_{args.freq}Hz_{tag}.png"

    plot_stdp(proto_preds, args.freq, out, invitro=invitro)


if __name__ == "__main__":
    main()

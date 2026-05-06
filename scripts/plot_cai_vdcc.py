#!/usr/bin/env python3
"""
Plot cai_VDCC_CR traces from simulation_traces.pkl files.

Usage
-----
# All pairs, both protocols (default):
    python scripts/plot_cai_vdcc.py

# Specific pair(s):
    python scripts/plot_cai_vdcc.py --pairs 180164-197248 180236-203656

# With params — also plots effcai and prints c_pre/c_post/theta_d/theta_p:
    python scripts/plot_cai_vdcc.py --params best_params_gb_vdcc_only_XXX.json --pairs 180164-197248

# Zoom to window around first pre-spike (ms):
    python scripts/plot_cai_vdcc.py --window 200

# Save instead of showing:
    python scripts/plot_cai_vdcc.py --save
"""

import argparse
import json
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TRACE_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "trace_results", "CHINDEMI_PARAMS",
)
THRESHOLD_DIR = os.path.join(TRACE_ROOT, "threshold_traces_out")
PROTOCOLS = ["10Hz_10ms", "10Hz_-10ms"]
_MIN_CA = 70e-6   # mM


# ── numpy replica of _jax_peak_effcai_zoh ────────────────────────────────────

def peak_effcai_zoh(cai_trace, tau, dt, min_ca=_MIN_CA):
    """ZOH-exact discrete peak of the effcai integrator (numpy)."""
    decay  = np.exp(-dt / tau)
    factor = tau * (1.0 - decay)
    eff    = 0.0
    peak   = 0.0
    for ca in cai_trace[1:]:
        ca_ext = max(float(ca) - min_ca, 0.0)
        eff    = eff * decay + ca_ext * factor
        if eff > peak:
            peak = eff
    return peak


# ── threshold trace loading ───────────────────────────────────────────────────

def load_threshold_pkl(pair_name):
    path = os.path.join(THRESHOLD_DIR, f"{pair_name}_threshold_traces.pkl")
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def _get_cai_vdcc(thresh_data, side):
    """Return (cai_vdcc_dict, t_array) from a threshold pkl for 'pre' or 'post'."""
    sect = thresh_data.get(side, thresh_data)   # fall back to flat if no sub-key
    cai  = sect.get("cai_VDCC_CR", {})
    # prefer t_recorded (matches cai length), fall back to t
    t_rec = sect.get("t_recorded", None)
    t_raw = sect.get("t", None)
    if t_rec is not None:
        t = np.asarray(t_rec, dtype=np.float64)
    elif t_raw is not None:
        t = np.asarray(t_raw, dtype=np.float64)
    else:
        t = None
    return cai, t


def compute_thresholds(pair_name, params, gids, is_apical_map):
    """
    Compute and return per-synapse dict with:
        c_pre_vdcc, c_post_vdcc, theta_d, theta_p
    """
    tau = params["tau_eff"]
    thresh = load_threshold_pkl(pair_name)
    if thresh is None:
        return None

    pre_cai,  t_pre  = _get_cai_vdcc(thresh, "pre")
    post_cai, t_post = _get_cai_vdcc(thresh, "post")

    # dt from the time axis (ms — NEURON uses seconds, convert if needed)
    def _dt(t_arr, cai_dict):
        if t_arr is None or len(t_arr) < 2:
            return 0.025   # fallback 0.025 ms
        dt = float(t_arr[1] - t_arr[0])
        # if time is in seconds (values < 1), convert to ms
        if dt < 0.001:
            dt *= 1e3
        return dt

    dt_pre  = _dt(t_pre,  pre_cai)
    dt_post = _dt(t_post, post_cai)

    results = {}
    for gid in gids:
        apical = is_apical_map.get(gid, False)

        cai_pre_arr  = np.asarray(pre_cai.get(gid,  list(pre_cai.values())[0]),  dtype=np.float64)
        cai_post_arr = np.asarray(post_cai.get(gid, list(post_cai.values())[0]), dtype=np.float64)

        c_pre  = peak_effcai_zoh(cai_pre_arr,  tau, dt_pre)
        c_post = peak_effcai_zoh(cai_post_arr, tau, dt_post)

        if apical:
            theta_d = params["d_v0a"] * c_pre + params["d_v1a"] * c_post
            theta_p = params["p_v0a"] * c_pre + params["p_v1a"] * c_post
        else:
            theta_d = params["d_v0"] * c_pre + params["d_v1"] * c_post
            theta_p = params["p_v0"] * c_pre + params["p_v1"] * c_post

        results[gid] = dict(c_pre_vdcc=c_pre, c_post_vdcc=c_post,
                            theta_d=theta_d, theta_p=theta_p, apical=apical)
    return results


def print_thresholds(pair_name, proto, thresh_vals):
    print(f"\n  {'─'*70}")
    print(f"  Pair {pair_name}  |  {proto}")
    print(f"  {'GID':>12}  {'loc':>6}  {'c_pre_vdcc':>14}  {'c_post_vdcc':>14}  {'theta_d':>14}  {'theta_p':>14}")
    print(f"  {'─'*70}")
    for gid, v in thresh_vals.items():
        loc = "apical" if v["apical"] else "basal"
        print(f"  {gid:>12}  {loc:>6}  {v['c_pre_vdcc']:>14.6f}  "
              f"{v['c_post_vdcc']:>14.6f}  "
              f"{v['theta_d']:>14.6f}  "
              f"{v['theta_p']:>14.6f}")
    print(f"  {'─'*70}")
    print(f"  (all values in mM·ms)")


def load_pkl(pair_dir, protocol):
    pkl = os.path.join(pair_dir, protocol, "simulation_traces.pkl")
    if not os.path.isfile(pkl):
        return None
    with open(pkl, "rb") as f:
        return pickle.load(f)


def compute_effcai_trace(cai_vdcc_trace, t_ms, params):
    """Compute the effcai integrator trace (numpy) for one synapse."""
    tau    = params["tau_eff"]
    dt_arr = np.diff(t_ms, prepend=t_ms[0])
    dt_arr = np.where(dt_arr <= 0, 1e-6, dt_arr)
    eff    = np.zeros(len(cai_vdcc_trace))
    for i in range(1, len(cai_vdcc_trace)):
        dt     = dt_arr[i]
        decay  = np.exp(-dt / tau)
        factor = tau * (1.0 - decay)
        ca_ext = max(float(cai_vdcc_trace[i]) - _MIN_CA, 0.0)
        eff[i] = eff[i-1] * decay + ca_ext * factor
    return eff


def plot_pair(pair_name, data_by_proto, window_ms, save, out_dir, params=None, thresh_by_proto=None):
    n_proto   = len(data_by_proto)
    proto_list = list(data_by_proto.keys())
    max_syn   = max(len(d["cai_VDCC_CR"]) for d in data_by_proto.values())

    # 2 rows per synapse when params supplied (cai_VDCC row + effcai row)
    n_rows_per_syn = 2 if params is not None else 1
    n_rows = max_syn * n_rows_per_syn

    fig, axes = plt.subplots(
        n_rows, n_proto,
        figsize=(6 * n_proto, 2.5 * n_rows),
        sharex="col", sharey=False,
        squeeze=False,
    )
    title = f"cai_VDCC_CR  |  pair {pair_name}"
    if params:
        title += f"  |  tau_eff={params['tau_eff']:.1f} ms"
    fig.suptitle(title, fontsize=12, fontweight="bold")

    for col, proto in enumerate(proto_list):
        d    = data_by_proto[proto]
        t    = np.asarray(d["t"])
        # convert s → ms if needed
        if t[-1] < 1000:
            t = t * 1e3
        gids = list(d["cai_VDCC_CR"].keys())

        # x-limits
        if window_ms is not None and len(d["prespikes"]) > 0:
            t0 = float(np.asarray(d["prespikes"])[0])
            if t0 < 1000:
                t0 *= 1e3
            xlim = (t0 - 20, t0 + window_ms)
        else:
            xlim = (t[0], t[-1])

        prespikes  = np.asarray(d["prespikes"])
        postspikes = np.asarray(d["postspikes"])
        if prespikes[-1] < 1000:
            prespikes  = prespikes  * 1e3
            postspikes = postspikes * 1e3

        thresh_map = (thresh_by_proto or {}).get(proto, {})

        for syn_idx, gid in enumerate(gids):
            cai_row = syn_idx * n_rows_per_syn
            ax_cai  = axes[cai_row, col]
            cai     = np.asarray(d["cai_VDCC_CR"][gid])

            ax_cai.plot(t, cai * 1e6, color="steelblue", lw=0.8, label="cai_VDCC_CR")
            for sp in prespikes:
                if xlim[0] <= sp <= xlim[1]:
                    ax_cai.axvline(sp, color="tomato", lw=0.6, alpha=0.7,
                                   label="pre" if sp == prespikes[0] else "")
            for sp in postspikes:
                if xlim[0] <= sp <= xlim[1]:
                    ax_cai.axvline(sp, color="forestgreen", lw=0.6, alpha=0.7,
                                   ls="--", label="post" if sp == postspikes[0] else "")
            ax_cai.set_xlim(xlim)
            ax_cai.set_ylabel("cai_VDCC [nM]", fontsize=8)
            if syn_idx == 0:
                ax_cai.set_title(proto, fontsize=10)
                ax_cai.legend(fontsize=7, loc="upper right")
            ax_cai.annotate(f"gid {gid}", xy=(0.02, 0.90), xycoords="axes fraction",
                            fontsize=7, color="gray")

            # ── effcai subplot ──────────────────────────────────────────────
            if params is not None:
                ax_eff = axes[cai_row + 1, col]
                effcai = compute_effcai_trace(cai, t, params)
                ax_eff.plot(t, effcai, color="darkorange", lw=0.9, label="effcai")

                tv = thresh_map.get(gid, {})
                if tv:
                    for name, val, color in [("θ_d", tv["theta_d"], "crimson"),
                                             ("θ_p", tv["theta_p"], "purple")]:
                        ax_eff.axhline(val, color=color, lw=1.0, ls="--",
                                       label=f"{name}={val:.6f} mM·ms")
                ax_eff.set_xlim(xlim)
                ax_eff.set_ylabel("effcai [mM·ms]", fontsize=8)
                ax_eff.legend(fontsize=7, loc="upper right")

        # hide unused rows
        for syn_idx in range(len(gids), max_syn):
            for r in range(n_rows_per_syn):
                axes[syn_idx * n_rows_per_syn + r, col].set_visible(False)

        axes[n_rows - 1, col].set_xlabel("Time (ms)", fontsize=9)

    plt.tight_layout()

    if save:
        fname = os.path.join(out_dir, f"cai_vdcc_{pair_name}.png")
        fig.savefig(fname, dpi=150)
        print(f"Saved → {fname}")
        plt.close(fig)
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot cai_VDCC_CR traces")
    parser.add_argument("--trace-root", default=TRACE_ROOT,
                        help="Root directory containing pair folders")
    parser.add_argument("--pairs", nargs="+", default=None,
                        help="Pair folder names to plot (default: all)")
    parser.add_argument("--protocol", choices=PROTOCOLS + ["both"], default="both",
                        help="Protocol to plot (default: both)")
    parser.add_argument("--window", type=float, default=None,
                        help="x-axis window in ms after first pre-spike")
    parser.add_argument("--save", action="store_true",
                        help="Save figures to PNG instead of displaying")
    parser.add_argument("--out-dir", default=".",
                        help="Output directory for saved figures")
    parser.add_argument("--max-pairs", type=int, default=None,
                        help="Limit number of pairs plotted")
    parser.add_argument("--params", default=None,
                        help="JSON param file — enables effcai subplot + threshold printing")
    args = parser.parse_args()

    protos = PROTOCOLS if args.protocol == "both" else [args.protocol]

    # Load params if given
    params = None
    if args.params:
        with open(args.params) as fp:
            params = json.load(fp)
        print(f"\nParams loaded from {args.params}")
        print(f"  tau_eff = {params['tau_eff']:.3f} ms")

    # Collect pairs
    if args.pairs:
        pair_list = args.pairs
    else:
        pair_list = sorted(
            p for p in os.listdir(args.trace_root)
            if os.path.isdir(os.path.join(args.trace_root, p))
        )

    if args.max_pairs:
        pair_list = pair_list[:args.max_pairs]

    if args.save:
        os.makedirs(args.out_dir, exist_ok=True)

    for pair_name in pair_list:
        pair_dir = os.path.join(args.trace_root, pair_name)
        data_by_proto = {}
        for proto in protos:
            d = load_pkl(pair_dir, proto)
            if d is not None:
                data_by_proto[proto] = d
            else:
                print(f"  [skip] {pair_name}/{proto} — pkl not found")

        if not data_by_proto:
            continue

        # ── compute thresholds (if params given) ──────────────────────────
        thresh_by_proto = {}
        if params is not None:
            # Determine is_apical from the first available protocol's syn_props
            first_d = next(iter(data_by_proto.values()))
            syn_props = first_d.get("syn_props", {})
            loc_list  = syn_props.get("loc", [])
            first_gids = list(first_d["cai_VDCC_CR"].keys())
            is_apical_map = {
                gid: (loc_list[i] == "apical" if i < len(loc_list) else False)
                for i, gid in enumerate(first_gids)
            }

            tv = compute_thresholds(pair_name, params, first_gids, is_apical_map)
            if tv is not None:
                for proto in data_by_proto:
                    thresh_by_proto[proto] = tv
                for proto in data_by_proto:
                    print_thresholds(pair_name, proto, tv)
            else:
                print(f"  [warn] No threshold pkl found for {pair_name}")

        print(f"Plotting {pair_name} ...")
        plot_pair(pair_name, data_by_proto, args.window, args.save, args.out_dir,
                  params=params, thresh_by_proto=thresh_by_proto)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Plot cai_CR threshold traces (cpre / cpost) for one pair.

Usage:
    python scripts/plot_cai_cr_threshold.py                          # first pair found
    python scripts/plot_cai_cr_threshold.py --pair 180164-197248     # specific pair
    python scripts/plot_cai_cr_threshold.py --tau 278.318            # custom tau_eff
    python scripts/plot_cai_cr_threshold.py --out my_plot.png        # custom output path
"""

import argparse
import pickle
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

THRESHOLD_TRACE_DIR = Path(
    "/project/rrg-emuller/dhuruva/plastyfitting/trace_results/"
    "CHINDEMI_PARAMS/threshold_traces_out"
)
TAU_EFF = 278.318   # ms — same value used in cicr_common / gb_only

def peak_effcai(cai_arr, dt_arr, tau):
    """Pure-numpy effcai integrator, returns full trace and peak."""
    effcai = 0.0
    trace = np.empty(len(cai_arr))
    trace[0] = 0.0
    for i, (dt, ca) in enumerate(zip(dt_arr, cai_arr[1:]), start=1):
        if dt <= 0:
            trace[i] = effcai
            continue
        ca_ext = max(0.0, ca - 70e-6)
        decay  = np.exp(-dt / tau)
        effcai = effcai * decay + ca_ext * tau * (1.0 - decay)
        trace[i] = effcai
    return trace, float(np.max(trace))


def load_pair(pair_name):
    path = THRESHOLD_TRACE_DIR / f"{pair_name}_threshold_traces.pkl"
    if not path.exists():
        sys.exit(f"ERROR: {path} not found")
    with open(path, "rb") as f:
        return pickle.load(f)


def find_first_pair():
    pkls = sorted(THRESHOLD_TRACE_DIR.glob("*_threshold_traces.pkl"))
    if not pkls:
        sys.exit(f"ERROR: no threshold trace pkl files found in {THRESHOLD_TRACE_DIR}")
    return pkls[0].name.replace("_threshold_traces.pkl", "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair",  default=None, help="Pair name, e.g. 180164-197248")
    parser.add_argument("--tau",   type=float, default=TAU_EFF, help="tau_eff in ms")
    parser.add_argument("--out",   default=None, help="Output PNG path")
    args = parser.parse_args()

    pair_name = args.pair or find_first_pair()
    print(f"Pair: {pair_name}  |  tau_eff = {args.tau} ms")

    data = load_pair(pair_name)
    gids = list(data["pre"]["cai_CR"].keys())
    t_pre_raw  = np.asarray(data["pre"]["t"],  dtype=np.float64)
    t_post_raw = np.asarray(data["post"]["t"], dtype=np.float64)

    n_syn = len(gids)
    ncols = 2   # pre | post
    nrows = n_syn
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(10, max(2 * nrows, 4)),
                             sharex=False, squeeze=False)
    fig.suptitle(f"cai_CR threshold traces — {pair_name}\n"
                 f"tau_eff = {args.tau:.3f} ms", fontsize=11)

    def _make_t(t_raw, n_pts, dt_ms=0.1):
        return t_raw if len(t_raw) == n_pts else np.arange(n_pts) * dt_ms

    for row, gid in enumerate(gids):
        cai_pre_arr  = np.asarray(data["pre"]["cai_CR"][gid],  dtype=np.float64)
        cai_post_arr = np.asarray(data["post"]["cai_CR"][gid], dtype=np.float64)

        t_pre  = _make_t(t_pre_raw,  len(cai_pre_arr))
        t_post = _make_t(t_post_raw, len(cai_post_arr))

        dt_pre  = np.diff(t_pre,  prepend=t_pre[0])
        dt_post = np.diff(t_post, prepend=t_post[0])

        eff_pre,  cp = peak_effcai(cai_pre_arr,  dt_pre[1:],  args.tau)
        eff_post, cq = peak_effcai(cai_post_arr, dt_post[1:], args.tau)

        for col, (t_arr, cai_arr, eff_arr, peak, label) in enumerate([
            (t_pre,  cai_pre_arr,  eff_pre,  cp, "pre"),
            (t_post, cai_post_arr, eff_post, cq, "post"),
        ]):
            ax = axes[row][col]
            ax2 = ax.twinx()

            ax.plot(t_arr, cai_arr * 1e3, color="steelblue", lw=1.2, label="cai_CR (mM×1e3)")
            ax2.plot(t_arr, eff_arr,      color="tomato",    lw=1.0, ls="--", label="effcai")

            ax.set_ylabel("cai_CR (µM)", color="steelblue", fontsize=7)
            ax2.set_ylabel("effcai (mM·ms)", color="tomato", fontsize=7)
            ax.tick_params(axis="y", labelcolor="steelblue", labelsize=6)
            ax2.tick_params(axis="y", labelcolor="tomato",    labelsize=6)
            ax.tick_params(axis="x", labelsize=6)
            ax.set_xlabel("t (ms)", fontsize=7)
            ax.set_title(f"GID {gid} — c{label} = {peak:.4g}", fontsize=8)

    plt.tight_layout()
    out = args.out or f"cai_cr_threshold_{pair_name}.png"
    plt.savefig(out, dpi=150)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()

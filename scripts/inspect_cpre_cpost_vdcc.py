#!/usr/bin/env python3
"""
Inspect c_pre_vdcc / c_post_vdcc for a single pair — the values that
gb_vdcc_only_shared.py (and gb_vdcc_only.py) feed into _jax_peak_effcai_zoh.

These are computed from the *VDCC* calcium traces (cai_VDCC_CR), not the
full cai_CR traces used by inspect_cpre_cpost.py.

Also shows the resulting thresholds and estimated theta_d / theta_p for
a set of a00/a01/a10/a11 coefficients (default or overridden via --a00
--a01 --a10 --a11).

Usage
-----
    python inspect_cpre_cpost_vdcc.py <pair_name> [--tau_eff 278.318]
        [--min_ca 70e-6] [--a00 2] [--a01 2] [--a10 4] [--a11 4]

Example
-------
    python inspect_cpre_cpost_vdcc.py L5TTPC_10Hz_10ms_pair_000
"""
import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from cicr_common import THRESHOLD_TRACE_DIR, _build_dt

DEFAULT_TAU   = 278.318
DEFAULT_MIN_CA = 70e-6


def _peak_effcai_zoh(cai_arr: np.ndarray, dt_arr: np.ndarray,
                     tau: float, min_ca: float) -> float:
    """Numpy replica of _jax_peak_effcai_zoh (ZOH exact discrete integration)."""
    effcai = 0.0
    peak   = 0.0
    for dt, ca in zip(dt_arr, cai_arr[1:]):
        decay   = np.exp(-dt / tau)
        ca_ext  = float(ca) - min_ca
        effcai  = effcai * decay + max(ca_ext, 0.0) * tau * (1.0 - decay)
        if effcai > peak:
            peak = effcai
    return peak


def main():
    parser = argparse.ArgumentParser(
        description="Inspect c_pre_vdcc / c_post_vdcc (from cai_VDCC_CR threshold traces)"
    )
    parser.add_argument("pair_name")
    parser.add_argument("--tau_eff", type=float, default=DEFAULT_TAU,
                        help="effcai time constant in ms (default: %(default)s)")
    parser.add_argument("--min_ca", type=float, default=DEFAULT_MIN_CA,
                        help="Resting Ca baseline in mM (default: %(default)s)")
    # Default threshold coefficients (matching GBVdccOnlySharedModel.DEFAULT_PARAMS)
    parser.add_argument("--a00", type=float, default=2.0, help="LTD pre-weight  (default: %(default)s)")
    parser.add_argument("--a01", type=float, default=2.0, help="LTD post-weight (default: %(default)s)")
    parser.add_argument("--a10", type=float, default=4.0, help="LTP pre-weight  (default: %(default)s)")
    parser.add_argument("--a11", type=float, default=4.0, help="LTP post-weight (default: %(default)s)")
    args = parser.parse_args()

    pkl_path = os.path.join(THRESHOLD_TRACE_DIR, f"{args.pair_name}_threshold_traces.pkl")
    if not os.path.exists(pkl_path):
        sys.exit(f"ERROR: threshold traces not found:\n  {pkl_path}")

    with open(pkl_path, "rb") as f:
        thresh_data = pickle.load(f)

    # --- sanity-check available keys ---
    for side in ("pre", "post"):
        section = thresh_data.get(side, {})
        avail = [k for k in section if k != "t"]
        print(f"[{side}] keys: {avail},  t len={len(section.get('t', []))}")
        if "cai_VDCC_CR" not in section:
            sys.exit(
                f"ERROR: 'cai_VDCC_CR' not in thresh_data['{side}']. "
                f"Available: {avail}"
            )

    ca_pre  = thresh_data["pre"]["cai_VDCC_CR"]
    ca_post = thresh_data["post"]["cai_VDCC_CR"]

    gids = list(ca_pre.keys())
    n_pre  = len(next(iter(ca_pre.values())))
    n_post = len(next(iter(ca_post.values())))

    dt_pre  = _build_dt(thresh_data["pre"]["t"],  n_pre)
    dt_post = _build_dt(thresh_data["post"]["t"], n_post)

    print(f"\nPair:    {args.pair_name}")
    print(f"tau_eff: {args.tau_eff} ms   min_ca: {args.min_ca:.3e} mM   N_gids: {len(gids)}")
    print(f"a00={args.a00}  a01={args.a01}  (theta_d = a00*cpre + a01*cpost)")
    print(f"a10={args.a10}  a11={args.a11}  (theta_p = a10*cpre + a11*cpost)")
    print(f"pre  trace: {n_pre} pts,  dt={dt_pre[0]:.4f} ms")
    print(f"post trace: {n_post} pts, dt={dt_post[0]:.4f} ms\n")

    hdr = (f"{'GID':<12} {'c_pre_vdcc':>14} {'c_post_vdcc':>14} "
           f"{'theta_d':>12} {'theta_p':>12}  {'LTD?':>5}  {'LTP?':>5}")
    print(hdr)
    print("-" * len(hdr))

    for gid in gids:
        cpre  = _peak_effcai_zoh(np.asarray(ca_pre[gid],  dtype=np.float64), dt_pre,  args.tau_eff, args.min_ca)
        cpost = _peak_effcai_zoh(np.asarray(ca_post[gid], dtype=np.float64), dt_post, args.tau_eff, args.min_ca)
        theta_d = args.a00 * cpre + args.a01 * cpost
        theta_p = args.a10 * cpre + args.a11 * cpost
        # "active" means effcai ever exceeds the threshold during the full trace;
        # here we just compare cpre+cpost-based thresholds against each other
        ltd_flag = "yes" if theta_d < theta_p else "---"  # sanity: LTP > LTD?
        ltp_flag = "yes" if theta_p > theta_d else "---"
        print(f"{str(gid):<12} {cpre:>14.6e} {cpost:>14.6e} "
              f"{theta_d:>12.6e} {theta_p:>12.6e}  {ltd_flag:>5}  {ltp_flag:>5}")


if __name__ == "__main__":
    main()

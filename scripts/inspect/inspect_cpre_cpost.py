#!/usr/bin/env python3
import argparse
import pickle
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from plastyfitting.cicr_common import THRESHOLD_TRACE_DIR, compute_cpre_cpost_zoh, compute_cpre_cpost_analytical

DEFAULT_TAU = 278.318


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pair_name")
    parser.add_argument("--tau_eff", type=float, default=DEFAULT_TAU)
    parser.add_argument("--ca_key", default="cai_CR", choices=["cai_CR", "shaft_cai"])
    parser.add_argument("--min_ca", type=float, default=70e-6)
    args = parser.parse_args()

    pkl_path = os.path.join(THRESHOLD_TRACE_DIR, f"{args.pair_name}_threshold_traces.pkl")
    if not os.path.exists(pkl_path):
        sys.exit(f"ERROR: not found: {pkl_path}")

    with open(pkl_path, "rb") as f:
        thresh_data = pickle.load(f)

    for side in ("pre", "post"):
        t = thresh_data[side]["t"]
        ca = thresh_data[side]["cai_CR"]
        first_gid = list(ca.keys())[0]
        trace = ca[first_gid]
        print(f"[{side}] t: len={len(t)}, t[0]={t[0]:.3f}, t[-1]={t[-1]:.3f} ms")
        print(f"       cai_CR[{first_gid}]: len={len(trace)}, max={max(trace):.6e}, min={min(trace):.6e}")

    if args.ca_key not in thresh_data.get("pre", {}):
        sys.exit(f"ERROR: key '{args.ca_key}' not in thresh_data['pre']. Available: {list(thresh_data['pre'].keys())}")

    gids = list(thresh_data["pre"][args.ca_key].keys())
    print(f"\nPair: {args.pair_name}, tau_eff: {args.tau_eff} ms, ca_key: {args.ca_key}, N: {len(gids)}")
    print(f"GIDs: {gids}\n")

    zoh_pre,  zoh_post  = compute_cpre_cpost_zoh(thresh_data, gids, args.tau_eff, args.ca_key, args.min_ca)
    anal_pre, anal_post = compute_cpre_cpost_analytical(thresh_data, gids, args.tau_eff, args.ca_key, args.min_ca)

    print(f"{'GID':<12} {'zoh_cpre':>14} {'zoh_cpost':>14} {'anal_cpre':>14} {'anal_cpost':>14}")
    print("-" * 70)
    for gid, zp, zq, ap, aq in zip(gids, zoh_pre, zoh_post, anal_pre, anal_post):
        print(f"{gid:<12} {zp:>14.6e} {zq:>14.6e} {ap:>14.6e} {aq:>14.6e}")


if __name__ == "__main__":
    main()

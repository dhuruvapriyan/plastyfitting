#!/usr/bin/env python3
"""
Pre-bake combined calcium traces:
    cai_CR_modded = alpha * cai_NMDA_CR + beta * cai_VDCC_CR

Processes two kinds of pkl files:

1. Simulation traces  (per-pair, per-protocol)
   cai_CR is replaced with alpha*cai_NMDA_CR + beta*cai_VDCC_CR.
   All other keys (rho_GB, syn_props, nves, t, prespikes, …) are preserved.

2. Threshold traces  (pre/post single-pulse, for c_pre / c_post integration)
   cai_CR in both the 'pre' and 'post' sections is likewise replaced so that
   the effcai integrator in _load_pkl sees the same weighted calcium signal,
   giving consistent theta_d / theta_p thresholds.

The output directory is self-contained: cicr_common._load_pkl automatically
prefers a sibling threshold_traces_out/ when one exists, so no code changes
are needed at fit time.

Usage
-----
    python precompute_weighted_cai.py                    # alpha=3, beta=1
    python precompute_weighted_cai.py --alpha 3 --beta 1 --protocols 10Hz_10ms 10Hz_-10ms
    python precompute_weighted_cai.py --dry-run          # print plan, write nothing

Output
------
    trace_results/PRECOMPUTED_A<alpha>B<beta>/
        <pre_gid>-<post_gid>/
            10Hz_10ms/simulation_traces.pkl
            10Hz_-10ms/simulation_traces.pkl
        threshold_traces_out/
            <pair>_threshold_traces.pkl
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

BASE_DIR = Path("/project/rrg-emuller/dhuruva/plastyfitting")
SRC_L5_DIR  = BASE_DIR / "trace_results/CHINDEMI_PARAMS"


MIN_CA_CR = 70e-6  # mM — resting calcium, matches min_ca_CR in GluSynapse.mod


def _bake_cai_dict(cai_nmda: dict, cai_vdcc: dict, alpha: float, beta: float) -> dict:
    """Combine two {gid: array} dicts into one weighted dict.

    Each component decays to min_ca_CR (not zero), so we weight only the
    *excess* above resting and then add the baseline back once:

        cai_CR = min_ca + alpha*(cai_NMDA - min_ca) + beta*(cai_VDCC - min_ca)

    This keeps the resting level at min_ca regardless of alpha/beta.
    """
    out = {}
    for gid, nmda_arr in cai_nmda.items():
        nmda = np.asarray(nmda_arr, dtype=np.float64)
        vdcc_arr = cai_vdcc.get(gid, None)
        vdcc = np.asarray(vdcc_arr, dtype=np.float64) if vdcc_arr is not None else np.full_like(nmda, MIN_CA_CR)
        if vdcc.shape != nmda.shape:
            vdcc = np.full_like(nmda, MIN_CA_CR)
        out[gid] = MIN_CA_CR + alpha * (nmda - MIN_CA_CR) + beta * (vdcc - MIN_CA_CR)
    return out


def _bake_threshold_pkl(src_path: Path, dst_path: Path, alpha: float, beta: float) -> str:
    """Load a threshold traces pkl, replace cai_CR in both pre and post with
    alpha*cai_NMDA_CR + beta*cai_VDCC_CR, dump to dst."""
    with open(src_path, "rb") as f:
        data = pickle.load(f)

    out = dict(data)
    for side in ("pre", "post"):
        if side not in data:
            continue
        section = dict(data[side])
        nmda = section.get("cai_NMDA_CR")
        vdcc = section.get("cai_VDCC_CR")
        if nmda is None:
            return f"SKIP threshold ({side} missing cai_NMDA_CR)"
        if vdcc is None:
            return f"SKIP threshold ({side} missing cai_VDCC_CR)"
        if not isinstance(nmda, dict):
            return f"SKIP threshold ({side} cai_NMDA_CR is not a dict)"
        section["cai_CR"] = _bake_cai_dict(nmda, vdcc, alpha, beta)
        section["cai_CR_modded_alpha"] = float(alpha)
        section["cai_CR_modded_beta"]  = float(beta)
        out[side] = section

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_path, "wb") as f:
        pickle.dump(out, f, protocol=4)

    n_gids = len(out["pre"]["cai_CR"])
    return f"OK  ({n_gids} syns, both sides baked)"


def _bake_protocol_pkl(src_path: Path, dst_path: Path, alpha: float, beta: float) -> str:
    """Load src pkl, replace cai_CR with alpha*NMDA+beta*VDCC, dump to dst."""
    with open(src_path, "rb") as f:
        data = pickle.load(f)

    cai_nmda = data.get("cai_NMDA_CR")
    cai_vdcc = data.get("cai_VDCC_CR")

    if cai_nmda is None:
        return f"SKIP (no cai_NMDA_CR)"
    if cai_vdcc is None:
        return f"SKIP (no cai_VDCC_CR)"

    # Support both dict-of-arrays (new format) and plain ndarray (legacy)
    if isinstance(cai_nmda, dict):
        cai_modded = _bake_cai_dict(cai_nmda, cai_vdcc if isinstance(cai_vdcc, dict) else {}, alpha, beta)
    else:
        nmda = np.asarray(cai_nmda, dtype=np.float64)
        vdcc = np.asarray(cai_vdcc, dtype=np.float64) if cai_vdcc is not None else np.full_like(nmda, MIN_CA_CR)
        cai_modded = MIN_CA_CR + alpha * (nmda - MIN_CA_CR) + beta * (vdcc - MIN_CA_CR)

    out = dict(data)
    out["cai_CR"] = cai_modded
    # Record provenance so we can verify later
    out["cai_CR_modded_alpha"] = float(alpha)
    out["cai_CR_modded_beta"] = float(beta)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_path, "wb") as f:
        pickle.dump(out, f, protocol=4)  # protocol 4 = Python 3.8+, compact

    n_gids = len(cai_modded) if isinstance(cai_modded, dict) else cai_modded.shape[0]
    return f"OK  ({n_gids} syns)"


def main():
    parser = argparse.ArgumentParser(
        description="Pre-bake cai_CR = alpha*cai_NMDA_CR + beta*cai_VDCC_CR into new pkl files"
    )
    parser.add_argument("--alpha", type=float, default=3.0, help="Weight for cai_NMDA_CR (default: 3)")
    parser.add_argument("--beta",  type=float, default=1.0, help="Weight for cai_VDCC_CR (default: 1)")
    parser.add_argument(
        "--protocols", nargs="+",
        default=["10Hz_10ms", "10Hz_-10ms"],
        help="Protocols to process (default: 10Hz_10ms 10Hz_-10ms)",
    )
    parser.add_argument("--src-dir", type=Path, default=SRC_L5_DIR,
                        help=f"Source L5 trace directory (default: {SRC_L5_DIR})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be done without writing any files")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing output files (default: skip)")
    args = parser.parse_args()

    alpha, beta = args.alpha, args.beta
    # Build a clean tag like "A3B1" or "A2p5B1"
    def _fmt(v):
        return str(int(v)) if v == int(v) else f"{v:.2g}".replace(".", "p")
    tag = f"A{_fmt(alpha)}B{_fmt(beta)}"
    dst_l5_dir = BASE_DIR / f"trace_results/PRECOMPUTED_{tag}"

    dst_thresh_dir = dst_l5_dir / "threshold_traces_out"
    # source threshold traces live in threshold_traces_out/ inside src_dir
    src_thresh_dir = Path(str(args.src_dir)) / "threshold_traces_out"

    print(f"=== Precomputing cai_CR = {alpha}*cai_NMDA_CR + {beta}*cai_VDCC_CR ===")
    print(f"Source : {args.src_dir}")
    print(f"Dest   : {dst_l5_dir}")
    print(f"Protos : {args.protocols}")
    print(f"Thresh : {src_thresh_dir} -> {dst_thresh_dir}")
    if args.dry_run:
        print("DRY-RUN: no files will be written\n")
    else:
        print()

    if not args.src_dir.exists():
        print(f"ERROR: source directory does not exist: {args.src_dir}", file=sys.stderr)
        sys.exit(1)

    pair_dirs = sorted(d for d in args.src_dir.iterdir() if d.is_dir() and len(d.name.split("-")) == 2)
    n_ok = n_skip = n_exist = n_err = 0

    # --- 1. Bake simulation trace pkls ---
    print("\n-- Simulation traces --")
    for pair_dir in pair_dirs:
        for proto in args.protocols:
            src_pkl = pair_dir / proto / "simulation_traces.pkl"
            if not src_pkl.exists():
                continue

            dst_pkl = dst_l5_dir / pair_dir.name / proto / "simulation_traces.pkl"

            if dst_pkl.exists() and not args.overwrite:
                n_exist += 1
                print(f"  EXISTS  {pair_dir.name}/{proto}")
                continue

            if args.dry_run:
                print(f"  WOULD   {pair_dir.name}/{proto}  ->  {dst_pkl}")
                n_ok += 1
                continue

            try:
                status = _bake_protocol_pkl(src_pkl, dst_pkl, alpha, beta)
            except Exception as exc:
                status = f"ERROR: {exc}"
                n_err += 1
                print(f"  {status}  [{pair_dir.name}/{proto}]", file=sys.stderr)
                continue

            if status.startswith("OK"):
                n_ok += 1
            else:
                n_skip += 1
            print(f"  {status}  {pair_dir.name}/{proto}")

    # --- 2. Bake threshold trace pkls (pre/post cai_CR) ---
    print("\n-- Threshold traces --")
    t_ok = t_skip = t_exist = t_err = 0
    if src_thresh_dir.exists():
        for src_thresh_pkl in sorted(src_thresh_dir.glob("*_threshold_traces.pkl")):
            dst_thresh_pkl = dst_thresh_dir / src_thresh_pkl.name

            if dst_thresh_pkl.exists() and not args.overwrite:
                t_exist += 1
                print(f"  EXISTS  {src_thresh_pkl.name}")
                continue

            if args.dry_run:
                print(f"  WOULD   {src_thresh_pkl.name}  ->  {dst_thresh_pkl}")
                t_ok += 1
                continue

            try:
                status = _bake_threshold_pkl(src_thresh_pkl, dst_thresh_pkl, alpha, beta)
            except Exception as exc:
                status = f"ERROR: {exc}"
                t_err += 1
                print(f"  {status}  [{src_thresh_pkl.name}]", file=sys.stderr)
                continue

            if status.startswith("OK"):
                t_ok += 1
            else:
                t_skip += 1
            print(f"  {status}  {src_thresh_pkl.name}")
    else:
        print(f"  WARNING: threshold source dir not found: {src_thresh_dir}")

    print()
    print(f"Simulation traces: Written={n_ok}  Skipped={n_skip}  AlreadyExist={n_exist}  Errors={n_err}")
    print(f"Threshold traces:  Written={t_ok}  Skipped={t_skip}  AlreadyExist={t_exist}  Errors={t_err}")
    if not args.dry_run and (n_ok + t_ok) > 0:
        print(f"\nTo fit with precomputed traces:")
        print(f"  python gb_only_precomputed.py --trace-dir {dst_l5_dir} --method de")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate Cpre/Cpost pipeline against NEURON ground truth.

Compares four sources of Cpre/Cpost for each synapse:
  1. stored     — scalar stored in threshold_traces.pkl (pre-computed at generation time)
  2. recomputed — ZOH from cai_CR at 1ms (via cicr_common.compute_cpre_cpost_from_shaft_cai)
  3. neuron     — synprop['Cpre'/'Cpost'] from simulation_traces.pkl (NEURON CVODE resolution)
  4. theta      — back-computes Cpre from NEURON's stored theta_d_GB using Chindemi a-coefficients

Checks whether our JAX threshold formula theta_d = a00*c_pre + a01*c_post
agrees with NEURON's theta_d_GB stored in synprop.

Usage:
    python scripts/validate_cpre_cpost.py
    python scripts/validate_cpre_cpost.py --all-pairs --protocol 10Hz_5ms
"""

import argparse
import pickle
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from plastyfitting.cicr_common import compute_cpre_cpost_from_shaft_cai

CHINDEMI_PARAMS = {
    "a00": 1.002, "a01": 1.954,
    "a10": 1.159, "a11": 2.483,
    "a20": 1.127, "a21": 2.456,
    "a30": 5.236, "a31": 1.782,
    "tau_effca": 278.318,
}

BASE_DIR  = "/project/rrg-emuller/dhuruva/plastyfitting"
TRACE_DIR = BASE_DIR + "/trace_results/CHINDEMI_PARAMS"


def peak_zoh(cai, dt, tau, min_ca=70e-6):
    """ZOH peak effcai (exact for fixed-dt recordings)."""
    eff, peak = 0.0, 0.0
    decay  = np.exp(-dt / tau)
    factor = tau * (1.0 - decay)
    for ca in np.asarray(cai, dtype=np.float64)[1:]:
        u = max(0.0, ca - min_ca)
        eff = eff * decay + u * factor
        if eff > peak:
            peak = eff
    return peak


def theta(cp, cq, loc, p):
    if loc == "apical":
        return p["a20"]*cp + p["a21"]*cq, p["a30"]*cp + p["a31"]*cq
    return p["a00"]*cp + p["a01"]*cq, p["a10"]*cp + p["a11"]*cq


def validate_pair(thresh_pkl, sim_pkl, tau, params):
    with open(thresh_pkl, "rb") as f: th  = pickle.load(f)
    with open(sim_pkl,    "rb") as f: sim = pickle.load(f)

    gids = list(th["pre"]["cai_CR"].keys())
    sp   = sim.get("synprop", {})
    locs = sp.get("loc", ["basal"] * len(gids))

    # t_recorded gives the correct 1ms time axis for cai_CR
    dt_pre  = float(np.asarray(th["pre"]["t_recorded"])[1]
                    - np.asarray(th["pre"]["t_recorded"])[0])
    dt_post = float(np.asarray(th["post"]["t_recorded"])[1]
                    - np.asarray(th["post"]["t_recorded"])[0])

    # Source 2: cicr_common pipeline (uses _build_dt internally)
    c_pre_cc, c_post_cc = compute_cpre_cpost_from_shaft_cai(
        th, gids, tau_effca=tau, ca_key="cai_CR")

    print(f"  {'syn':>3}  {'loc':6}  "
          f"{'stored_pre':>11}  {'cc_pre':>11}  {'neuron_pre':>11}  "
          f"{'stored_post':>12}  {'cc_post':>12}  {'neuron_post':>12}  "
          f"{'td_calc':>9}  {'td_neu':>9}  {'td_match':>8}  "
          f"{'tp_calc':>9}  {'tp_neu':>9}  {'tp_match':>8}")

    all_ok = True
    results = []

    for i, g in enumerate(gids):
        loc = locs[i]

        # Source 1: stored scalars
        stored_pre  = float(th["pre"]["c_pre"][g])
        stored_post = float(th["post"]["c_post"][g])

        # Source 2: cicr_common recompute
        cc_pre  = c_pre_cc[i]
        cc_post = c_post_cc[i]

        # Source 3: NEURON synprop (high-res CVODE, used to compute theta_d/p)
        n_pre  = float(sp["Cpre"][i])  if "Cpre"  in sp else float("nan")
        n_post = float(sp["Cpost"][i]) if "Cpost" in sp else float("nan")

        # Source 4: NEURON stored thresholds
        td_neu = float(sp["theta_d_GB"][i]) if "theta_d_GB" in sp else float("nan")
        tp_neu = float(sp["theta_p_GB"][i]) if "theta_p_GB" in sp else float("nan")

        # Compute thresholds from cicr_common c_pre/c_post
        td_calc, tp_calc = theta(cc_pre, cc_post, loc, params)

        td_rel = abs(td_calc - td_neu) / (abs(td_neu) + 1e-12)
        tp_rel = abs(tp_calc - tp_neu) / (abs(tp_neu) + 1e-12)
        td_ok  = td_rel < 0.02   # <2% relative
        tp_ok  = tp_rel < 0.02

        if not (td_ok and tp_ok):
            all_ok = False

        print(f"  {i:>3}  {loc:6}  "
              f"{stored_pre:>11.6f}  {cc_pre:>11.6f}  {n_pre:>11.6f}  "
              f"{stored_post:>12.6f}  {cc_post:>12.6f}  {n_post:>12.6f}  "
              f"{td_calc:>9.5f}  {td_neu:>9.5f}  {'OK' if td_ok else 'FAIL':>8}  "
              f"{tp_calc:>9.5f}  {tp_neu:>9.5f}  {'OK' if tp_ok else 'FAIL':>8}")

        results.append(dict(
            gid=g, loc=loc,
            stored_pre=stored_pre, cc_pre=cc_pre, neuron_pre=n_pre,
            stored_post=stored_post, cc_post=cc_post, neuron_post=n_post,
            td_calc=td_calc, td_neu=td_neu, td_rel=td_rel,
            tp_calc=tp_calc, tp_neu=tp_neu, tp_rel=tp_rel,
        ))

    # Summary
    pre_diffs  = [abs(r["cc_pre"]  - r["stored_pre"])  for r in results]
    post_diffs = [abs(r["cc_post"] - r["stored_post"]) for r in results]
    td_rels    = [r["td_rel"] for r in results]
    print(f"\n  cicr_common vs stored:  Δc_pre max={max(pre_diffs):.2e}  Δc_post max={max(post_diffs):.2e}")
    print(f"  theta_d rel err:        mean={np.mean(td_rels)*100:.3f}%  max={max(td_rels)*100:.3f}%")
    print(f"  → thresholds match NEURON (±2%)? {'YES ✓' if all_ok else 'NO ✗'}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Validate Cpre/Cpost pipeline")
    parser.add_argument("--pair",      type=str, default="180164-197248")
    parser.add_argument("--protocol",  type=str, default="10Hz_5ms")
    parser.add_argument("--all-pairs", action="store_true")
    parser.add_argument("--tau",       type=float, default=CHINDEMI_PARAMS["tau_effca"])
    args = parser.parse_args()

    print(f"tau_effca = {args.tau} ms\n")
    print("Columns: stored=threshold pkl scalar | cc=cicr_common recomputed | neuron=synprop CVODE")

    trace_root = Path(TRACE_DIR)
    pair_dirs  = (sorted(d for d in trace_root.iterdir() if d.is_dir())
                  if args.all_pairs else [trace_root / args.pair])

    all_results = []
    for pair_dir in pair_dirs:
        thresh_pkl = (Path(TRACE_DIR).parent
                      / "CHINDEMI_PARAMS/threshold_traces_out"
                      / f"{pair_dir.name}_threshold_traces.pkl")
        sim_pkl    = pair_dir / args.protocol / "simulation_traces.pkl"

        if not thresh_pkl.exists() or not sim_pkl.exists():
            print(f"SKIP {pair_dir.name}: missing files")
            continue

        print(f"\n=== {pair_dir.name} / {args.protocol} ===")
        try:
            res = validate_pair(str(thresh_pkl), str(sim_pkl), args.tau, CHINDEMI_PARAMS)
            for r in res:
                r["pair"] = pair_dir.name
            all_results.extend(res)
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}"); traceback.print_exc()

    if len(all_results) > 1:
        td_rels = [r["td_rel"] for r in all_results]
        print(f"\n{'='*60}")
        print(f"OVERALL  ({len(all_results)} synapses, {len(pair_dirs)} pairs)")
        print(f"  theta_d rel err: mean={np.mean(td_rels)*100:.3f}%  max={max(td_rels)*100:.3f}%")
        n_ok = sum(1 for r in all_results if r["td_rel"] < 0.02 and r["tp_rel"] < 0.02)
        print(f"  thresholds match (±2%): {n_ok}/{len(all_results)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Diagnostic: evaluate gb_vdcc_only params and print per-pair EPSP ratios.

This replicates *exactly* what the optimizer computed, then breaks it down
per-pair so you can see why the mean ratio ends up where it does even if
large-VDCC synapses dominate the average.

Usage
-----
    python scripts/diag_epsp_ratio.py \
        --params best_params_gb_vdcc_only_20260307_104913.json \
        --protocols 10Hz_10ms 10Hz_-10ms
"""

import argparse
import json
import os
import sys
import logging

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gb_vdcc_only import GBVdccOnlyModel
from cicr_common_weighted import (
    EXPERIMENTAL_TARGETS,
    EXPERIMENTAL_ERRORS,
    preload_all_data,
    collate_protocol_to_jax,
    _jax_peak_effcai_zoh,
)


_MIN_CA = 70e-6


def per_pair_ratios(model, params_x):
    """
    Re-run sim_pair for every pair in every protocol and return raw ratio arrays,
    PLUS per-synapse rho_final and c_pre/c_post/thresholds.
    """
    params      = model.unpack_params(params_x)
    gb, aux     = params
    tau_effca   = float(aux["tau_eff"])
    step_fn     = model.get_step_factory()
    init_fn     = model.get_init_fn()

    results = {}
    for proto, cd in model.collated_data.items():
        n_pairs   = cd["cai_vdcc"].shape[0]
        ratio_list = []
        syn_details = []

        for i in range(n_pairs):
            cai_vdcc_p   = np.asarray(cd["cai_vdcc"][i])    # (n_syn, T)
            cai_nmda_p   = np.asarray(cd["cai_nmda"][i])
            t_p          = np.asarray(cd["t"][i])            # (T,)
            nves_p       = np.asarray(cd["nves"][i])
            isapi_p      = np.asarray(cd["is_apical"][i])
            rho0_p       = np.asarray(cd["rho0"][i])
            bmean        = float(cd["baseline"][i])
            sm_p         = np.asarray(cd["singletons"][i])
            valid_p      = np.asarray(cd["valid"][i])

            cai_pre_vdcc_p  = np.asarray(cd["cai_pre_vdcc"][i])   # (n_syn, T_pre)
            cai_post_vdcc_p = np.asarray(cd["cai_post_vdcc"][i])  # (n_syn, T_post)
            t_pre_p  = np.asarray(cd["t_pre"][i])                 # (T_pre,)  or (n_syn, T_pre) 
            t_post_p = np.asarray(cd["t_post"][i])

            dt_trace = np.diff(t_p, prepend=t_p[0])
            dt_trace = np.where(dt_trace <= 0, 1e-6, dt_trace)

            # dt for threshold traces
            if t_pre_p.ndim == 1:
                dt_pre  = max(float(t_pre_p[1]  - t_pre_p[0]),  1e-6) if len(t_pre_p)  > 1 else 0.025
                dt_post = max(float(t_post_p[1] - t_post_p[0]), 1e-6) if len(t_post_p) > 1 else 0.025
            else:
                dt_pre  = 0.025
                dt_post = 0.025

            n_syn     = cai_vdcc_p.shape[0]
            contribs_after  = np.zeros(n_syn)
            contribs_before = np.zeros(n_syn)

            syn_rows = []
            for s in range(n_syn):
                if not valid_p[s]:
                    syn_rows.append(None)
                    continue

                apical = bool(isapi_p[s])

                # Per-synapse threshold traces
                cv_pre  = cai_pre_vdcc_p[s]
                cv_post = cai_post_vdcc_p[s]

                c_pre_v  = float(_jax_peak_effcai_zoh(jnp.array(cv_pre),  tau_effca, dt_pre,  _MIN_CA))
                c_post_v = float(_jax_peak_effcai_zoh(jnp.array(cv_post), tau_effca, dt_post, _MIN_CA))

                if apical:
                    theta_d = float(gb["d_v0a"]) * c_pre_v + float(gb["d_v1a"]) * c_post_v
                    theta_p = float(gb["p_v0a"]) * c_pre_v + float(gb["p_v1a"]) * c_post_v
                else:
                    theta_d = float(gb["d_v0"])  * c_pre_v + float(gb["d_v1"])  * c_post_v
                    theta_p = float(gb["p_v0"])  * c_pre_v + float(gb["p_v1"])  * c_post_v

                # Integrate effcai + rho
                decay_arr  = np.exp(-dt_trace / tau_effca)
                factor_arr = tau_effca * (1.0 - decay_arr)

                eff = 0.0
                rho = float(rho0_p[s])
                gamma_d = float(gb["gamma_d"])
                gamma_p = float(gb["gamma_p"])

                for t_idx in range(1, len(t_p)):
                    dt    = float(dt_trace[t_idx])
                    ca    = float(cai_vdcc_p[s, t_idx])
                    decay = float(decay_arr[t_idx])
                    fac   = float(factor_arr[t_idx])

                    eff = eff * decay + (ca - _MIN_CA) * fac

                    dep = 1.0 if eff > theta_d else 0.0
                    pot = 1.0 if eff > theta_p else 0.0

                    drho = (
                        -rho * (1.0 - rho) * (0.5 - rho)
                        + pot * gamma_p * (1.0 - rho)
                        - dep * gamma_d * rho
                    ) / 70000.0
                    rho = float(np.clip(rho + dt * drho, 0.0, 1.0))

                rho_final = rho
                sm = float(sm_p[s])
                potentiated_final  = rho_final  >= 0.5
                potentiated_before = float(rho0_p[s]) >= 0.5

                contribs_after[s]  = (sm - bmean) if potentiated_final  else 0.0
                contribs_before[s] = (sm - bmean) if potentiated_before else 0.0

                syn_rows.append(dict(
                    apical=apical, rho0=float(rho0_p[s]), rho_final=rho_final,
                    c_pre=c_pre_v, c_post=c_post_v,
                    theta_d=theta_d, theta_p=theta_p,
                    eff_peak=max(eff, 0.0),    # approximate last eff value
                    sm=sm, pot_final=potentiated_final,
                ))

            epsp_after  = bmean + float(np.sum(contribs_after[valid_p]))
            epsp_before = bmean + float(np.sum(contribs_before[valid_p]))
            ratio = epsp_after / epsp_before if epsp_before > 0 else float("nan")

            ratio_list.append(ratio)
            syn_details.append(syn_rows)

        results[proto] = dict(
            ratios=np.array(ratio_list, dtype=np.float64),
            syn_details=syn_details,
        )
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params",    required=True, help="JSON param file")
    parser.add_argument("--protocols", nargs="+", default=["10Hz_10ms", "10Hz_-10ms"])
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--dt-step",   type=int, default=10)
    parser.add_argument("--top",       type=int, default=20,
                        help="Print per-pair table for the top-N highest/lowest ratio pairs")
    parser.add_argument("--verbose",   action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    with open(args.params) as f:
        p = json.load(f)
    print(f"\nParams: {args.params}")
    for k, v in p.items():
        print(f"  {k:>12} = {v:.6f}")

    targets = {proto: EXPERIMENTAL_TARGETS[proto] for proto in args.protocols
               if proto in EXPERIMENTAL_TARGETS}

    print(f"\nLoading data for {list(targets.keys())} ...")
    protocol_data = preload_all_data(
        max_pairs=args.max_pairs,
        protocols=list(targets.keys()),
        needs_threshold_traces=True,
        include_raw_cai=False,
        include_base_threshold_traces=True,
        include_cpre_cpost=False,
    )

    model = GBVdccOnlyModel()
    model.setup_jax(protocol_data, targets, dt_step=args.dt_step)

    # Build param vector
    dp = dict(model.DEFAULT_PARAMS)
    dp.update(p)
    x = jnp.array([dp[n] for n, *_ in model.FIT_PARAMS])

    # ── Fast JAX mean (what optimizer saw) ───────────────────────────────────
    preds = np.array(model.forward_batch(x[None])[0])
    loss  = float(model.objective_single(x))

    print(f"\n{'═'*60}")
    print(f"  Optimizer view (JAX nanmean over all pairs)")
    print(f"{'─'*60}")
    print(f"  {'Protocol':<16}  {'Predicted':>10}  {'Target':>10}  {'Error':>10}")
    print(f"{'─'*60}")
    for proto, pred in zip(model.proto_names, preds):
        tgt = targets[proto]
        print(f"  {proto:<16}  {pred:>10.4f}  {tgt:>10.4f}  {pred-tgt:>+10.4f}")
    print(f"{'─'*60}")
    print(f"  Loss (objective): {loss:.8f}")
    print(f"{'═'*60}")

    # ── Per-pair breakdown (numpy, slower but per-pair) ───────────────────────
    print(f"\nRunning per-pair numpy integration (dt_step={args.dt_step}) ...")
    res = per_pair_ratios(model, x)

    for proto, data in res.items():
        ratios = data["ratios"]
        valid  = ~np.isnan(ratios)
        tgt    = targets[proto]

        print(f"\n{'═'*70}")
        print(f"  Protocol: {proto}   n_pairs={len(ratios)}  n_valid={valid.sum()}")
        print(f"  nanmean={np.nanmean(ratios):.4f}  target={tgt:.4f}  "
              f"RMSE={np.sqrt(np.nanmean((ratios[valid]-tgt)**2)):.4f}")
        print(f"{'─'*70}")

        # Distribution
        bins = [0, 0.8, 0.95, 1.05, 1.2, 1.5, 2.0, 999]
        labels = ["<0.80", "0.80-0.95", "0.95-1.05", "1.05-1.20", "1.20-1.50",
                  "1.50-2.00", ">2.00"]
        r_v = ratios[valid]
        print(f"  Ratio distribution:")
        for lo, hi, lbl in zip(bins, bins[1:], labels):
            count = int(np.sum((r_v >= lo) & (r_v < hi)))
            bar   = "█" * int(count / max(len(r_v), 1) * 40)
            print(f"    {lbl:>12}  {count:>4}  {bar}")

        # Top extreme pairs
        idxs_sorted = np.argsort(ratios)
        top_n = min(args.top, len(ratios))
        show_idxs = np.concatenate([idxs_sorted[:top_n//2],
                                    idxs_sorted[-(top_n - top_n//2):]])
        show_idxs = sorted(set(show_idxs.tolist()))

        print(f"\n  Per-pair detail (top {top_n} extreme ratios):")
        hdr = (f"  {'pair':>6}  {'ratio':>7}  {'n_syn':>5}  "
               f"{'n_pot_before':>12}  {'n_pot_after':>11}  {'n_dep_cross':>11}  {'n_pot_cross':>11}")
        print(f"{'─'*len(hdr)}")
        print(hdr)
        print(f"{'─'*len(hdr)}")
        for idx in show_idxs:
            r  = ratios[idx]
            sd = data["syn_details"][idx]
            if sd is None:
                continue
            valid_syns = [s for s in sd if s is not None]
            n_syn          = len(valid_syns)
            n_pot_before   = sum(1 for s in valid_syns if s["rho0"]     >= 0.5)
            n_pot_after    = sum(1 for s in valid_syns if s["pot_final"])
            # dep/pot cross = theta exceeded at least once (approximated by rho change)
            n_dep_cross    = sum(1 for s in valid_syns if s["theta_d"] > 0 and s["c_pre"] + s["c_post"] > 0
                                 and (s["c_pre"] + s["c_post"]) * 1.5 > s["theta_d"])   # rough heuristic
            n_pot_cross    = sum(1 for s in valid_syns if s["theta_p"] > 0
                                 and (s["c_pre"] + s["c_post"]) * 1.5 > s["theta_p"])
            flag = "  ← EXTREME" if abs(r - 1.0) > 0.3 else ""
            print(f"  {idx:>6}  {r:>7.4f}  {n_syn:>5}  "
                  f"{n_pot_before:>12}  {n_pot_after:>11}  {n_dep_cross:>11}  {n_pot_cross:>11}{flag}")

        # Print synapse detail for the single most extreme pair
        extreme_idx = int(idxs_sorted[0] if abs(ratios[idxs_sorted[0]] - 1.0) >
                           abs(ratios[idxs_sorted[-1]] - 1.0) else idxs_sorted[-1])
        sd = data["syn_details"][extreme_idx]
        print(f"\n  Synapse detail for pair {extreme_idx} (ratio={ratios[extreme_idx]:.4f}):")
        print(f"  {'syn':>4}  {'loc':>6}  {'rho0':>6}  {'rho_f':>6}  "
              f"{'c_pre':>10}  {'c_post':>10}  {'theta_d':>12}  {'theta_p':>12}  {'sm':>8}")
        for si, s in enumerate(sd):
            if s is None:
                continue
            loc = "apical" if s["apical"] else "basal"
            print(f"  {si:>4}  {loc:>6}  {s['rho0']:>6.3f}  {s['rho_final']:>6.3f}  "
                  f"{s['c_pre']:>10.6f}  {s['c_post']:>10.6f}  "
                  f"{s['theta_d']:>12.6f}  {s['theta_p']:>12.6f}  {s['sm']:>8.4f}")


if __name__ == "__main__":
    main()

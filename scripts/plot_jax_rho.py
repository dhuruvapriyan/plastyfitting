#!/usr/bin/env python3
"""
JAX rho-trace collector and plotter for gb_vdcc_only.

Uses JAX jit+vmap over ALL pairs × synapses in one compiled call — GPU/CPU
batched instead of a Python loop over timesteps.

Produces:
  1. A CSV with the same columns as rho_v17.csv:
         pregid, postgid, freq, delay, syn_id,
         initial_rho_raw, final_rho_raw, initial_rho, final_rho
  2. Optional plots:
       --plot summary  →  rho transition bars + potentiated-fraction curve
       --plot traces   →  per-pair effcai+rho(t) traces (1 fig per pair/proto)
       --plot both     →  all of the above

Usage
-----
    python scripts/plot_jax_rho.py \\
        --params best_params_gb_vdcc_only_20260307_104913.json \\
        --protocols 10Hz_10ms 10Hz_-10ms \\
        --output rho_jax.csv \\
        --plot summary \\
        --neuron-csv /home/dhuruva/.../rho_v17.csv
"""

import argparse
import json
import os
import sys
from pathlib import Path
from functools import partial

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gb_vdcc_only import GBVdccOnlyModel
from cicr_common_weighted import _load_pkl_weighted, collate_protocol_to_jax
import cicr_common as _base
from cicr_common import L5_TRACE_DIR, L5_BASIS_DIR, _jax_peak_effcai_zoh

_MIN_CA = 70e-6   # mM — matches gb_vdcc_only.py


# ──────────────────────────────────────────────────────────────────────────────
# Data loading — same walk order as preload_all_data, but keeps pair names
# ──────────────────────────────────────────────────────────────────────────────

def load_data_with_names(protocols, max_pairs=None, trace_dir=None):
    """
    Walk trace directory, load pairs, and track (pregid, postgid) in load order.

    Returns:
        protocol_data : dict {proto: [pair_dict, ...]}   (same as preload_all_data)
        pair_names    : dict {proto: [(pregid, postgid), ...]}
    """
    search_dir = Path(trace_dir) if trace_dir else Path(L5_TRACE_DIR)
    protocol_data = {p: [] for p in protocols}
    pair_names    = {p: [] for p in protocols}
    n_loaded      = {p: 0  for p in protocols}

    for pair_dir in sorted(search_dir.iterdir()):
        if not pair_dir.is_dir():
            continue
        parts = pair_dir.name.split("-")
        if len(parts) != 2:
            continue
        try:
            pre_gid, post_gid = int(parts[0]), int(parts[1])
        except ValueError:
            continue

        if max_pairs and all(n_loaded[p] >= max_pairs for p in protocols):
            break

        basis = _base._load_basis(pre_gid, post_gid, L5_BASIS_DIR)
        if not basis:
            continue

        for proto in protocols:
            if max_pairs and n_loaded[proto] >= max_pairs:
                continue
            pkl_path = pair_dir / proto / "simulation_traces.pkl"
            if not pkl_path.exists():
                continue
            try:
                pd_item = _load_pkl_weighted(
                    str(pkl_path),
                    needs_threshold_traces=True,
                    include_raw_cai=False,
                    include_base_threshold_traces=True,
                    include_cpre_cpost=False,
                )
            except Exception as e:
                print(f"  Skipping {pair_dir.name}/{proto}: {e}")
                continue
            if pd_item["cai_nmda"].shape[0] != basis["n_syn"]:
                continue
            pd_item.update(basis)
            protocol_data[proto].append(pd_item)
            pair_names[proto].append((pre_gid, post_gid))
            n_loaded[proto] += 1

        # progress every 10 pairs across any protocol
        if any(n % 10 == 0 and n > 0 for n in n_loaded.values()):
            total = sum(n_loaded.values())
            if total % (10 * len(protocols)) == 0:
                print(f"  loaded {pair_dir.name} ({total} pair-protocols so far)")

    total = sum(n_loaded.values())
    print(f"  done — {total} pair-protocols loaded")
    return protocol_data, pair_names


# ──────────────────────────────────────────────────────────────────────────────
# JAX batched integration — vmap over (pairs × synapses), scan over time
# ──────────────────────────────────────────────────────────────────────────────

def make_batch_integrator(params_x, model):
    """
    Build a JIT-compiled function that integrates ALL synapses for ALL pairs
    of one protocol in a single call.

        batch_integrate(cd) -> (eff_all, rho_all)
            cd      : collated dict from collate_protocol_to_jax
            eff_all : (n_pairs, max_syn, T)  effcai traces
            rho_all : (n_pairs, max_syn, T)  rho traces
    """
    params    = model.unpack_params(jnp.array(params_x))
    gb, aux   = params
    tau_effca = aux["tau_effca"]
    gamma_d   = gb["gamma_d"]
    gamma_p   = gb["gamma_p"]

    def _per_syn(cai_s, rho0_s, theta_d_s, theta_p_s, dt_arr):
        """Scan over time for one synapse — returns (eff_trace, rho_trace) (T,)."""
        def step(carry, ins):
            eff, rho = carry
            ca_t, dt = ins
            decay   = jnp.exp(-dt / tau_effca)
            factor  = tau_effca * (1.0 - decay)
            eff_new = jnp.where(dt > 0,
                                eff * decay + (ca_t - _MIN_CA) * factor, eff)
            dep = jnp.where(eff_new > theta_d_s, 1.0, 0.0)
            pot = jnp.where(eff_new > theta_p_s, 1.0, 0.0)
            drho = (
                -rho * (1 - rho) * (0.5 - rho)
                + pot * gamma_p * (1 - rho)
                - dep * gamma_d * rho
            ) / 70000.0
            rho_new = jnp.where(dt > 0,
                                jnp.clip(rho + dt * drho, 0.0, 1.0), rho)
            return (eff_new, rho_new), (eff_new, rho_new)

        _, (eff_tr, rho_tr) = jax.lax.scan(
            step, (0.0, rho0_s), (cai_s, dt_arr)
        )
        return eff_tr, rho_tr

    # vmap: over synapses within one pair
    _per_pair_syns = jax.vmap(_per_syn, in_axes=(0, 0, 0, 0, None))

    def _per_pair(cai_vdcc_p,      # (max_syn, T)
                  t_p,             # (T,)
                  rho0_p,          # (max_syn,)
                  cai_pre_vdcc_p,  # (max_syn, T_pre)
                  cai_post_vdcc_p, # (max_syn, T_post)
                  t_pre_p,         # (T_pre,)
                  t_post_p,        # (T_post,)
                  is_apical_p,     # (max_syn,)
                  ):
        dt_pre  = jnp.maximum(t_pre_p[1]  - t_pre_p[0],  1e-6)
        dt_post = jnp.maximum(t_post_p[1] - t_post_p[0], 1e-6)

        c_pre  = jax.vmap(
            lambda cv: _jax_peak_effcai_zoh(cv, tau_effca, dt_pre,  _MIN_CA)
        )(cai_pre_vdcc_p)
        c_post = jax.vmap(
            lambda cv: _jax_peak_effcai_zoh(cv, tau_effca, dt_post, _MIN_CA)
        )(cai_post_vdcc_p)

        theta_d = jnp.where(is_apical_p,
            gb["a20"] * c_pre + gb["a21"] * c_post,
            gb["a00"] * c_pre + gb["a01"] * c_post)
        theta_p = jnp.where(is_apical_p,
            gb["a30"] * c_pre + gb["a31"] * c_post,
            gb["a10"] * c_pre + gb["a11"] * c_post)

        dt_arr = jnp.diff(t_p, prepend=t_p[0])
        dt_arr = jnp.where(dt_arr <= 0, 1e-6, dt_arr)

        return _per_pair_syns(cai_vdcc_p, rho0_p, theta_d, theta_p, dt_arr)

    # vmap: over pairs
    _batch = jax.vmap(_per_pair, in_axes=(0, 0, 0, 0, 0, 0, 0, 0))

    @jax.jit
    def batch_integrate(cd):
        return _batch(
            cd["cai_vdcc"],
            cd["t"],
            cd["rho0"],
            cd["cai_pre_vdcc"],
            cd["cai_post_vdcc"],
            cd["t_pre"],
            cd["t_post"],
            cd["is_apical"],
        )

    return batch_integrate


# ──────────────────────────────────────────────────────────────────────────────
# Orchestration
# ──────────────────────────────────────────────────────────────────────────────

def collect_jax_rho(params_dict, protocols, max_pairs=None,
                    trace_dir=None, dt_step=10):
    """
    Load data, run JAX batch integration, return CSV rows + per-pair trace dict.

    Returns:
        rows        — list of dicts (CSV rows, same columns as rho_v17.csv)
        pair_traces — dict  { (pregid, postgid, proto): {t, rho, eff, valid, ...} }
    """
    model = GBVdccOnlyModel()
    dp    = dict(model.DEFAULT_PARAMS)
    dp.update(params_dict)
    params_x = np.array([dp[n] for n, *_ in model.FIT_PARAMS])

    print("Loading data ...")
    protocol_data, pair_names = load_data_with_names(
        protocols, max_pairs=max_pairs, trace_dir=trace_dir
    )

    integrator = None   # built lazily (shape-dependent JIT)
    rows        = []
    pair_traces = {}

    for proto in protocols:
        pairs_list = protocol_data.get(proto, [])
        names      = pair_names.get(proto, [])
        if not pairs_list:
            print(f"  No data for {proto}")
            continue

        try:
            freq_str, dt_str = proto.split("_")
            freq  = float(freq_str.replace("Hz", ""))
            delay = float(dt_str.replace("ms", ""))
        except ValueError:
            freq, delay = 0.0, 0.0

        print(f"  Collating {proto} ({len(pairs_list)} pairs) ...")
        cd = collate_protocol_to_jax(
            pairs_list,
            dt_step=dt_step,
            include_raw_cai=False,
            include_base_threshold_traces=True,
            include_cpre_cpost=False,
        )
        if cd is None:
            continue

        print(f"  JIT+vmap integrate {proto} "
              f"shape=({cd['cai_vdcc'].shape}) ...")

        if integrator is None:
            integrator = make_batch_integrator(params_x, model)

        eff_all, rho_all = integrator(cd)
        # eff_all, rho_all : (n_pairs, max_syn, T)
        eff_all  = np.asarray(eff_all)
        rho_all  = np.asarray(rho_all)
        rho0_all = np.asarray(cd["rho0"])    # (n_pairs, max_syn)
        valid_all = np.asarray(cd["valid"])  # (n_pairs, max_syn)
        t_all    = np.asarray(cd["t"])       # (n_pairs, T)

        n_pairs, max_syn, T = rho_all.shape
        for i in range(n_pairs):
            pre_gid, post_gid = names[i]
            for s in range(max_syn):
                if not valid_all[i, s]:
                    continue
                rho0_raw   = float(rho0_all[i, s])
                rho_fin_raw = float(rho_all[i, s, -1])
                rows.append({
                    "pregid":          pre_gid,
                    "postgid":         post_gid,
                    "freq":            freq,
                    "delay":           delay,
                    "syn_id":          s,
                    "initial_rho_raw": rho0_raw,
                    "final_rho_raw":   rho_fin_raw,
                    "initial_rho":     1 if rho0_raw   >= 0.5 else 0,
                    "final_rho":       1 if rho_fin_raw >= 0.5 else 0,
                })

            # store full traces for plotting
            n_syn = int(valid_all[i].sum())
            pair_traces[(pre_gid, post_gid, proto)] = dict(
                t=t_all[i],
                rho=rho_all[i, :n_syn],
                eff=eff_all[i, :n_syn],
                rho0=rho0_all[i, :n_syn],
                is_apical=np.asarray(cd["is_apical"])[i, :n_syn],
                theta_d=None,   # not stored to save memory
                theta_p=None,
            )

        print(f"  done {proto}")

    return rows, pair_traces


# ──────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────────────────────────────────────

def _transition_counts(df, delay):
    sub = df[df["delay"] == delay]
    return {
        "0→0": int(((sub["initial_rho"] == 0) & (sub["final_rho"] == 0)).sum()),
        "0→1": int(((sub["initial_rho"] == 0) & (sub["final_rho"] == 1)).sum()),
        "1→0": int(((sub["initial_rho"] == 1) & (sub["final_rho"] == 0)).sum()),
        "1→1": int(((sub["initial_rho"] == 1) & (sub["final_rho"] == 1)).sum()),
    }


def plot_summary(df, neuron_df=None, out_dir="."):
    """Rho transition bars + potentiated-fraction curve."""
    delays   = sorted(df["delay"].unique())
    n_delays = len(delays)

    fig, axes = plt.subplots(2, n_delays, figsize=(5 * n_delays, 8))
    if n_delays == 1:
        axes = axes.reshape(2, 1)

    for col, delay in enumerate(delays):
        ax   = axes[0, col]
        tc_j = _transition_counts(df, delay)
        lbls = list(tc_j.keys())
        v_j  = [tc_j[k] for k in lbls]
        x    = np.arange(len(lbls))
        w    = 0.35

        if neuron_df is not None:
            tc_n = _transition_counts(neuron_df, delay)
            v_n  = [tc_n.get(k, 0) for k in lbls]
            ax.bar(x - w/2, v_j, w, label="JAX",    color="steelblue")
            ax.bar(x + w/2, v_n, w, label="NEURON",  color="coral")
            ax.legend(fontsize=8)
        else:
            ax.bar(x, v_j, color="steelblue")

        ax.set_xticks(x); ax.set_xticklabels(lbls)
        ax.set_ylabel("# synapses")
        ax.set_title(f"delay={delay:+.0f} ms")
        for xi, v in zip(x, v_j):
            ax.text(xi - (w/2 if neuron_df is not None else 0), v + 0.5,
                    str(v), ha="center", va="bottom", fontsize=8)

    ax_r = axes[1, 0]
    ax_r.plot(delays,
              [df[df["delay"] == d]["initial_rho"].mean() for d in delays],
              "s--", color="gray",      label="initial rho")
    ax_r.plot(delays,
              [df[df["delay"] == d]["final_rho"].mean()   for d in delays],
              "o-",  color="steelblue", label="JAX final rho")
    if neuron_df is not None:
        nd = neuron_df
        d_neu = [d for d in delays if d in nd["delay"].values]
        ax_r.plot(d_neu,
                  [nd[nd["delay"] == d]["final_rho"].mean() for d in d_neu],
                  "^-", color="coral", label="NEURON final rho")
    ax_r.set_xlabel("Delay (ms)"); ax_r.set_ylabel("Mean rho (potentiated fraction)")
    ax_r.legend(fontsize=8); ax_r.set_ylim(0, 1)
    for col in range(1, n_delays):
        axes[1, col].set_visible(False)

    fig.suptitle("JAX gb_vdcc_only — rho summary", fontsize=13)
    fig.tight_layout()
    out = os.path.join(out_dir, "jax_rho_summary.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"Saved → {out}")


def plot_pair_traces(pair_traces, pre_gid, post_gid, protocols, out_dir="."):
    """Per-synapse effcai + rho(t); one fig per protocol."""
    for proto in protocols:
        key = (pre_gid, post_gid, proto)
        if key not in pair_traces:
            print(f"  No data for {pre_gid}-{post_gid}/{proto}")
            continue

        tr    = pair_traces[key]
        t     = tr["t"]
        n_syn = tr["rho"].shape[0]
        fig, axes = plt.subplots(n_syn, 2, figsize=(12, 3 * n_syn), squeeze=False)

        for s in range(n_syn):
            loc = "apical" if tr["is_apical"][s] else "basal"
            rho0_v = float(tr["rho0"][s])
            rho_f  = float(tr["rho"][s, -1])

            ax_e = axes[s, 0]
            ax_e.plot(t, tr["eff"][s], color="royalblue", lw=1.2)
            ax_e.set_ylabel("effcai"); ax_e.set_xlabel("time (ms)")
            ax_e.set_title(f"syn {s} ({loc})")

            ax_r = axes[s, 1]
            ax_r.plot(t, tr["rho"][s], color="seagreen", lw=1.4)
            ax_r.axhline(0.5, color="gray", ls=":", lw=0.8)
            ax_r.set_ylim(-0.05, 1.05)
            ax_r.set_ylabel("rho"); ax_r.set_xlabel("time (ms)")
            ax_r.set_title(f"rho0={rho0_v:.3f}  →  {rho_f:.3f}")

        fig.suptitle(f"{pre_gid}–{post_gid}  /  {proto}", fontsize=12)
        fig.tight_layout()
        fname = f"jax_rho_traces_{pre_gid}-{post_gid}_{proto}.png"
        out   = os.path.join(out_dir, fname)
        fig.savefig(out, dpi=130); plt.close(fig)
        print(f"Saved → {out}")


def plot_all_traces(pair_traces, protocols, out_dir="."):
    seen = sorted({(p, q) for (p, q, _) in pair_traces})
    for p, q in seen:
        plot_pair_traces(pair_traces, p, q, protocols, out_dir=out_dir)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="JAX rho-trace collector/plotter for gb_vdcc_only (batched)"
    )
    parser.add_argument("--params",     required=True,
                        help="JSON parameter file (gb_vdcc_only best_params_*.json)")
    parser.add_argument("--protocols",  nargs="+", default=["10Hz_10ms", "10Hz_-10ms"])
    parser.add_argument("--max-pairs",  type=int, default=None)
    parser.add_argument("--dt-step",    type=int, default=10,
                        help="Downsample factor for time axis (default 10 ≈ 0.25 ms)")
    parser.add_argument("--output",     default="rho_jax.csv",
                        help="Output CSV filename (same columns as rho_v17.csv)")
    parser.add_argument("--plot",       choices=["summary", "traces", "both", "none"],
                        default="summary")
    parser.add_argument("--out-dir",    default=".",
                        help="Directory for output figures")
    parser.add_argument("--neuron-csv", default=None,
                        help="Optional NEURON rho CSV (e.g. rho_v17.csv) for overlay")
    parser.add_argument("--trace-dir",  default=None,
                        help="Override trace directory")
    parser.add_argument("--pairs",      nargs="+", default=None,
                        help="Subset of pairs for trace plots (e.g. 180164-197248)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.params) as f:
        params_dict = json.load(f)

    print(f"\nParams: {args.params}")
    for k, v in params_dict.items():
        print(f"  {k:>12} = {v:.6f}")

    rows, pair_traces = collect_jax_rho(
        params_dict,
        protocols=args.protocols,
        max_pairs=args.max_pairs,
        trace_dir=args.trace_dir,
        dt_step=args.dt_step,
    )

    if not rows:
        print("No data collected."); return

    import pandas as pd
    df = pd.DataFrame(rows, columns=[
        "pregid", "postgid", "freq", "delay", "syn_id",
        "initial_rho_raw", "final_rho_raw", "initial_rho", "final_rho",
    ])
    df.to_csv(args.output, index=False)
    print(f"\nWrote {len(df)} rows → {args.output}")

    print("\nTransition counts (JAX):")
    for delay in sorted(df["delay"].unique()):
        tc = _transition_counts(df, delay)
        print(f"  delay={delay:+.0f}ms: " + "  ".join(f"{k}: {v}" for k, v in tc.items()))

    print("\nMean rho:")
    print(df.groupby("delay")[["initial_rho", "final_rho"]].mean().to_string())

    neuron_df = None
    if args.neuron_csv and os.path.exists(args.neuron_csv):
        neuron_df = pd.read_csv(args.neuron_csv)
        print(f"\nNEURON CSV loaded: {args.neuron_csv}  ({len(neuron_df)} rows)")
        print("\nTransition counts (NEURON):")
        for delay in sorted(neuron_df["delay"].unique()):
            tc = _transition_counts(neuron_df, delay)
            print(f"  delay={delay:+.0f}ms: " + "  ".join(f"{k}: {v}" for k, v in tc.items()))

    if args.plot in ("summary", "both"):
        plot_summary(df, neuron_df=neuron_df, out_dir=args.out_dir)

    if args.plot in ("traces", "both"):
        if args.pairs:
            for pair_str in args.pairs:
                try:
                    pre_gid, post_gid = map(int, pair_str.split("-"))
                    plot_pair_traces(pair_traces, pre_gid, post_gid,
                                     args.protocols, out_dir=args.out_dir)
                except ValueError:
                    print(f"  Bad pair spec: {pair_str}")
        else:
            plot_all_traces(pair_traces, args.protocols, out_dir=args.out_dir)


if __name__ == "__main__":
    main()

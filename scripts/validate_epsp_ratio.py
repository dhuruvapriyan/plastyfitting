#!/usr/bin/env python3
"""
Validate JAX EPSP ratio prediction against NEURON ground truth.

Three methods compared per simulation:

  A) NEURON voltage  — peak voltage deflection in 100ms window after each C01/C02
                       spike (same as Experiment.compute_epsp_ratio in ephysutils).

  B) Basis + NEURON rho — linear superposition model using the stored rho_GB trace
                           to get initial/final binary state, combined with
                           pre-measured singleton EPSP amplitudes from basis_*.csv.
                           Tests: how good is the linear basis approximation?

  C) Basis + JAX rho   — same linear model but rho_final comes from our numpy
                          re-implementation of gb_only's scan_step.
                          Tests: does our JAX rho simulation match NEURON?

Usage:
    python scripts/validate_epsp_ratio.py
    python scripts/validate_epsp_ratio.py --pair 180164-197248 --protocol 10Hz_5ms
    python scripts/validate_epsp_ratio.py --all-pairs --protocols 10Hz_5ms 10Hz_-10ms
"""

import sys, os, argparse, glob, pickle
import numpy as np
import pandas as pd

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACE_DIR    = os.path.join(BASE_DIR, "trace_results/CHINDEMI_PARAMS")
BASIS_DIR    = os.path.join(BASE_DIR, "basis_results")

# Experiment parameters (from L5TTPC_L5TTPC.yaml / plot_stdp_curves_full.py)
C01_DURATION_MIN    = 4.0
C02_DURATION_MIN    = 4.0
TEST_PULSE_PERIOD_S = 4.0
N_EPSP              = 60
EPSP_WINDOW_MS      = 100.0
SPIKE_THRESHOLD     = -30.0   # mV

# CHINDEMI_PARAMS
CHINDEMI = {
    "gamma_d": 101.5, "gamma_p": 216.2,
    "a00": 1.002, "a01": 1.954,
    "a10": 1.159, "a11": 2.483,
    "a20": 1.127, "a21": 2.456,
    "a30": 5.236, "a31": 1.782,
    "tau_eff": 278.318,
}
TAU_EFFCA = CHINDEMI["tau_eff"]
MIN_CA    = 70e-6


# ── Method A: NEURON voltage-based EPSP ratio ────────────────────────────────

def get_epsp_vector(t, v, spikes, window=EPSP_WINDOW_MS):
    epsps = np.zeros(len(spikes))
    for i, s in enumerate(spikes):
        w0 = np.searchsorted(t, s)
        w1 = np.searchsorted(t, s + window)
        v_baseline = v[w0]
        peak_idx   = np.argmax(v[w0:w1]) + w0
        v_peak     = v[peak_idx]
        if v_peak > -SPIKE_THRESHOLD:
            raise RuntimeError(f"Post-synaptic spike at t={s:.0f}ms")
        epsps[i] = v_peak - v_baseline
    return epsps


def epsp_ratio_voltage(sim_data, c01dur=C01_DURATION_MIN, c02dur=C02_DURATION_MIN,
                       period=TEST_PULSE_PERIOD_S, n=N_EPSP):
    t      = np.asarray(sim_data["t"],         dtype=np.float64)
    v      = np.asarray(sim_data["v"],         dtype=np.float64)
    spikes = np.asarray(sim_data["prespikes"],  dtype=np.float64)
    period_ms  = period * 1000.0
    n_c01      = int(c01dur * 60 * 1000 / period_ms)
    n_c02      = int(c02dur * 60 * 1000 / period_ms)
    c01_spikes = spikes[:n_c01]
    c02_spikes = spikes[-n_c02:]
    epsps_c01  = get_epsp_vector(t, v, c01_spikes)
    epsps_c02  = get_epsp_vector(t, v, c02_spikes)
    epsp_before = np.mean(epsps_c01[-n:])
    epsp_after  = np.mean(epsps_c02[-n:])
    return epsp_before, epsp_after, epsp_after / epsp_before


# ── Basis CSV loader ──────────────────────────────────────────────────────────

def load_basis(pre_gid, post_gid, basis_dir=BASIS_DIR):
    csv_path = os.path.join(basis_dir, f"basis_{pre_gid}_{post_gid}.csv")
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    configs  = df["config"].apply(lambda x: [int(i) for i in x.split(",")])
    n_syn    = len(configs.iloc[0])
    base_row = df[configs.apply(lambda x: sum(x) == 0)]
    if base_row.empty:
        return None
    bmean = float(base_row["mean"].values[0])
    sm    = np.zeros(n_syn)
    for i in range(n_syn):
        row = df[configs.apply(lambda x: sum(x) == 1 and x[i] == 1)]
        if row.empty:
            return None
        sm[i] = float(row["mean"].values[0])
    return {"bmean": bmean, "sm": sm, "n_syn": n_syn}


# ── Method B/C: basis-model EPSP ratio from rho arrays ───────────────────────

def epsp_ratio_basis(basis, rho_initial, rho_final):
    bmean, sm = basis["bmean"], basis["sm"]
    eb = bmean + np.sum(np.where(rho_initial >= 0.5, sm - bmean, 0.0))
    ea = bmean + np.sum(np.where(rho_final   >= 0.5, sm - bmean, 0.0))
    return eb, ea, ea / eb if eb > 0 else np.nan


# ── Method C: numpy rho simulation (mirrors gb_only scan_step exactly) ───────

def simulate_rho_numpy(sim_data, sp, params=CHINDEMI):
    t       = np.asarray(sim_data["t"],      dtype=np.float64)
    cai_raw = np.asarray(sim_data["cai_CR"], dtype=np.float64)
    if cai_raw.shape[0] == len(t):
        cai_raw = cai_raw.T        # → (n_syn, T)
    rho_gb  = np.asarray(sim_data["rho_GB"], dtype=np.float64)
    if rho_gb.shape[0] == len(t):
        rho_gb  = rho_gb.T         # → (n_syn, T)

    dt         = np.diff(t, prepend=t[0]); dt[0] = dt[1]
    td         = np.array(sp["theta_d_GB"])
    tp_arr     = np.array(sp["theta_p_GB"])
    gd, gp     = params["gamma_d"], params["gamma_p"]
    tau        = params["tau_eff"]
    n_syn      = cai_raw.shape[0]
    rho_final  = np.zeros(n_syn)

    for s in range(n_syn):
        rho     = rho_gb[s, 0]
        eff     = 0.0
        ca_prev = cai_raw[s, 0]
        for i in range(1, len(t)):
            d  = dt[i]; ca = cai_raw[s, i]
            if d > 0:
                f0    = max(0., ca_prev - MIN_CA)
                f1    = max(0., ca      - MIN_CA)
                decay = np.exp(-d / tau)
                slope = (f1 - f0) / d
                eff_new = (eff * decay
                           + f0  * tau * (1. - decay)
                           + slope * (tau*d - tau**2*(1.-decay)))
            else:
                eff_new = eff
            ca_prev = ca
            # dep/pot from START-of-interval effcai (NEURON convention)
            pot  = 1. if eff > tp_arr[s] else 0.
            dep  = 1. if eff > td[s]     else 0.
            drho = (-rho*(1-rho)*(0.5-rho) + pot*gp*(1-rho) - dep*gd*rho) / 70000.
            if d > 0:
                rho = np.clip(rho + d * drho, 0., 1.)
            eff = eff_new
        rho_final[s] = rho
    return rho_final


# ── Main validation ───────────────────────────────────────────────────────────

def validate_pair_protocol(pair_name, protocol, verbose=True, skip_jax_sim=False):
    pkl_path = os.path.join(TRACE_DIR, pair_name, protocol, "simulation_traces.pkl")
    if not os.path.exists(pkl_path):
        print(f"  Not found: {pkl_path}")
        return None
    pre_gid, post_gid = [int(x) for x in pair_name.split("-")]
    basis = load_basis(pre_gid, post_gid)
    if basis is None:
        print(f"  No basis CSV for {pair_name}")
        return None

    with open(pkl_path, "rb") as f:
        sim_data = pickle.load(f)
    sp = sim_data["synprop"]

    rho_gb = np.asarray(sim_data["rho_GB"], dtype=np.float64)
    t      = np.asarray(sim_data["t"],      dtype=np.float64)
    if rho_gb.shape[0] == len(t):
        rho_gb = rho_gb.T
    rho_initial = rho_gb[:, 0]
    rho_neuron  = rho_gb[:, -1]

    # A
    try:
        eb_v, ea_v, ratio_v = epsp_ratio_voltage(sim_data)
    except Exception as e:
        print(f"  Voltage method failed: {e}")
        ratio_v = eb_v = ea_v = np.nan
    # B
    eb_bn, ea_bn, ratio_bn = epsp_ratio_basis(basis, rho_initial, rho_neuron)
    # C (optional — slow for large batches)
    if not skip_jax_sim:
        rho_jax = simulate_rho_numpy(sim_data, sp)
        eb_bj, ea_bj, ratio_bj = epsp_ratio_basis(basis, rho_initial, rho_jax)
    else:
        rho_jax = None
        eb_bj = ea_bj = ratio_bj = float("nan")

    if verbose:
        print(f"\n{'─'*70}")
        print(f"{pair_name} / {protocol}")
        print(f"{'─'*70}")
        print(f"  n_syn={len(rho_initial)}")
        print(f"  rho_initial : {np.round(rho_initial,3)}")
        print(f"  rho_neuron  : {np.round(rho_neuron,3)}")
        if rho_jax is not None:
            print(f"  rho_jax     : {np.round(rho_jax,3)}")
        print(f"  Δrho(j-n)   : {np.round(rho_jax-rho_neuron,4)}")
        print()
        print(f"  Method A (voltage):            before={eb_v:.4f}  after={ea_v:.4f}  ratio={ratio_v:.4f}")
        print(f"  Method B (basis+NEURON rho):   before={eb_bn:.4f}  after={ea_bn:.4f}  ratio={ratio_bn:.4f}")
        print(f"  Method C (basis+JAX rho):      before={eb_bj:.4f}  after={ea_bj:.4f}  ratio={ratio_bj:.4f}")
        print()
        print(f"  B vs A (basis model error):  Δ={abs(ratio_bn-ratio_v):.4f}  ({100*abs(ratio_bn-ratio_v)/max(abs(ratio_v),1e-9):.1f}%)")
        print(f"  C vs A (full JAX error):     Δ={abs(ratio_bj-ratio_v):.4f}  ({100*abs(ratio_bj-ratio_v)/max(abs(ratio_v),1e-9):.1f}%)")
        print(f"  C vs B (rho sim accuracy):   Δ={abs(ratio_bj-ratio_bn):.4f}  ({100*abs(ratio_bj-ratio_bn)/max(abs(ratio_bn),1e-9):.1f}%)")

    return {
        "pair": pair_name, "protocol": protocol,
        "ratio_voltage": ratio_v, "ratio_basis_neuron": ratio_bn, "ratio_basis_jax": ratio_bj,
        "epsp_before_v": eb_v, "epsp_after_v": ea_v,
        "rho_initial": rho_initial.tolist(),
        "rho_neuron":  rho_neuron.tolist(),
        "rho_jax":     rho_jax.tolist() if rho_jax is not None else [],
    }


def run_all(protocols=None, max_pairs=None, skip_jax_sim=True):
    all_pairs = sorted([
        d for d in os.listdir(TRACE_DIR)
        if os.path.isdir(os.path.join(TRACE_DIR, d)) and "-" in d
    ])
    if max_pairs:
        all_pairs = all_pairs[:max_pairs]
    results = []
    for pair in all_pairs:
        pair_dir = os.path.join(TRACE_DIR, pair)
        protos = protocols or sorted(os.listdir(pair_dir))
        for proto in protos:
            if not os.path.isdir(os.path.join(pair_dir, proto)):
                continue
            r = validate_pair_protocol(pair, proto, verbose=False, skip_jax_sim=skip_jax_sim)
            if r:
                results.append(r)
                print(f"  {pair}/{proto}: A={r['ratio_voltage']:.3f}  "
                      f"B={r['ratio_basis_neuron']:.3f}  C={r['ratio_basis_jax']:.3f}")
    if not results:
        print("No results."); return
    df = pd.DataFrame(results)
    df["B_vs_A"] = (df["ratio_basis_neuron"] - df["ratio_voltage"]).abs()
    df["C_vs_A"] = (df["ratio_basis_jax"]    - df["ratio_voltage"]).abs()
    df["C_vs_B"] = (df["ratio_basis_jax"]    - df["ratio_basis_neuron"]).abs()
    print(f"\n{'='*60}")
    print(f"SUMMARY  ({len(df)} sims, {df['pair'].nunique()} pairs)")
    print(f"{'='*60}")
    print(f"  B vs A (basis model vs voltage):  mean={df['B_vs_A'].mean():.4f}  max={df['B_vs_A'].max():.4f}")
    print(f"  C vs A (full JAX vs voltage):     mean={df['C_vs_A'].mean():.4f}  max={df['C_vs_A'].max():.4f}")
    print(f"  C vs B (JAX rho vs NEURON rho):   mean={df['C_vs_B'].mean():.4f}  max={df['C_vs_B'].max():.4f}")
    print(f"\nPer-protocol breakdown (B vs A | C vs A):")
    print(df.groupby("protocol")[["B_vs_A","C_vs_A"]].mean().round(4).to_string())
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair",       default=None)
    parser.add_argument("--protocol",   default="10Hz_5ms")
    parser.add_argument("--all-pairs",  action="store_true")
    parser.add_argument("--protocols",  nargs="+", default=None)
    parser.add_argument("--max-pairs",  type=int, default=None)
    parser.add_argument("--run-jax-sim", action="store_true",
                        help="Also run Method C (numpy rho sim). Slow for batches.")
    args = parser.parse_args()
    skip_jax_sim = not args.run_jax_sim

    if args.all_pairs or args.protocols:
        run_all(protocols=args.protocols, max_pairs=args.max_pairs, skip_jax_sim=skip_jax_sim)
    else:
        pair = args.pair or "180164-197248"
        validate_pair_protocol(pair, args.protocol, skip_jax_sim=skip_jax_sim)

#!/usr/bin/env python3
"""Validate JAX ZOH effcai reconstruction against NEURON effcai_GB trace.

For each synapse in a simulation_traces.pkl, replays the ZOH effcai ODE:
    effcai[n+1] = effcai[n]*exp(-dt/tau) + max(cai-min_ca,0)*tau*(1-exp(-dt/tau))
using the recorded cai_CR and CVODE t array, then compares point-by-point
against the stored effcai_GB variable.

Usage:
    python scripts/validate_effcai.py
    python scripts/validate_effcai.py --pair 180164-197248 --protocol 10Hz_5ms
    python scripts/validate_effcai.py --all-pairs --protocol 10Hz_10ms
"""

import argparse
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

CHINDEMI_PARAMS = {
    "gamma_d_GB": 101.5,
    "gamma_p_GB": 216.2,
    "a00": 1.002, "a01": 1.954,
    "a10": 1.159, "a11": 2.483,
    "a20": 1.127, "a21": 2.456,
    "a30": 5.236, "a31": 1.782,
    "tau_effca": 278.318,   # tau_effca_GB_GluSynapse
}

BASE_DIR = "/project/rrg-emuller/dhuruva/plastyfitting"
TRACE_DIR = BASE_DIR + "/trace_results/CHINDEMI_PARAMS"
CAI_REST  = 70e-6   # min_ca_CR in GluSynapse.mod


def pwl_effcai_numpy(cai_arr, dt_arr, tau, min_ca=CAI_REST):
    """Piecewise-linear effcai integrator — matches NEURON CVODE exactly.

    Replicates the NEURON GluSynapse.mod ODE integrated under CVODE:
        effcai' = -effcai/tau + (cai - min_ca)

    Since CVODE records calcium at adaptive checkpoints, the input between
    consecutive checkpoints follows a linear ramp (not a step).  Using the
    exact ZOH solution (constant input = endpoint value) causes ~1% error
    on large adaptive steps.  The piecewise-linear (PWL) solution is exact
    for a linearly-varying input:

        effcai[n+1] = effcai[n]*exp(-dt/tau)
                    + f0*tau*(1-exp(-dt/tau))
                    + (f1-f0)/dt * (tau*dt - tau^2*(1-exp(-dt/tau)))

    where f0 = max(cai[n]-min_ca, 0) and f1 = max(cai[n+1]-min_ca, 0).
    This reduces peak error from ~1% to ~0.01%.

    Returns the full effcai trace (same length as cai_arr).
    """
    n = len(cai_arr)
    out = np.zeros(n, dtype=np.float64)
    effcai = 0.0
    for i in range(1, n):
        dt = dt_arr[i - 1]
        if dt <= 0:
            out[i] = effcai
            continue
        f0    = max(0.0, cai_arr[i - 1] - min_ca)
        f1    = max(0.0, cai_arr[i]     - min_ca)
        slope = (f1 - f0) / dt
        decay = np.exp(-dt / tau)
        effcai = (effcai * decay
                  + f0    * tau * (1.0 - decay)
                  + slope * (tau * dt - tau**2 * (1.0 - decay)))
        out[i] = effcai
    return out


def validate_pair(pkl_path, tau, plot_out=None, max_syn=4, zoom_ms=2000.0):
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)

    t   = np.asarray(d['t'],         dtype=np.float64)  # (T,)
    cai = np.asarray(d['cai_CR'],    dtype=np.float64)
    eff = np.asarray(d['effcai_GB'], dtype=np.float64)

    # old format is (T, n_syn) — transpose to (n_syn, T)
    if cai.ndim == 2 and cai.shape[0] == len(t):
        cai = cai.T
        eff = eff.T

    n_syn  = cai.shape[0]
    dt_arr = np.diff(t)  # length T-1

    results = []

    for s in range(min(n_syn, max_syn)):
        eff_jax = pwl_effcai_numpy(cai[s], dt_arr, tau)
        eff_neu = eff[s]

        err      = eff_jax - eff_neu
        rmse     = float(np.sqrt(np.mean(err**2)))
        max_err  = float(np.max(np.abs(err)))
        peak_neu = float(np.max(eff_neu))
        peak_jax = float(np.max(eff_jax))
        rel_peak = abs(peak_jax - peak_neu) / (peak_neu + 1e-12)

        results.append(dict(syn=s, rmse=rmse, max_err=max_err,
                            peak_neu=peak_neu, peak_jax=peak_jax,
                            rel_peak_err=rel_peak))

        print(f"  syn {s}: RMSE={rmse:.3e}  max|err|={max_err:.3e}"
              f"  peak NEURON={peak_neu:.6f}  JAX={peak_jax:.6f}"
              f"  rel_peak_err={rel_peak*100:.4f}%")

    if plot_out is not None:
        n_plot = min(n_syn, max_syn)
        fig, axes = plt.subplots(n_plot, 2, figsize=(14, 3 * n_plot), squeeze=False)

        mask = t <= zoom_ms

        for s in range(n_plot):
            eff_jax = pwl_effcai_numpy(cai[s], dt_arr, tau)
            eff_neu = eff[s]

            ax_trace, ax_err = axes[s]

            ax_trace.plot(t[mask], eff_neu[mask], lw=1.0,
                          label='NEURON effcai_GB', color='steelblue')
            ax_trace.plot(t[mask], eff_jax[mask], lw=0.8, ls='--',
                          label='JAX ZOH', color='tomato', alpha=0.85)
            ax_trace.set_title(f"syn {s} — first {zoom_ms:.0f} ms")
            ax_trace.set_xlabel("t (ms)")
            ax_trace.set_ylabel("effcai")
            ax_trace.legend(fontsize=7)

            ax_err.plot(t[mask], (eff_jax - eff_neu)[mask], lw=0.7, color='purple')
            ax_err.axhline(0, color='k', lw=0.5)
            ax_err.set_title(f"syn {s} — JAX − NEURON error")
            ax_err.set_xlabel("t (ms)")
            ax_err.set_ylabel("Δeffcai")

        fig.suptitle(f"{pkl_path.parent.parent.name} / {pkl_path.parent.name}  τ={tau} ms",
                     fontsize=10)
        fig.tight_layout()
        fig.savefig(plot_out, dpi=120)
        plt.close(fig)
        print(f"  → plot saved: {plot_out}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Validate JAX ZOH effcai vs NEURON effcai_GB")
    parser.add_argument("--pair",       type=str,   default="180164-197248")
    parser.add_argument("--protocol",   type=str,   default="10Hz_5ms")
    parser.add_argument("--all-pairs",  action="store_true",
                        help="Validate all pairs in TRACE_DIR for the given protocol")
    parser.add_argument("--tau",        type=float, default=CHINDEMI_PARAMS["tau_effca"])
    parser.add_argument("--max-syn",    type=int,   default=4)
    parser.add_argument("--zoom-ms",    type=float, default=5000.0)
    parser.add_argument("--no-plot",    action="store_true")
    args = parser.parse_args()

    print(f"tau_effca = {args.tau} ms  (min_ca = {CAI_REST*1e6:.1f} µM)\n")

    trace_root = Path(TRACE_DIR)

    if args.all_pairs:
        pair_dirs = sorted(d for d in trace_root.iterdir() if d.is_dir())
    else:
        pair_dirs = [trace_root / args.pair]

    all_results = []

    for pair_dir in pair_dirs:
        pkl_path = pair_dir / args.protocol / "simulation_traces.pkl"
        if not pkl_path.exists():
            print(f"SKIP {pair_dir.name}: no {args.protocol}/simulation_traces.pkl")
            continue

        print(f"=== {pair_dir.name} / {args.protocol} ===")
        plot_out = (None if args.no_plot
                    else f"validate_effcai_{pair_dir.name}_{args.protocol}.png")

        try:
            res = validate_pair(pkl_path, args.tau,
                                plot_out=plot_out,
                                max_syn=args.max_syn,
                                zoom_ms=args.zoom_ms)
            for r in res:
                r['pair'] = pair_dir.name
            all_results.extend(res)
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
        print()

    if all_results:
        rmse_vals  = [r['rmse']         for r in all_results]
        max_errs   = [r['max_err']      for r in all_results]
        rel_peaks  = [r['rel_peak_err'] for r in all_results]
        print("=" * 55)
        print(f"SUMMARY  ({len(all_results)} synapses, {len(pair_dirs)} pairs)")
        print(f"  RMSE:          mean={np.mean(rmse_vals):.3e}  max={np.max(rmse_vals):.3e}")
        print(f"  max|err|:      mean={np.mean(max_errs):.3e}   max={np.max(max_errs):.3e}")
        print(f"  rel peak err:  mean={np.mean(rel_peaks)*100:.4f}%  max={np.max(rel_peaks)*100:.4f}%")
        perfect = sum(1 for r in all_results if r['max_err'] < 1e-8)
        print(f"  exact matches (|err|<1e-8): {perfect}/{len(all_results)}")


if __name__ == "__main__":
    main()

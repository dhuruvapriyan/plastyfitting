#!/usr/bin/env python3
"""
GB-Only model using pre-baked cai_CR traces.

Identical to gb_only.py in every respect, except it loads calcium traces from
a pre-computed directory (created by precompute_weighted_cai.py) where cai_CR
has already been replaced by:

    cai_CR = alpha * cai_NMDA_CR + beta * cai_VDCC_CR

with frozen alpha/beta (e.g. alpha=3, beta=1).  Because the combination is
done offline, JAX never sees separate NMDA/VDCC arrays and memory usage is
the same as the original gb_only.py.

Usage
-----
    # Step 1 – pre-bake (once):
    python precompute_weighted_cai.py --alpha 3 --beta 1

    # Step 2 – fit:
    python gb_only_precomputed.py \\
        --trace-dir /project/rrg-emuller/dhuruva/plastyfitting/trace_results/PRECOMPUTED_A3B1 \\
        --method de --max-iter 2000 --dt-step 10

    # Or evaluate saved params:
    python gb_only_precomputed.py \\
        --trace-dir .../PRECOMPUTED_A3B1 \\
        --eval plastyfitting/best_params_gb-only_baseline_20260304_005226.json
"""

import sys
import argparse
import logging

import numpy as np
import jax
import jax.numpy as jnp

import cicr_common as _cc
from cicr_common import (
    CICRModel,
    EXPERIMENTAL_TARGETS,
    preload_all_data,
    _jax_peak_effcai_zoh,
)


def _apply_gb_only_debug(cai_1syn, t, cp, cq, is_apical, dp):
    n = len(cai_1syn)
    return {
        "cai_total": np.copy(cai_1syn),
        "priming":   np.zeros(n),
        "ca_er":     np.zeros(n),
        "ca_cicr":   np.zeros(n),
    }


class GBOnlyPrecomputedModel(CICRModel):
    """GB-Only model with pre-baked weighted cai_CR.

    Structurally identical to GBOnlyModel; the difference is purely in which
    trace directory is loaded at runtime (set via --trace-dir).
    """

    DESCRIPTION      = "GB-Only (Pre-baked weighted cai_CR)"
    NEEDS_THRESHOLD_TRACES = True

    FIT_PARAMS = [
        ("gamma_d", 50.0, 500.0), ("gamma_p", 150.0, 500.0),
        ("a00", 1.0, 5.0), ("a01", 1.0, 5.0),
        ("a10", 1.0, 5.0), ("a11", 1.0, 5.0),
        ("a20", 1.0, 5.0), ("a21", 1.0, 5.0),
        ("a30", 1.0, 5.0), ("a31", 1.0, 5.0),
        ("tau_eff", 10.0, 500.0),
    ]

    DEFAULT_PARAMS = {
        "gamma_d": 51.01743564360413, "gamma_p": 499.53628856083213,
        "a00": 1.0699840626103416, "a01": 3.1427902571617885,
        "a10": 2.6192084284067754, "a11": 3.5391905721599493,
        "a20": 1.3144260462014046, "a21": 3.2449643447237335,
        "a30": 1.859040007848069,  "a31": 4.525739974526342,
        "tau_eff": 52.04,
    }

    def unpack_params(self, x):
        gb   = {"gamma_d": x[0], "gamma_p": x[1],
                "a00": x[2], "a01": x[3], "a10": x[4], "a11": x[5],
                "a20": x[6], "a21": x[7], "a30": x[8], "a31": x[9]}
        cicr = {"tau_eff": x[10]}
        return gb, cicr

    def get_debug_sim_fn(self):
        return _apply_gb_only_debug

    def get_init_fn(self):
        def init_fn(cai_first, rho0):
            return (0.0, rho0)
        return init_fn

    def get_step_factory(self):
        def scan_factory(params, syn_params):
            gb, cicr = params
            _c_pre, _c_post, is_apical, cai_pre, t_pre, cai_post, t_post = syn_params

            tau_effca = cicr["tau_eff"]
            cai_rest  = 70e-6

            dt_pre  = t_pre[1]  - t_pre[0]
            dt_post = t_post[1] - t_post[0]
            c_pre  = _jax_peak_effcai_zoh(cai_pre,  tau_effca, dt_pre,  cai_rest)
            c_post = _jax_peak_effcai_zoh(cai_post, tau_effca, dt_post, cai_rest)

            theta_d = jnp.where(is_apical,
                                gb["a20"] * c_pre + gb["a21"] * c_post,
                                gb["a00"] * c_pre + gb["a01"] * c_post)
            theta_p = jnp.where(is_apical,
                                gb["a30"] * c_pre + gb["a31"] * c_post,
                                gb["a10"] * c_pre + gb["a11"] * c_post)

            def scan_step(carry, inputs):
                effcai, rho = carry
                cai_raw, dt, _nves = inputs

                decay_eff  = jnp.exp(-dt / tau_effca)
                effcai_new = jnp.where(
                    dt > 0,
                    effcai * decay_eff + (cai_raw - cai_rest) * tau_effca * (1.0 - decay_eff),
                    effcai,
                )
                pot = jnp.where(effcai_new > theta_p, 1.0, 0.0)
                dep = jnp.where(effcai_new > theta_d, 1.0, 0.0)
                drho = (-rho * (1.0 - rho) * (0.5 - rho)
                        + pot * gb["gamma_p"] * (1.0 - rho)
                        - dep * gb["gamma_d"] * rho) / 70000.0
                rho_new = jnp.where(dt > 0, jnp.clip(rho + dt * drho, 0.0, 1.0), rho)
                return (effcai_new, rho_new), None

            return scan_step

        return scan_factory

    # ------------------------------------------------------------------
    # Override run() to inject --trace-dir before data loading
    # ------------------------------------------------------------------
    def run(self):
        parser = argparse.ArgumentParser(description=f"{self.DESCRIPTION}")
        parser.add_argument("--trace-dir", required=True,
                            help="Pre-baked trace directory produced by precompute_weighted_cai.py")
        parser.add_argument("--method",   choices=["de", "cmaes", "optax", "optuna", "pso"], default="de")
        parser.add_argument("--max-iter", type=int,   default=1000)
        parser.add_argument("--protocols", nargs="+", choices=list(EXPERIMENTAL_TARGETS.keys()))
        parser.add_argument("--max-pairs", type=int,  default=None)
        parser.add_argument("--dt-step",   type=int,  default=1)
        parser.add_argument("--popsize",   type=int,  default=15)
        parser.add_argument("--de-batch-size", type=int, default=8)
        parser.add_argument("--early-stopping", type=int, default=100)
        parser.add_argument("--interp-dt", type=float, default=None)
        parser.add_argument("--eval",    type=str,    default=None)
        parser.add_argument("--loss-v2", action="store_true")
        parser.add_argument("--loss-basic", action="store_true", help="Pure weighted MSE: sum_p w_p*(pred_p - target_p)^2, no sep_penalty")
        parser.add_argument("--verbose", action="store_true")
        args = parser.parse_args()

        if args.verbose:
            logging.getLogger().setLevel(logging.INFO)

        targets = (
            {p: v for p, v in EXPERIMENTAL_TARGETS.items() if p in args.protocols}
            if args.protocols else EXPERIMENTAL_TARGETS
        )

        protocol_data = preload_all_data(
            max_pairs=args.max_pairs,
            protocols=list(targets.keys()),
            needs_threshold_traces=self.NEEDS_THRESHOLD_TRACES,
            l5_trace_dir=args.trace_dir,   # ← use pre-baked directory
        )
        self.setup_jax(
            protocol_data, targets,
            dt_step=args.dt_step,
            interp_dt=args.interp_dt,
            use_loss_v2=args.loss_v2,
            use_loss_basic=args.loss_basic,
        )

        import time
        t0 = time.time()

        if args.eval:
            import json
            with open(args.eval) as fp:
                saved = json.load(fp)
            x = np.array([saved.get(name, self.DEFAULT_PARAMS.get(name, 0.0))
                          for name, *_ in self.FIT_PARAMS])
            preds = self.forward_batch(x)
            print("\n=== Evaluation ===")
            from cicr_common import EXPERIMENTAL_ERRORS
            for pname, pred in zip(self.proto_names, np.asarray(preds)):
                tgt  = EXPERIMENTAL_TARGETS.get(pname, float("nan"))
                err  = EXPERIMENTAL_ERRORS.get(pname, 0.0)
                print(f"  {pname:15s}  pred={pred:.4f}  target={tgt:.4f} ± {err:.4f}")
            opt_result = {"x": x, "fun": float(self.objective_single(x))}
        else:
            method_fn = {
                "de":     self.run_de,
                "cmaes":  self.run_cmaes,
                "optax":  self.run_optax,
                "optuna": self.run_optuna,
                "pso":    self.run_pso,
            }[args.method]
            opt_result = method_fn(
                max_iter=args.max_iter,
                popsize=getattr(args, "popsize", 15),
                patience=getattr(args, "early_stopping", 100),
                batch_size=getattr(args, "de_batch_size", 8),
            )

        elapsed = time.time() - t0
        print(f"\nElapsed: {elapsed:.1f}s")
        self.print_results(opt_result)


if __name__ == "__main__":
    model = GBOnlyPrecomputedModel()
    model.run()

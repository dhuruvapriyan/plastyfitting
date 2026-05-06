#!/usr/bin/env python3
"""
GB model with a single VDCC effcai integrator.

Pure VDCC variant distilled from gb_dual_effcai.py experiments:
  - Only cai_VDCC_CR drives the effcai integrator (cai_NMDA_CR ignored).
  - LTD:  VDCC effcai > theta_d_vdcc
  - LTP:  VDCC effcai > theta_p_vdcc

The VDCC channel alone encodes timing direction:
  +10ms (pre→post): EPSP + bAP overlap → large VDCC transient → effcai
    can exceed theta_d and theta_p → net outcome determined by gamma_p/gamma_d.
  -10ms (post→pre): bAP fires first, EPSP follows → different VDCC accumulation
    profile → different effcai trajectory → different dep/pot balance.

Parameters (11 total):
  gamma_d, gamma_p
  d_v0, d_v1        — LTD VDCC threshold (pre-weight, post-weight) — basal
  p_v0, p_v1        — LTP VDCC threshold (pre-weight, post-weight) — basal
  d_v0a, d_v1a      — LTD VDCC threshold (pre-weight, post-weight) — apical
  p_v0a, p_v1a      — LTP VDCC threshold (pre-weight, post-weight) — apical
  tau_eff

Usage
-----
    python gb_vdcc_only.py --method de --max-iter 2000 --dt-step 10 \\
        --protocols 10Hz_10ms 10Hz_-10ms --loss-basic
"""

import argparse
import json
import logging
import os
import time
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

from cicr_common_weighted import (
    WeightedCICRModel,
    _jax_peak_effcai_zoh,
    collate_protocol_to_jax,
    preload_all_data,
    EXPERIMENTAL_TARGETS,
    EXPERIMENTAL_ERRORS,
)

jax.config.update("jax_enable_x64", True)
logger = logging.getLogger(__name__)

_MIN_CA = 70e-6  # mM


class GBVdccOnlyModel(WeightedCICRModel):
    """Single VDCC effcai integrator; LTD and LTP both driven by VDCC alone."""

    DESCRIPTION = "GB VDCC-Only EffCai (11 params, basal/apical split)"
    NEEDS_THRESHOLD_TRACES      = True
    NEEDS_RAW_CAI_TRACE         = False
    NEEDS_BASE_THRESHOLD_TRACES = True
    NEEDS_CPRE_CPOST            = False

    # fmt: off
    FIT_PARAMS = [
        ("gamma_d",   50.0, 500.0),
        ("gamma_p",   150.0, 500.0),
        # VDCC thresholds — basal (a00=dep_pre, a01=dep_post, a10=pot_pre, a11=pot_post)
        ("a00",  1, 5.0),  ("a01",  1, 10.0),
        ("a10",  1, 10.0), ("a11",  1, 10.0),
        # VDCC thresholds — apical (a20=dep_pre, a21=dep_post, a30=pot_pre, a31=pot_post)
        ("a20", 1, 10.0),  ("a21", 1, 10.0),
        ("a30", 1, 10.0),  ("a31", 1, 10.0),
        # shared effcai time constant
        ("tau_effca", 250.0, 300.0),
    ]
    # fmt: on

    DEFAULT_PARAMS = {
        "gamma_d": 100.0, "gamma_p": 300.0,
        "a00":  2.0, "a01":  2.0,
        "a10":  4.0, "a11":  4.0,
        "a20": 2.0, "a21": 2.0,
        "a30": 4.0, "a31": 4.0,
        "tau_effca": 100.0,
    }

    def unpack_params(self, x):
        gb = {
            "gamma_d": x[0], "gamma_p": x[1],
            "a00":  x[2], "a01":  x[3],
            "a10":  x[4], "a11":  x[5],
            "a20": x[6], "a21": x[7],
            "a30": x[8], "a31": x[9],
        }
        aux = {"tau_effca": x[10]}
        return gb, aux

    def get_init_fn(self):
        def init_fn(cai_first, rho0):
            return (0.0, rho0)   # (effcai_vdcc, rho)
        return init_fn

    def get_step_factory(self):
        def scan_factory(params, syn_params):
            gb, aux = params
            tau_effca = aux["tau_effca"]

            (_, _, is_apical,
             _, _cai_pre_nmda, cai_pre_vdcc, t_pre,
             _, _cai_post_nmda, cai_post_vdcc, t_post) = syn_params

            dt_pre  = jnp.maximum(t_pre[1]  - t_pre[0],  1e-6)
            dt_post = jnp.maximum(t_post[1] - t_post[0], 1e-6)

            # Single-pulse peak VDCC effcai (thresholds calibrated to AP-driven peaks)
            c_pre_vdcc  = _jax_peak_effcai_zoh(cai_pre_vdcc,  tau_effca, dt_pre,  _MIN_CA)
            c_post_vdcc = _jax_peak_effcai_zoh(cai_post_vdcc, tau_effca, dt_post, _MIN_CA)

            # VDCC thresholds (basal vs apical)
            theta_d_vdcc = jnp.where(is_apical,
                gb["a20"] * c_pre_vdcc + gb["a21"] * c_post_vdcc,
                gb["a00"] * c_pre_vdcc + gb["a01"] * c_post_vdcc)
            theta_p_vdcc = jnp.where(is_apical,
                gb["a30"] * c_pre_vdcc + gb["a31"] * c_post_vdcc,
                gb["a10"] * c_pre_vdcc + gb["a11"] * c_post_vdcc)

            def scan_step(carry, inputs):
                effcai_vdcc, rho = carry
                _cai_nmda_t, cai_vdcc_t, dt, _nves = inputs

                decay  = jnp.exp(-dt / tau_effca)
                factor = tau_effca * (1.0 - decay)

                effcai_vdcc_new = jnp.where(
                    dt > 0,
                    effcai_vdcc * decay + (cai_vdcc_t - _MIN_CA) * factor,
                    effcai_vdcc,
                )

                dep = jnp.where(effcai_vdcc_new > theta_d_vdcc, 1.0, 0.0)
                pot = jnp.where(effcai_vdcc_new > theta_p_vdcc, 1.0, 0.0)

                drho = (
                    -rho * (1.0 - rho) * (0.5 - rho)
                    + pot * gb["gamma_p"] * (1.0 - rho)
                    - dep * gb["gamma_d"] * rho
                ) / 70000.0
                rho_new = jnp.where(dt > 0, jnp.clip(rho + dt * drho, 0.0, 1.0), rho)

                return (effcai_vdcc_new, rho_new), None

            return scan_step

        return scan_factory

    def setup_jax(self, protocol_data, targets,
                  lambda_reg=0.0, dt_step=1, interp_dt=None,
                  use_loss_v2=False, use_loss_basic=False):

        self.targets_dict   = targets
        self.proto_names    = list(targets.keys())
        self.target_vals    = jnp.array([targets[p] for p in self.proto_names])
        self.target_errs    = jnp.array([EXPERIMENTAL_ERRORS.get(p, 0.1) for p in self.proto_names])
        self.weights        = jnp.array([1.2 if p == "10Hz_-10ms" else 1.0 for p in self.proto_names])
        self.lambda_reg     = lambda_reg
        self.use_loss_v2    = use_loss_v2
        self.use_loss_basic = use_loss_basic

        raw_collated = {
            p: collate_protocol_to_jax(
                data,
                dt_step=dt_step,
                interp_dt=interp_dt,
                include_raw_cai=False,
                include_base_threshold_traces=True,
                include_cpre_cpost=False,
            )
            for p, data in protocol_data.items() if p in targets
        }
        self.collated_data = {p: d for p, d in raw_collated.items() if d is not None}
        self.proto_names   = [p for p in self.proto_names if p in self.collated_data]
        self.target_vals   = jnp.array([targets[p] for p in self.proto_names])
        self.target_errs   = jnp.array([EXPERIMENTAL_ERRORS.get(p, 0.1) for p in self.proto_names])
        self.weights       = jnp.array([1.2 if p == "10Hz_-10ms" else 1.0 for p in self.proto_names])

        step_factory = self.get_step_factory()
        init_fn      = self.get_init_fn()

        def sim_synapse(cai_nmda_trace, cai_vdcc_trace, t_trace, nves_trace,
                        is_apical, rho0, sm, bmean, valid,
                        cai_pre_nmda, cai_pre_vdcc, t_pre,
                        cai_post_nmda, cai_post_vdcc, t_post,
                        params):
            dt_trace = jnp.diff(t_trace, prepend=t_trace[0])
            dt_trace = jnp.where(dt_trace <= 0, 1e-6, dt_trace)
            init = init_fn(cai_vdcc_trace[0], rho0)

            syn_params = (
                0.0, 0.0, is_apical,
                jnp.zeros(1), cai_pre_nmda, cai_pre_vdcc, t_pre,
                jnp.zeros(1), cai_post_nmda, cai_post_vdcc, t_post,
            )
            scan_fn = step_factory(params, syn_params)

            final, _ = jax.lax.scan(
                scan_fn, init, (cai_nmda_trace, cai_vdcc_trace, dt_trace, nves_trace)
            )
            rho_final = final[-1]
            return jnp.where(valid & (rho_final >= 0.5), sm - bmean, 0.0)

        vmapsyn = jax.vmap(
            sim_synapse,
            in_axes=(
                0, 0, None, None,
                0, 0, 0, None, 0,
                0, 0, None,
                0, 0, None,
                None,
            )
        )

        def sim_pair(cai_nmda_p, cai_vdcc_p, t_p, nves_p,
                     isapi_p, rho0_p, bmean, sm_p, valid_p,
                     cai_pre_nmda_p, cai_pre_vdcc_p, t_pre_p,
                     cai_post_nmda_p, cai_post_vdcc_p, t_post_p,
                     params):
            contribs = vmapsyn(
                cai_nmda_p, cai_vdcc_p, t_p, nves_p,
                isapi_p, rho0_p, sm_p, bmean, valid_p,
                cai_pre_nmda_p, cai_pre_vdcc_p, t_pre_p,
                cai_post_nmda_p, cai_post_vdcc_p, t_post_p,
                params,
            )
            epsp_after   = bmean + jnp.sum(contribs)
            contribs_bef = jnp.where(valid_p & (rho0_p >= 0.5), sm_p - bmean, 0.0)
            epsp_before  = bmean + jnp.sum(contribs_bef)
            return jnp.where(epsp_before > 0, epsp_after / epsp_before, jnp.nan)

        vmappair = jax.vmap(
            sim_pair,
            in_axes=(0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0,  0, 0, 0,  None)
        )

        def sim_protocol(proto_data, params):
            ratios = vmappair(
                proto_data["cai_nmda"], proto_data["cai_vdcc"],
                proto_data["t"], proto_data["nves"],
                proto_data["is_apical"], proto_data["rho0"],
                proto_data["baseline"], proto_data["singletons"], proto_data["valid"],
                proto_data["cai_pre_nmda"], proto_data["cai_pre_vdcc"], proto_data["t_pre"],
                proto_data["cai_post_nmda"], proto_data["cai_post_vdcc"], proto_data["t_post"],
                params,
            )
            return jnp.nanmean(ratios)

        def forward_single(x_array, collated):
            params = self.unpack_params(x_array)
            return jnp.stack([sim_protocol(collated[p], params) for p in self.proto_names])

        @jax.jit
        def forward_batch(x_matrix, collated):
            return jax.vmap(forward_single, in_axes=(0, None))(x_matrix, collated)

        idx_ltp = self.proto_names.index("10Hz_10ms")  if "10Hz_10ms"  in self.proto_names else -1
        idx_ltd = self.proto_names.index("10Hz_-10ms") if "10Hz_-10ms" in self.proto_names else -1

        @jax.jit
        def objective_single(x_array, collated):
            preds = forward_single(x_array, collated)
            if self.use_loss_basic:
                return jnp.sum(self.weights * (preds - self.target_vals) ** 2)
            if self.use_loss_v2:
                base_loss = jnp.sum(self.weights * jnp.abs(preds - self.target_vals) / self.target_errs)
            else:
                base_loss = jnp.sum(self.weights * (preds - self.target_vals) ** 2)
            sep_penalty = 0.0
            if idx_ltp >= 0 and idx_ltd >= 0:
                actual_sep = preds[idx_ltp] - preds[idx_ltd]
                sep_penalty = 20.0 * jnp.maximum(0.0, 0.41 - actual_sep) ** 2
            loss = base_loss + sep_penalty
            if self.lambda_reg > 0:
                loss = loss + self.lambda_reg * jnp.sum((x_array - self.DEFAULT_X0) ** 2)
            return loss

        @jax.jit
        def objective_batch(x_matrix, collated):
            return jax.vmap(objective_single, in_axes=(0, None))(x_matrix, collated)

        self.forward_batch    = partial(forward_batch,    collated=self.collated_data)
        self.objective_single = partial(objective_single, collated=self.collated_data)
        self.objective_batch  = partial(objective_batch,  collated=self.collated_data)

    def run(self):
        parser = argparse.ArgumentParser(description=f"{self.DESCRIPTION}")
        parser.add_argument("--method",   choices=["de", "cmaes", "optax", "optuna", "pso"], default="de")
        parser.add_argument("--max-iter", type=int, default=1000)
        parser.add_argument("--protocols", nargs="+", choices=list(EXPERIMENTAL_TARGETS.keys()))
        parser.add_argument("--max-pairs",      type=int,   default=None)
        parser.add_argument("--dt-step",        type=int,   default=1)
        parser.add_argument("--popsize",        type=int,   default=15)
        parser.add_argument("--de-batch-size",  type=int,   default=8)
        parser.add_argument("--early-stopping", type=int,   default=100)
        parser.add_argument("--interp-dt",      type=float, default=None)
        parser.add_argument("--eval",           type=str,   default=None)
        parser.add_argument("--target-ltp",     type=float, default=None,
                            help="Override target EPSP ratio for 10Hz_10ms")
        parser.add_argument("--target-ltd",     type=float, default=None,
                            help="Override target EPSP ratio for 10Hz_-10ms")
        parser.add_argument("--loss-tol",       type=float, default=0.001,
                            help="Stop DE when loss drops below this value (default: 0.001)")
        parser.add_argument("--loss-v2",    action="store_true")
        parser.add_argument("--loss-basic", action="store_true",
                            help="Pure weighted MSE, no sep_penalty")
        parser.add_argument("--verbose",    action="store_true")
        args = parser.parse_args()

        if args.verbose:
            logging.getLogger().setLevel(logging.INFO)

        targets = (
            {p: v for p, v in EXPERIMENTAL_TARGETS.items() if p in args.protocols}
            if args.protocols else dict(EXPERIMENTAL_TARGETS)
        )
        if args.target_ltp is not None and "10Hz_10ms" in targets:
            logger.info("Overriding 10Hz_10ms target: %.4f → %.4f",
                        targets["10Hz_10ms"], args.target_ltp)
            targets["10Hz_10ms"] = args.target_ltp
        if args.target_ltd is not None and "10Hz_-10ms" in targets:
            logger.info("Overriding 10Hz_-10ms target: %.4f → %.4f",
                        targets["10Hz_-10ms"], args.target_ltd)
            targets["10Hz_-10ms"] = args.target_ltd

        protocol_data = preload_all_data(
            max_pairs=args.max_pairs,
            protocols=list(targets.keys()),
            needs_threshold_traces=True,
            include_raw_cai=False,
            include_base_threshold_traces=True,
            include_cpre_cpost=False,
        )
        self.setup_jax(
            protocol_data, targets,
            dt_step=args.dt_step,
            interp_dt=args.interp_dt,
            use_loss_v2=args.loss_v2,
            use_loss_basic=args.loss_basic,
        )

        t0 = time.time()
        ts = time.strftime("%Y%m%d_%H%M%S")
        model_tag = "gb_vdcc_only"

        if args.eval:
            if os.path.isfile(args.eval):
                with open(args.eval) as fp:
                    ep = json.load(fp)
            else:
                import json as _json
                ep = _json.loads(args.eval)
            dp = dict(self.DEFAULT_PARAMS)
            dp.update(ep)
            x_eval = np.array([dp[n] for n, *_ in self.FIT_PARAMS])
            res = {
                "method": "eval", "x": x_eval.tolist(),
                "fun": float(self.objective_single(jnp.array(x_eval))),
                "time": time.time() - t0,
            }
        else:
            method_fn = {
                "de":     self.run_de,
                "cmaes":  self.run_cmaes,
                "optax":  self.run_optax,
                "optuna": self.run_optuna,
                "pso":    self.run_pso,
            }[args.method]
            res = method_fn(
                max_iter=args.max_iter,
                popsize=getattr(args, "popsize", 15),
                patience=getattr(args, "early_stopping", 100),
                batch_size=getattr(args, "de_batch_size", 8),
                loss_tol=args.loss_tol,
            )
            res["time"] = time.time() - t0

            out_json = f"best_params_{model_tag}_{ts}.json"
            best_dict = {n: float(v) for n, v in zip(self.PARAM_NAMES, res["x"])}
            with open(out_json, "w") as fp:
                json.dump(best_dict, fp, indent=4)
            print(f"\nSaved best parameters → {out_json}")

        elapsed = time.time() - t0
        print(f"Elapsed: {elapsed:.1f}s")
        self.print_results(res)


if __name__ == "__main__":
    model = GBVdccOnlyModel()
    model.run()

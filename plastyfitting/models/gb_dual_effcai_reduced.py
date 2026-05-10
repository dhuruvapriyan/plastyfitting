#!/usr/bin/env python3
"""
GB Dual EffCai — REDUCED 9-parameter variant.

Motivated by what gb_dual_effcai.py found after fitting:

  LTD  = VDCC-only, post-synaptic drive:
           theta_d_vdcc = d_v1 * c_post_vdcc   (NMDA-LTD removed entirely)

  LTP  = AND gate, biologically specific roles:
           theta_p_nmda = p_n1 * c_post_nmda   (post-NMDA dominates)
           theta_p_vdcc = p_v0 * c_pre_vdcc    (pre-VDCC dominates)

Parameters (9 total, basal + apical split for each threshold):
  gamma_d, gamma_p
  d_v1,  d_v1a    — LTD  VDCC post-weight  (basal, apical)
  p_n1,  p_n1a    — LTP  NMDA post-weight  (basal, apical)
  p_v0,  p_v0a    — LTP  VDCC pre-weight   (basal, apical)
  tau_eff

Story:
  +10ms (pre→post): EPSP → Mg²⁺ unblock → NMDA effcai driven by post
      fires → post-NMDA AND pre-VDCC both exceed thresholds → LTP.
  -10ms (post→pre): bAP first → VDCC post-driven → LTD via OR;
      NMDA effcai low (no Mg²⁺ unblock before pre) → AND gate fails → net LTD.

Usage:
    python gb_dual_effcai_reduced.py --method de --max-iter 2000 \\
        --dt-step 10 --protocols 10Hz_10ms 10Hz_-10ms --loss-basic
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

from ..cicr_common_weighted import (
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


class GBDualEffcaiReducedModel(WeightedCICRModel):
    """
    Reduced dual-effcai model.

    LTD: VDCC effcai > theta_d_vdcc  (NMDA gate removed)
    LTP: NMDA effcai > theta_p_nmda  AND  VDCC effcai > theta_p_vdcc

    theta_d_vdcc = d_v1  * c_post_vdcc   (back-propagating AP)
    theta_p_nmda = p_n1  * c_post_nmda   (Mg²⁺ unblock via post depol.)
    theta_p_vdcc = p_v0  * c_pre_vdcc    (pre-synaptic release)

    Separate coefficients for apical vs basal dendrites (suffixed 'a').
    """

    DESCRIPTION = "GB Dual EffCai REDUCED (9 params, biophysics-constrained)"
    NEEDS_THRESHOLD_TRACES     = True
    NEEDS_RAW_CAI_TRACE        = False
    NEEDS_BASE_THRESHOLD_TRACES = True
    NEEDS_CPRE_CPOST           = False

    # fmt: off
    FIT_PARAMS = [
        ("gamma_d",  50.0, 500.0),
        ("gamma_p",  50.0, 500.0),
        # LTD: VDCC post-weight  (basal, apical)
        ("d_v1",   0.1, 30.0), ("d_v1a",  0.1, 30.0),
        # LTP NMDA: post-weight  (basal, apical)
        ("p_n1",   0.1, 30.0), ("p_n1a",  0.1, 30.0),
        # LTP VDCC: pre-weight   (basal, apical)
        ("p_v0",   0.1, 30.0), ("p_v0a",  0.1, 30.0),
        # shared effcai time constant
        ("tau_eff", 10.0, 500.0),
    ]
    # fmt: on

    DEFAULT_PARAMS = {
        "gamma_d":  100.0,  "gamma_p":  300.0,
        "d_v1":  2.0,  "d_v1a":  2.0,
        "p_n1":  4.0,  "p_n1a":  4.0,
        "p_v0":  4.0,  "p_v0a":  4.0,
        "tau_eff": 100.0,
    }

    def unpack_params(self, x):
        gb = {
            "gamma_d": x[0], "gamma_p": x[1],
            "d_v1":  x[2], "d_v1a":  x[3],
            "p_n1":  x[4], "p_n1a":  x[5],
            "p_v0":  x[6], "p_v0a":  x[7],
        }
        aux = {"tau_eff": x[8]}
        return gb, aux

    def get_init_fn(self):
        def init_fn(cai_first, rho0):
            return (0.0, 0.0, rho0)   # (effcai_nmda, effcai_vdcc, rho)
        return init_fn

    def get_step_factory(self):
        def scan_factory(params, syn_params):
            gb, aux = params
            tau_effca = aux["tau_eff"]

            (_, _, is_apical,
             _, cai_pre_nmda, cai_pre_vdcc, t_pre,
             _, cai_post_nmda, cai_post_vdcc, t_post) = syn_params

            dt_pre  = jnp.maximum(t_pre[1]  - t_pre[0],  1e-6)
            dt_post = jnp.maximum(t_post[1] - t_post[0], 1e-6)

            # Single-pulse peak effcai for each channel
            c_pre_vdcc  = _jax_peak_effcai_zoh(cai_pre_vdcc,  tau_effca, dt_pre,  _MIN_CA)
            c_post_nmda = _jax_peak_effcai_zoh(cai_post_nmda, tau_effca, dt_post, _MIN_CA)
            c_post_vdcc = _jax_peak_effcai_zoh(cai_post_vdcc, tau_effca, dt_post, _MIN_CA)

            # LTD: VDCC only, driven by back-propagating AP (post)
            theta_d_vdcc = jnp.where(is_apical,
                gb["d_v1a"] * c_post_vdcc,
                gb["d_v1"]  * c_post_vdcc)

            # LTP NMDA: post-NMDA (Mg²⁺ unblock requires post depolarisation)
            theta_p_nmda = jnp.where(is_apical,
                gb["p_n1a"] * c_post_nmda,
                gb["p_n1"]  * c_post_nmda)

            # LTP VDCC: pre-VDCC (pre-synaptic release site coupling)
            theta_p_vdcc = jnp.where(is_apical,
                gb["p_v0a"] * c_pre_vdcc,
                gb["p_v0"]  * c_pre_vdcc)

            def scan_step(carry, inputs):
                effcai_nmda, effcai_vdcc, rho = carry
                cai_nmda_t, cai_vdcc_t, dt, _nves = inputs

                decay  = jnp.exp(-dt / tau_effca)
                factor = tau_effca * (1.0 - decay)

                effcai_nmda_new = jnp.where(
                    dt > 0,
                    effcai_nmda * decay + (cai_nmda_t - _MIN_CA) * factor,
                    effcai_nmda,
                )
                effcai_vdcc_new = jnp.where(
                    dt > 0,
                    effcai_vdcc * decay + (cai_vdcc_t - _MIN_CA) * factor,
                    effcai_vdcc,
                )

                # LTD: VDCC effcai exceeds post-driven threshold
                dep = jnp.where(effcai_vdcc_new > theta_d_vdcc, 1.0, 0.0)

                # LTP: coincidence — NMDA-post AND VDCC-pre both exceeded
                pot = jnp.where(
                    (effcai_nmda_new > theta_p_nmda) & (effcai_vdcc_new > theta_p_vdcc),
                    1.0, 0.0,
                )

                drho = (
                    -rho * (1.0 - rho) * (0.5 - rho)
                    + pot * gb["gamma_p"] * (1.0 - rho)
                    - dep * gb["gamma_d"] * rho
                ) / 70000.0
                rho_new = jnp.where(dt > 0, jnp.clip(rho + dt * drho, 0.0, 1.0), rho)

                return (effcai_nmda_new, effcai_vdcc_new, rho_new), None

            return scan_step

        return scan_factory

    # ------------------------------------------------------------------
    # setup_jax: identical structure to GBDualEffcaiModel but uses
    # the reduced unpack_params / scan.
    # ------------------------------------------------------------------
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
            init = init_fn(cai_nmda_trace[0], rho0)

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
            epsp_after  = bmean + jnp.sum(contribs)
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
        parser.add_argument("--max-pairs",     type=int,   default=None)
        parser.add_argument("--dt-step",       type=int,   default=1)
        parser.add_argument("--popsize",       type=int,   default=15)
        parser.add_argument("--de-batch-size", type=int,   default=8)
        parser.add_argument("--early-stopping",type=int,   default=100)
        parser.add_argument("--interp-dt",     type=float, default=None)
        parser.add_argument("--eval",          type=str,   default=None)
        parser.add_argument("--loss-v2",   action="store_true")
        parser.add_argument("--loss-basic", action="store_true",
                            help="Pure weighted MSE, no sep_penalty")
        parser.add_argument("--verbose",   action="store_true")
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
        model_tag = "gb_dual_effcai_reduced"

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
    model = GBDualEffcaiReducedModel()
    model.run()

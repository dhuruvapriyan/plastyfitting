#!/usr/bin/env python3
import numpy as np
import jax
import jax.numpy as jnp
from ..cicr_common import CICRModel, _jax_peak_effcai_zoh


def _apply_gb_only_debug(cai_1syn, t, cp, cq, is_apical, dp):
    n = len(cai_1syn)
    return {
        "cai_total": np.copy(cai_1syn),
        "priming":   np.zeros(n),
        "ca_er":     np.zeros(n),
        "ca_cicr":   np.zeros(n),
    }


class GBOnlyModel(CICRModel):
    DESCRIPTION = "GB-Only Baseline (No CICR)"
    NEEDS_THRESHOLD_TRACES = True

    FIT_PARAMS = [
        ("gamma_d", 50.0, 200.0), ("gamma_p", 150.0, 300.0),
        ("a00", 1.0, 10.0), ("a01", 1.0, 10.0),
        ("a10", 1.0, 10.0), ("a11", 1.0, 10.0),
        ("a20", 1.0, 10.0), ("a21", 1.0, 10.0),
        ("a30", 1.0, 10.0), ("a31", 1.0, 10.0),
        ("tau_eff", 30.0, 700.0),
    ]

    # Best parameters:
    # gamma_d                        =   211.7601  (default: 100.0000, +111.8%)
    # gamma_p                        =   160.0662  (default: 300.0000, -46.6%)
    # a00                            =     4.4781  (default: 2.0000, +123.9%)
    # a01                            =     8.6235  (default: 2.0000, +331.2%)
    # a10                            =     2.6474  (default: 4.0000, -33.8%)
    # a11                            =     2.2063  (default: 4.0000, -44.8%)
    # a20                            =     3.7901  (default: 2.0000, +89.5%)
    # a21                            =     3.7069  (default: 2.0000, +85.3%)
    # a30                            =     4.9317  (default: 4.0000, +23.3%)
    # a31                            =     6.0168  (default: 4.0000, +50.4%)
    # tau_effca                      =   280.8576  (default: 100.0000, +180.9%)
    # Best parameters from gb_vdcc_only V19 (weighted, dt=1)
    # SEED_PARAMS = [
    #     {
    #         "gamma_d": 211.7601, "gamma_p": 160.0662,
    #         "a00": 4.4781, "a01": 8.6235,
    #         "a10": 2.6474, "a11": 2.2063,
    #         "a20": 3.7901, "a21": 3.7069,
    #         "a30": 4.9317, "a31": 6.0168,
    #         "tau_eff": 280.8576,
    #     },
    # ]

    # DEFAULT_PARAMS = {
    #     "gamma_d": 379.5256059628456,
    #     "gamma_p": 162.02006326916813,
    #     "a00": 1.0770144130928545,
    #     "a01": 2.0379105635626904,
    #     "a10": 2.894174205875408,
    #     "a11": 1.3571983004135566,
    #     "a20": 2.3097860482130477,
    #     "a21": 2.317590020162771,
    #     "a30": 3.692021341986094,
    #     "a31": 1.927119096974426,
    #     "tau_effca": 257.9025228124917,
    # }
    DEFAULT_PARAMS = {
    "gamma_d": 101.5,
    "gamma_p": 216.2,
    "a00": 1.002,
    "a01": 1.954,
    "a10": 1.159,
    "a11": 2.483,
    "a20": 1.127,
    "a21": 2.456,
    "a30": 5.236,
    "a31": 1.782,
    "tau_eff": 278.318,
}


    def unpack_params(self, x):
        gb = {'gamma_d': x[0], 'gamma_p': x[1], 'a00': x[2], 'a01': x[3],
              'a10': x[4], 'a11': x[5], 'a20': x[6], 'a21': x[7], 'a30': x[8], 'a31': x[9]}
        cicr = {'tau_eff': x[10]}
        return gb, cicr

    def get_debug_sim_fn(self):
        return _apply_gb_only_debug

    def get_init_fn(self):
        def init_fn(cai_first, rho0):
            # carry = (effcai, cai_prev, rho)
            # cai_prev is needed for piecewise-linear effcai integration
            return (0.0, cai_first, rho0)
        return init_fn

    def get_step_factory(self):
        def scan_factory(params, syn_params):
            gb, cicr = params
            # When tau_eff is fitted, c_pre/c_post must be recomputed from the
            # single-pulse threshold traces for the current candidate tau_eff.
            _c_pre, _c_post, is_apical, cai_pre, t_pre, cai_post, t_post = syn_params

            tau_effca = cicr['tau_eff']
            cai_rest  = 70e-6

            dt_pre = jnp.maximum(t_pre[1] - t_pre[0], 1e-6)
            dt_post = jnp.maximum(t_post[1] - t_post[0], 1e-6)
            c_pre = _jax_peak_effcai_zoh(cai_pre, tau_effca, dt_pre, cai_rest)
            c_post = _jax_peak_effcai_zoh(cai_post, tau_effca, dt_post, cai_rest)

            theta_d = jnp.where(is_apical,
                                 gb['a20']*c_pre + gb['a21']*c_post,
                                 gb['a00']*c_pre + gb['a01']*c_post)
            theta_p = jnp.where(is_apical,
                                 gb['a30']*c_pre + gb['a31']*c_post,
                                 gb['a10']*c_pre + gb['a11']*c_post)

            def scan_step(carry, inputs):
                effcai, cai_prev, rho = carry
                cai_raw, dt, _nves = inputs

                # Piecewise-linear (PWL) effcai integration — matches NEURON CVODE
                # For constant input u over [t, t+dt]:  ZOH gives ~1% error on
                # large CVODE steps. PWL interpolates linearly between cai_prev
                # and cai_raw, reducing error by ~100x.
                f0 = jnp.maximum(cai_prev - cai_rest, 0.0)
                f1 = jnp.maximum(cai_raw  - cai_rest, 0.0)
                decay_eff = jnp.exp(-dt / tau_effca)
                safe_dt   = jnp.where(dt > 0, dt, 1.0)  # avoid div-by-zero
                slope     = (f1 - f0) / safe_dt
                effcai_new = jnp.where(
                    dt > 0,
                    effcai * decay_eff
                    + f0    * tau_effca * (1.0 - decay_eff)
                    + slope * (tau_effca * dt - tau_effca**2 * (1.0 - decay_eff)),
                    effcai,
                )

                # NEURON convention: dep/pot signals are gated by effcai at the
                # START of the interval (carried value), not the end.
                # Using effcai_new (end-of-step) shifts the switching by one step
                # and produces ~2% rho error over 170-spike simulations.
                pot = jnp.where(effcai     > theta_p, 1.0, 0.0)
                dep = jnp.where(effcai     > theta_d, 1.0, 0.0)

                drho = (-rho*(1.0-rho)*(0.5-rho)
                        + pot*gb['gamma_p']*(1.0-rho)
                        - dep*gb['gamma_d']*rho) / 70000.0
                rho_new = jnp.where(dt > 0, jnp.clip(rho + dt * drho, 0.0, 1.0), rho)

                return (effcai_new, cai_raw, rho_new), None
            return scan_step
        return scan_factory


if __name__ == "__main__":
    model = GBOnlyModel()
    model.run()

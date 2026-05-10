#!/usr/bin/env python3
import jax.numpy as jnp
from ..cicr_common_weighted import WeightedCICRModel, _jax_peak_effcai_zoh, combine_weighted_cai_jax


def _apply_gb_only_debug(cai_1syn, t, cp, cq, is_apical, dp):
    n = len(cai_1syn)
    return {
        "cai_total": cai_1syn,
        "priming": jnp.zeros(n),
        "ca_er": jnp.zeros(n),
        "ca_cicr": jnp.zeros(n),
    }


class GBOnlyWeightedModel(WeightedCICRModel):
    DESCRIPTION = "GB-Only Weighted NMDA+VDCC"
    NEEDS_THRESHOLD_TRACES = True
    NEEDS_RAW_CAI_TRACE = False
    NEEDS_BASE_THRESHOLD_TRACES = False
    NEEDS_CPRE_CPOST = False

    FIT_PARAMS = [
        ("gamma_d", 50.0, 500.0), ("gamma_p", 150.0, 500.0),
        ("a00", 1.0, 5.0), ("a01", 1.0, 5.0),
        ("a10", 1.0, 5.0), ("a11", 1.0, 5.0),
        ("a20", 1.0, 5.0), ("a21", 1.0, 5.0),
        ("a30", 1.0, 5.0), ("a31", 1.0, 5.0),
        ("tau_eff", 10.0, 500.0),
        ("alpha", 0.0, 5.0), ("beta", 0.0, 5.0),
    ]

    DEFAULT_PARAMS = {
        "gamma_d": 51.01743564360413,
        "gamma_p": 499.53628856083213,
        "a00": 1.0699840626103416,
        "a01": 3.1427902571617885,
        "a10": 2.6192084284067754,
        "a11": 3.5391905721599493,
        "a20": 1.3144260462014046,
        "a21": 3.2449643447237335,
        "a30": 1.859040007848069,
        "a31": 4.525739974526342,
        "tau_eff": 52.04,
        "alpha": 1.0,
        "beta": 1.0,
    }

    def unpack_params(self, x):
        gb = {
            "gamma_d": x[0], "gamma_p": x[1],
            "a00": x[2], "a01": x[3], "a10": x[4], "a11": x[5],
            "a20": x[6], "a21": x[7], "a30": x[8], "a31": x[9],
            "alpha": x[11], "beta": x[12],
        }
        aux = {"tau_eff": x[10]}
        return gb, aux

    def get_debug_sim_fn(self):
        return _apply_gb_only_debug

    def get_init_fn(self):
        def init_fn(cai_first, rho0):
            return (0.0, rho0)
        return init_fn

    def combine_cai_trace(self, cai_total, cai_nmda, cai_vdcc, params):
        gb, _ = params
        return combine_weighted_cai_jax(cai_nmda, cai_vdcc, gb["alpha"], gb["beta"])

    def get_step_factory(self):
        def scan_factory(params, syn_params):
            gb, aux = params
            (_c_pre, _c_post, is_apical,
             cai_pre, cai_pre_nmda, cai_pre_vdcc, t_pre,
             cai_post, cai_post_nmda, cai_post_vdcc, t_post) = syn_params

            tau_effca = aux["tau_eff"]
            cai_rest = 70e-6
            alpha = gb["alpha"]
            beta = gb["beta"]

            cai_pre_weighted = combine_weighted_cai_jax(cai_pre_nmda, cai_pre_vdcc, alpha, beta)
            cai_post_weighted = combine_weighted_cai_jax(cai_post_nmda, cai_post_vdcc, alpha, beta)

            dt_pre = jnp.maximum(t_pre[1] - t_pre[0], 1e-6)
            dt_post = jnp.maximum(t_post[1] - t_post[0], 1e-6)
            c_pre = _jax_peak_effcai_zoh(cai_pre_weighted, tau_effca, dt_pre, cai_rest)
            c_post = _jax_peak_effcai_zoh(cai_post_weighted, tau_effca, dt_post, cai_rest)

            theta_d = jnp.where(
                is_apical,
                gb["a20"] * c_pre + gb["a21"] * c_post,
                gb["a00"] * c_pre + gb["a01"] * c_post,
            )
            theta_p = jnp.where(
                is_apical,
                gb["a30"] * c_pre + gb["a31"] * c_post,
                gb["a10"] * c_pre + gb["a11"] * c_post,
            )

            def scan_step(carry, inputs):
                effcai, rho = carry
                cai_raw, dt, _nves = inputs
                decay_eff = jnp.exp(-dt / tau_effca)
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


if __name__ == "__main__":
    model = GBOnlyWeightedModel()
    model.run()

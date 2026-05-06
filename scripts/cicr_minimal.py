#!/usr/bin/env python3
import numpy as np
import jax.numpy as jnp
from cicr_common import CICRModel, _jax_peak_effcai_zoh

CAI_REST = 70e-6
TAU_REF  = 2000.0
K_H_REF  = 0.0005


def _debug_sim(cai_1syn, t, cp, cq, is_apical, dp):
    gamma_d      = float(dp.get("gamma_d_GB_GluSynapse", 150.0))
    gamma_p      = float(dp.get("gamma_p_GB_GluSynapse", 150.0))
    tau_eff      = float(dp.get("tau_eff", 278.318))
    delta_P      = float(dp.get("delta_P", 0.02))
    tau_P        = float(dp.get("tau_P", 150000.0))
    alpha_thresh = float(dp.get("alpha_thresh", 0.1))
    A_cicr       = float(dp.get("A_cicr", 0.01))
    tau_cicr     = float(dp.get("tau_cicr", 500.0))

    if is_apical:
        theta_d = dp.get("a20", 1.0) * cp + dp.get("a21", 1.0) * cq
        theta_p = dp.get("a30", 1.0) * cp + dp.get("a31", 1.0) * cq
    else:
        theta_d = dp.get("a00", 1.0) * cp + dp.get("a01", 1.0) * cq
        theta_p = dp.get("a10", 1.0) * cp + dp.get("a11", 1.0) * cq

    P_thresh = alpha_thresh * theta_p
    n = len(cai_1syn)
    cai_total   = np.copy(cai_1syn)
    P_out       = np.zeros(n)
    ca_cicr_out = np.zeros(n)
    h_ref_out   = np.zeros(n)
    effcai_out  = np.zeros(n)
    rho_out     = np.zeros(n)

    P = 0.0; ca_cicr = 0.0; h_ref = 1.0; effcai = 0.0
    rho = float(dp.get("rho0", 0.5))
    rho_out[0] = rho; h_ref_out[0] = 1.0

    for i in range(n - 1):
        dt = t[i + 1] - t[i]
        if dt <= 0:
            P_out[i+1] = P; ca_cicr_out[i+1] = ca_cicr
            h_ref_out[i+1] = h_ref; effcai_out[i+1] = effcai; rho_out[i+1] = rho
            continue

        ca_ext = max(0.0, cai_1syn[i] - CAI_REST)
        dP     = delta_P * ca_ext * (1.0 - P) - P / tau_P
        P_new  = max(0.0, min(1.0, P + dt * dP))
        P_peak = P_new

        if P_new > P_thresh and h_ref > 0.5:
            ca_cicr += A_cicr * h_ref
            P_new    = 0.0
            h_ref    = 0.0

        ca_cicr = max(0.0, ca_cicr * np.exp(-dt / tau_cicr))
        k_h2    = K_H_REF ** 2
        h_inf   = k_h2 / (k_h2 + ca_cicr ** 2 + 1e-30)
        h_ref   = min(1.0, max(0.0, h_ref + dt * (h_inf - h_ref) / TAU_REF))

        d_e    = np.exp(-dt / tau_eff)
        effcai = effcai * d_e + (ca_ext + ca_cicr) * tau_eff * (1.0 - d_e)

        pot = 1.0 if effcai > theta_p else 0.0
        dep = 1.0 if effcai > theta_d else 0.0
        drho = (-rho*(1-rho)*(0.5-rho) + gamma_p*(1-rho)*pot - gamma_d*rho*dep) / 70000.0
        rho  = min(1.0, max(0.0, rho + dt * drho))

        P = P_new
        P_out[i+1] = P_peak; ca_cicr_out[i+1] = ca_cicr
        h_ref_out[i+1] = h_ref; effcai_out[i+1] = effcai; rho_out[i+1] = rho

    return {
        "cai_total": cai_total, "priming": P_out, "ca_er": np.zeros(n),
        "ca_cicr": ca_cicr_out, "ca_ryr": np.zeros(n), "ca_ip3r": np.zeros(n),
        "h_ref": h_ref_out, "effcai": effcai_out, "rho": rho_out,
    }


class CICRPrimingModel(CICRModel):
    DESCRIPTION = "Minimal CICR Priming (P → sustained ER release)"
    NEEDS_THRESHOLD_TRACES = True

    FIT_PARAMS = [
        ("gamma_d_GB_GluSynapse", 50.0, 250.0),
        ("gamma_p_GB_GluSynapse", 50.0, 300.0),
        ("a00", 1.0, 3.0), ("a01", 1.0, 3.0),
        ("a10", 3.0, 15.0), ("a11", 3.0, 15.0),
        ("a20", 1.0, 3.0), ("a21", 1.0, 3.0),
        ("a30", 3.0, 15.0), ("a31", 3.0, 15.0),
        ("delta_P", 0.01, 1.0),
        ("tau_P", 60000.0, 300000.0),
        ("alpha_thresh", 0.5, 3.0),
        ("A_cicr", 0.0005, 0.005),
        ("tau_cicr", 100.0, 2000.0),
        ("tau_eff", 10.0, 500.0),
    ]

    DEFAULT_PARAMS = {
        "gamma_d_GB_GluSynapse": 244.27, "gamma_p_GB_GluSynapse": 201.89,
        "a00": 1.38, "a01": 2.65, "a10": 4.07, "a11": 4.20,
        "a20": 3.0,  "a21": 1.47, "a30": 5.0,  "a31": 3.5,
        "delta_P": 0.5, "tau_P": 150000.0, "alpha_thresh": 0.5,
        "A_cicr": 0.002, "tau_cicr": 500.0,
        "tau_eff": 278.318,
    }

    def unpack_params(self, x):
        gb = {
            'gamma_d': x[0], 'gamma_p': x[1],
            'a00': x[2], 'a01': x[3], 'a10': x[4], 'a11': x[5],
            'a20': x[6], 'a21': x[7], 'a30': x[8], 'a31': x[9],
        }
        cicr = {
            'delta_P': x[10], 'tau_P': x[11], 'alpha_thresh': x[12],
            'A_cicr': x[13], 'tau_cicr': x[14],
            'tau_eff': x[15],
        }
        return gb, cicr

    def get_init_fn(self):
        def init_fn(cai_first, rho0):
            return (0.0, 0.0, 1.0, 0.0, rho0)
        return init_fn

    def get_debug_sim_fn(self):
        return _debug_sim

    def get_step_factory(self):
        def scan_factory(params, syn_params):
            gb, cicr = params
            _c_pre, _c_post, is_apical, cai_pre, t_pre, cai_post, t_post = syn_params

            tau_effca    = cicr['tau_eff']
            delta_P      = cicr['delta_P']
            tau_P        = cicr['tau_P']
            alpha_thresh = cicr['alpha_thresh']
            A_cicr       = cicr['A_cicr']
            tau_cicr     = cicr['tau_cicr']

            dt_pre  = t_pre[1]  - t_pre[0]
            dt_post = t_post[1] - t_post[0]
            c_pre  = _jax_peak_effcai_zoh(cai_pre,  tau_effca, dt_pre,  CAI_REST)
            c_post = _jax_peak_effcai_zoh(cai_post, tau_effca, dt_post, CAI_REST)

            theta_d = jnp.where(is_apical,
                                 gb['a20']*c_pre + gb['a21']*c_post,
                                 gb['a00']*c_pre + gb['a01']*c_post)
            theta_p = jnp.where(is_apical,
                                 gb['a30']*c_pre + gb['a31']*c_post,
                                 gb['a10']*c_pre + gb['a11']*c_post)

            k_h2     = K_H_REF ** 2
            P_thresh = alpha_thresh * theta_p

            def scan_step(carry, inputs):
                P, ca_cicr, h_ref, effcai, rho = carry
                cai_raw, dt, nves_i = inputs

                ca_ext   = jnp.maximum(0.0, cai_raw - CAI_REST)
                dP       = delta_P * ca_ext * (1.0 - P) - P / tau_P
                P_updated = jnp.clip(P + dt * dP, 0.0, 1.0)

                trigger = jnp.where((P_updated > P_thresh) & (h_ref > 0.5), 1.0, 0.0)
                ca_cicr_inject      = A_cicr * h_ref * trigger
                P_new               = jnp.where(trigger > 0.5, 0.0, P_updated)
                h_ref_post_trigger  = jnp.where(trigger > 0.5, 0.0, h_ref)

                ca_cicr_new = jnp.maximum(0.0, (ca_cicr + ca_cicr_inject) * jnp.exp(-dt / tau_cicr))

                h_inf    = k_h2 / (k_h2 + ca_cicr_new**2 + 1e-30)
                h_ref_new = jnp.where(dt > 0,
                    jnp.clip(h_ref_post_trigger + dt * (h_inf - h_ref_post_trigger) / TAU_REF, 0.0, 1.0),
                    h_ref_post_trigger)

                decay_eff  = jnp.exp(-dt / tau_effca)
                effcai_new = jnp.where(dt > 0,
                    effcai * decay_eff + (ca_ext + ca_cicr_new) * tau_effca * (1.0 - decay_eff),
                    effcai)

                pot = jnp.where(effcai_new > theta_p, 1.0, 0.0)
                dep = jnp.where(effcai_new > theta_d, 1.0, 0.0)
                drho = (-rho*(1.0-rho)*(0.5-rho)
                        + pot*gb['gamma_p']*(1.0-rho)
                        - dep*gb['gamma_d']*rho) / 70000.0
                rho_new = jnp.where(dt > 0, jnp.clip(rho + dt * drho, 0.0, 1.0), rho)

                return (P_new, ca_cicr_new, h_ref_new, effcai_new, rho_new), None

            return scan_step
        return scan_factory


if __name__ == "__main__":
    CICRPrimingModel().run()

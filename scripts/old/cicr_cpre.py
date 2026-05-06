#!/usr/bin/env python3
"""
Primed-Junctional CICR Model V4.2 — Linear Slow Water Bucket

Architecture: 
1. ER acts as a slow cumulative integrator (Water Bucket) using strict linear filling.
2. The SERCA pump only turns on when effcai > theta_d (Meaningful Ca2+ entry).
3. The filling rate is bounded. It takes many pulses (2-5 mins) to fill.
4. Multiplicative CICR-Potentiation: Ca_CICR dynamically scales gamma_p.
"""

import numpy as np
import jax.numpy as jnp
from cicr_common import CICRModel, compute_effcai_piecewise_linear_jax


def _apply_primed_junctional_debug(cai_1syn, t, cp, cq, is_apical, dp):
    """Plain-NumPy exact mirror of the JAX scan loop for diagnostic plotting."""
    k_fill     = float(dp["k_fill"])
    tau_leak   = float(dp["tau_leak"])
    K_clip     = float(dp["K_clip"])
    K_trig     = float(dp["K_trig"])
    n_trig     = float(dp["n_trig"])
    Vmax_CICR  = float(dp["Vmax_CICR"])
    tau_CICR   = float(dp["tau_CICR"])
    g_cicr     = float(dp["g_cicr"])
    tau_eff    = float(dp["tau_eff"])
    cai_rest   = 70e-6

    # Extract dynamic thresholds for gating
    theta_d = float(dp.get("theta_d", 0.001))
    theta_p = float(dp.get("theta_p", 0.005))
    gamma_d = float(dp.get("gamma_d_GB_GluSynapse", 150.0))
    gamma_p = float(dp.get("gamma_p_GB_GluSynapse", 200.0))

    interp_dt = float(dp.get("interp_dt", 0))
    t_orig = t
    if interp_dt > 0:
        t_fine = np.arange(t[0], t[-1], interp_dt)
        cai_1syn = np.interp(t_fine, t, cai_1syn)
        t = t_fine

    n = len(cai_1syn)
    cai_total   = np.copy(cai_1syn)
    prime_out   = np.zeros(n)
    ca_cicr_out = np.zeros(n)
    effcai_out  = np.zeros(n)

    P = 0.0
    ca_cicr = 0.0
    effcai = 0.0

    prime_out[0] = P
    ca_cicr_out[0] = ca_cicr
    effcai_out[0] = effcai

    for i in range(n - 1):
        dt = t[i + 1] - t[i]
        if dt <= 0.0:
            prime_out[i + 1]   = P
            ca_cicr_out[i + 1] = ca_cicr
            effcai_out[i + 1]  = effcai
            continue

        ca_e = max(0.0, cai_1syn[i] - cai_rest)

        # 1. Integrator (Raw Ca2+ only, acting as threshold sensor)
        eff_decay = np.exp(-dt / tau_eff)
        effcai = effcai * eff_decay + ca_e * tau_eff * (1.0 - eff_decay)

        # 2. Slow Water Bucket (Linear Integration)
        fill_gate = 1.0 if effcai > theta_d else 0.0
        cai_clip = K_clip * np.tanh(ca_e / (K_clip + 1e-12))
        
        rate_load = k_fill * cai_clip * fill_gate
        T = (ca_e**n_trig) / (K_trig**n_trig + ca_e**n_trig + 1e-12)
        
        # Linear change in bucket level, strictly bounded [0, 1]
        dP = rate_load - (P / tau_leak) - (Vmax_CICR * P * T)
        P = min(1.0, max(0.0, P + dt * dP))

        # 3. CICR Release
        J_rel = Vmax_CICR * P * T
        decay_cicr = np.exp(-dt / tau_CICR)
        ca_cicr = ca_cicr * decay_cicr + J_rel * tau_CICR * (1.0 - decay_cicr)

        cai_total[i + 1]   = cai_1syn[i + 1] + ca_cicr
        prime_out[i + 1]   = P
        ca_cicr_out[i + 1] = ca_cicr
        effcai_out[i + 1]  = effcai

    if interp_dt > 0:
        cai_total   = np.interp(t_orig, t, cai_total)
        prime_out   = np.interp(t_orig, t, prime_out)
        ca_cicr_out = np.interp(t_orig, t, ca_cicr_out)
        effcai_out  = np.interp(t_orig, t, effcai_out)

    return {
        "cai_total": cai_total,
        "priming":   prime_out,
        "ca_er":     prime_out,
        "ca_cicr":   ca_cicr_out,
        "effcai_ci": effcai_out, 
    }


class PrimedJunctionalCICRModel(CICRModel):
    DESCRIPTION = "Primed-Junctional V4.2 (Strict Bounds + Slow Water Bucket)"

    FIT_PARAMS = [
        ("gamma_d_GB_GluSynapse", 10.0, 200.0),      
        ("gamma_p_GB_GluSynapse", 10.0, 300.0),      
        ("a00", 1.0, 5.0),             
        ("a01", 1.0, 5.0),             
        ("a10", 1.0, 5.0),
        ("a11", 1.0, 5.0),
        ("a20", 1.0, 10.0),
        ("a21", 1.0, 5.0),
        ("a30", 1.0, 10.0),
        ("a31", 1.0, 5.0),
        ("k_fill", 0.0001, 1.0),       # Bounded to ensure slow filling (water bucket)
        ("tau_leak", 5000.0, 300000.0),# 5 seconds to 5 minutes leak time
        ("K_clip", 0.0001, 0.001),     # Soft-clip limit 
        ("K_trig", 0.001, 0.005),      # Trigger Ca2+ threshold
        ("n_trig", 2.0, 15.0),         # Hill steepness
        ("Vmax_CICR", 0.001, 5.0),     # Max CICR rate 
        ("tau_CICR", 10.0, 1000.0),    # CICR calcium decay time
        ("g_cicr", 10.0, 5000.0),      # Reined-in gain multiplier to compensate for high gamma_p
        ("tau_eff", 50.0, 500.0),      # Leaky integrator tau
    ]

#   Best parameters:
#     gamma_d_GB_GluSynapse          =    68.9427  (default: 50.0000, +37.9%)
#     gamma_p_GB_GluSynapse          =    56.5987  (default: 50.0000, +13.2%)
#     a00                            =     1.0065  (default: 0.7130, +41.2%)
#     a01                            =     2.2490  (default: 2.1975, +2.3%)
#     a10                            =     1.0875  (default: 0.9384, +15.9%)
#     a11                            =     3.2048  (default: 3.4727, -7.7%)
#     a20                            =     1.6184  (default: 0.1469, +1001.7%)
#     a21                            =     2.0142  (default: 2.1019, -4.2%)
#     a30                            =     1.7612  (default: 1.1183, +57.5%)
#     a31                            =     2.5919  (default: 3.7722, -31.3%)
#     k_fill                         =     0.9651  (default: 27.8982, -96.5%)
#     tau_leak                       = 146935.4167  (default: 129306.8704, +13.6%)
#     K_clip                         =     0.0010  (default: 0.0006, +64.1%)
#     K_trig                         =     0.0035  (default: 0.0041, -13.9%)
#     n_trig                         =     5.5073  (default: 6.0495, -9.0%)
#     Vmax_CICR                      =     4.3122  (default: 3.7954, +13.6%)
#     tau_CICR                       =   741.2435  (default: 561.9705, +31.9%)
#     g_cicr                         =  4991.2752  (default: 86435.1983, -94.2%)
#     tau_eff                        =   466.8500  (default: 338.4028, +38.0%)

    DEFAULT_PARAMS = {
        "gamma_d_GB_GluSynapse": 68.9427, "gamma_p_GB_GluSynapse": 56.5987,
        "a00": 1.0065, "a01": 2.2490, "a10": 1.0875, "a11": 3.2048,
        "a20": 1.6184, "a21": 2.0142, "a30": 1.7612, "a31": 2.5919,
        "k_fill": 0.9651, "tau_leak": 146935.4167, 
        "K_clip": 0.0010, "K_trig": 0.0035, "n_trig": 5.5073,
        "Vmax_CICR": 4.3122, "tau_CICR": 741.2435,
        "g_cicr": 4991.2752, "tau_eff": 466.8500,
    }

    def unpack_params(self, x):
        gb = {
            'gamma_d': x[0], 'gamma_p': x[1],
            'a00': x[2], 'a01': x[3], 'a10': x[4], 'a11': x[5],
            'a20': x[6], 'a21': x[7], 'a30': x[8], 'a31': x[9],
        }
        cicr = {
            'k_fill': x[10], 'tau_leak': x[11], 'K_clip': x[12],
            'K_trig': x[13], 'n_trig': x[14], 'Vmax_CICR': x[15],
            'tau_CICR': x[16], 'g_cicr': x[17], 'tau_eff': x[18],
        }
        return gb, cicr

    def get_init_fn(self):
        def init_fn(cai_first, rho0):
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, rho0)
        return init_fn

    def get_debug_sim_fn(self):
        return _apply_primed_junctional_debug

    def get_step_factory(self):
        def scan_factory(params, syn_params):
            gb, cicr = params
            c_pre_opt, c_post_opt, is_apical, cai_pre, t_pre, cai_post, t_post = syn_params

            c_pre = jnp.max(compute_effcai_piecewise_linear_jax(cai_pre, t_pre, tau_effca=cicr['tau_eff']))
            c_post = jnp.max(compute_effcai_piecewise_linear_jax(cai_post, t_post, tau_effca=cicr['tau_eff']))

            theta_d = jnp.where(is_apical,
                                gb['a20']*c_pre + gb['a21']*c_post,
                                gb['a00']*c_pre + gb['a01']*c_post)
            raw_theta_p = jnp.where(is_apical,
                                    gb['a30']*c_pre + gb['a31']*c_post,
                                    gb['a10']*c_pre + gb['a11']*c_post)
            theta_p = jnp.maximum(raw_theta_p, theta_d + 0.01)

            cai_rest = 70e-6

            def scan_step(carry, inputs):
                P, ca_cicr, _d1, _d2, _d3, effcai, rho = carry
                cai_raw, dt = inputs

                ca_e = jnp.maximum(0.0, cai_raw - cai_rest)

                # 1. Integrator 
                eff_decay = jnp.exp(-dt / cicr['tau_eff'])
                effcai_new = effcai * eff_decay + ca_e * cicr['tau_eff'] * (1.0 - eff_decay)

                # 2. Slow Water Bucket (Linear Integration)
                fill_gate = jnp.where(effcai_new > theta_d, 1.0, 0.0)
                cai_clip = cicr['K_clip'] * jnp.tanh(ca_e / (cicr['K_clip'] + 1e-12))
                rate_load = cicr['k_fill'] * cai_clip * fill_gate
                T = (ca_e**cicr['n_trig']) / (cicr['K_trig']**cicr['n_trig'] + ca_e**cicr['n_trig'] + 1e-12)
                
                # Linear change, strictly clipped between 0 and 1
                dP = rate_load - (P / cicr['tau_leak']) - (cicr['Vmax_CICR'] * P * T)
                P_new = jnp.clip(P + dt * dP, 0.0, 1.0)

                # 3. CICR Release
                J_rel = cicr['Vmax_CICR'] * P_new * T
                decay_cicr = jnp.exp(-dt / cicr['tau_CICR'])
                ca_cicr_new = ca_cicr * decay_cicr + J_rel * cicr['tau_CICR'] * (1.0 - decay_cicr)

                # 4. Multiplicative Potentiation Modulator
                gamma_p_eff = gb['gamma_p'] + cicr['g_cicr'] * ca_cicr_new

                # 5. Graupner-Brunel Plasticity Core
                pot = jnp.where(effcai_new > theta_p, 1.0, 0.0)
                dep = jnp.where(effcai_new > theta_d, 1.0, 0.0)
                
                drho = (-rho*(1.0-rho)*(0.5-rho)
                        + gamma_p_eff*(1.0-rho)*pot
                        - gb['gamma_d']*rho*dep) / 70000.0
                        
                rho_new = jnp.where(dt > 0, jnp.clip(rho + dt * drho, 0.0, 1.0), rho)

                return (P_new, ca_cicr_new, 0.0, 0.0, 0.0, effcai_new, rho_new), None

            return scan_step
        return scan_factory


if __name__ == "__main__":
    PrimedJunctionalCICRModel().run()
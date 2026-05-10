#!/usr/bin/env python3
"""
Primed-Junctional CICR Model V4.4 — Slow Bucket + CaMKII Veto

Architecture: 
1. ER acts as a slow cumulative integrator (Water Bucket) using strict linear filling.
2. We integrate RAW clipped calcium, NOT effcai. Because raw calcium decays quickly, 
   the ER pump naturally turns off between spikes, creating a slow "staircase" fill.
3. Multiplicative CICR-Potentiation: Ca_CICR dynamically scales gamma_p.
4. CaMKII Veto (Exclusive GB): If the Ca2+ crosses the potentiation threshold, 
   it completely shuts off the depression mechanism, preventing the "tug-of-war".
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

        # 2. True Slow Water Bucket (Linear Integration of RAW Clipped Ca2+)
        cai_clip = K_clip * np.tanh(ca_e / (K_clip + 1e-12))
        rate_load = k_fill * cai_clip
        
        # Trigger
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
    DESCRIPTION = "Primed-Junctional V4.4 (CaMKII Veto)"

    FIT_PARAMS = [
        ("gamma_d_GB_GluSynapse", 50.0, 200.0),      
        ("gamma_p_GB_GluSynapse", 150.0, 300.0),      
        ("a00", 1.0, 5.0),             
        ("a01", 1.0, 5.0),             
        ("a10", 1.0, 5.0),
        ("a11", 1.0, 5.0),
        ("a20", 1.0, 10.0),
        ("a21", 1.0, 5.0),
        ("a30", 1.0, 10.0),
        ("a31", 1.0, 5.0),
        ("k_fill", 0.001, 2.0),        
        ("tau_leak", 30000.0, 300000.0), 
        ("K_clip", 0.0001, 0.001),     
        ("K_trig", 0.001, 0.005),      
        ("n_trig", 2.0, 15.0),         
        ("Vmax_CICR", 0.001, 5.0),     
        ("tau_CICR", 10.0, 1000.0),    
        ("g_cicr", 10.0, 50000.0),     
        ("tau_eff", 50.0, 500.0),      
    ]

#     {
#     "gamma_d_GB_GluSynapse": 121.35806451638638,
#     "gamma_p_GB_GluSynapse": 11.461585574347424,
#     "a00": 1.0095930400070956,
#     "a01": 1.614723868183439,
#     "a10": 1.137990905332909,
#     "a11": 2.919058755531716,
#     "a20": 1.5263215411235387,
#     "a21": 2.504172484911429,
#     "a30": 1.9027420183478547,
#     "a31": 3.405093590558612,
#     "k_fill": 1.8984663154123105,
#     "tau_leak": 127032.82039063737,
#     "K_clip": 0.0009498389272326957,
#     "K_trig": 0.0034740372715496492,
#     "n_trig": 5.8083918779782095,
#     "Vmax_CICR": 4.896536300272658,
#     "tau_CICR": 742.017220790156,
#     "g_cicr": 21720.533932450904,
#     "tau_eff": 452.6690380265944
# }
    DEFAULT_PARAMS = {
        "gamma_d_GB_GluSynapse": 121.35806451638638, "gamma_p_GB_GluSynapse": 181.461585574347424,
        "a00": 1.0095930400070956, "a01": 1.614723868183439, "a10": 1.137990905332909, "a11": 2.919058755531716,
        "a20": 1.5263215411235387, "a21": 2.504172484911429, "a30": 1.9027420183478547, "a31": 3.405093590558612,
        "k_fill": 1.8984663154123105, "tau_leak": 127032.82039063737, 
        "K_clip": 0.0009498389272326957, "K_trig": 0.0034740372715496492, "n_trig": 5.8083918779782095,
        "Vmax_CICR": 4.896536300272658, "tau_CICR": 742.017220790156,
        "g_cicr": 21720.533932450904, "tau_eff": 452.6690380265944,
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
                cai_raw, dt, _nves = inputs  # _nves unused; model uses Ca-driven IP3

                ca_e = jnp.maximum(0.0, cai_raw - cai_rest)

                # 1. Integrator 
                eff_decay = jnp.exp(-dt / cicr['tau_eff'])
                effcai_new = effcai * eff_decay + ca_e * cicr['tau_eff'] * (1.0 - eff_decay)

                # 2. True Slow Water Bucket
                cai_clip = cicr['K_clip'] * jnp.tanh(ca_e / (cicr['K_clip'] + 1e-12))
                rate_load = cicr['k_fill'] * cai_clip
                
                T = (ca_e**cicr['n_trig']) / (cicr['K_trig']**cicr['n_trig'] + ca_e**cicr['n_trig'] + 1e-12)
                
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
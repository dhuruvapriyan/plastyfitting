#!/usr/bin/env python3
"""
CICR Model V5 — ER Store Dynamics with IP3R/RyR Competition

Carry = (IP3, P, ca_ryr, effcai, rho)

  Ca_e(t)    = max(0, cai_raw(t) - cai_rest)
  dIP3/dt    = delta_IP3 · Ca_e/(K_PLC + Ca_e) - IP3/tau_IP3
  dP/dt      = phi_serca · Hill_H(Ca_e)·(1-P) - J_RyR - J_IP3R - k_leak·(P - P_ss)
  dCa_RyR/dt = eta · J_RyR - Ca_RyR/tau_RyR
  effcai     = leaky_integrator(Ca_e + Ca_RyR, tau_eff)
  dρ/dt      = GB(effcai, γ_p + g_RyR·Ca_RyR, γ_d)

  Hill_H(x)  = x^4 / (K_H^4 + x^4)
  Hill_T(x)  = x^8 / (K_T^8 + x^8)
  J_RyR      = V_RyR · P · Hill_T(Ca_e)
  J_IP3R     = V_IP3R · P · IP3/(K_IP3+IP3) · Ca_e/(K_ACT+Ca_e)

Literature-fixed constants (14):
  n_H    = 4        Satoh 2011 (5.7 in vivo / ~2 in vitro, compromise)
  K_H    = 0.4 µM   Satoh 2011 Kd=0.41µM; Lytton 1992 K½≈0.4µM
  K_ACT  = 0.4 µM   Neymotin 2015 Table 1 kact=0.0004 mM
  K_IP3  = 1.0      abstract units (physical: ~0.13µM, Neymotin kIP3)
  K_PLC  = 0.5 µM   standard PLC Ca²⁺ cofactor
  P_SS   = 0.15     resting store fraction
  TAU_RYR= 100 ms   RyR Ca²⁺ clearance
  CAI_REST=70 nM    resting [Ca²⁺]
  ETA_RYR= 1.0      release-to-cytosol gain
  n_T    = 8        RyR cluster cooperativity (V4.3 fit: n_trig≈5.9)
  K_T    = 2.5 µM   midpoint for coincidence discrimination (3µM vs 1µM)
  k_leak = 1e-5     Hong & Ross: AP priming 2-3min w/o IP3 → τ≈100s
  tau_IP3= 1000 ms  Caya-Bissonnette Table S1: τ_ER=999.25 ms

Free parameters (16):
  10 GB (gamma_d, gamma_p, a00-a31)
  + delta_IP3, phi_serca, V_RyR, V_IP3R, g_RyR, tau_eff
"""

import numpy as np
import jax.numpy as jnp
from cicr_common import CICRModel, compute_effcai_piecewise_linear_jax

N_H = 4
K_H = 0.0004
K_ACT = 0.0004
K_IP3 = 1.0
K_PLC = 0.0005
P_SS = 0.15
TAU_RYR = 100.0
CAI_REST = 70e-6
ETA_RYR = 1.0
N_T = 8.0
K_T = 0.0025
K_LEAK = 1e-5
TAU_IP3 = 1000.0
TRACE_TAU_REF = 200.0

K_H_NH = K_H ** N_H
K_T_NT = K_T ** N_T


def _v5_debug(cai_1syn, t, cp, cq, is_apical, dp):
    delta_IP3 = float(dp["delta_IP3"])
    phi_serca = float(dp["phi_serca"])
    V_RyR     = float(dp["V_RyR"])
    V_IP3R    = float(dp["V_IP3R"])
    g_RyR     = float(dp["g_RyR"])
    tau_eff   = float(dp["tau_eff"])
    gamma_d   = float(dp.get("gamma_d", dp.get("gamma_d_GB_GluSynapse", 150.0)))
    gamma_p   = float(dp.get("gamma_p", dp.get("gamma_p_GB_GluSynapse", 150.0)))

    scale = tau_eff / TRACE_TAU_REF
    cp_s, cq_s = cp * scale, cq * scale
    if is_apical:
        theta_d = float(dp["a20"]) * cp_s + float(dp["a21"]) * cq_s
        theta_p = float(dp["a30"]) * cp_s + float(dp["a31"]) * cq_s
    else:
        theta_d = float(dp["a00"]) * cp_s + float(dp["a01"]) * cq_s
        theta_p = float(dp["a10"]) * cp_s + float(dp["a11"]) * cq_s
    theta_p = max(theta_p, theta_d + 0.01)

    n = len(cai_1syn)
    cai_total  = np.copy(cai_1syn)
    ip3_out    = np.zeros(n)
    p_out      = np.zeros(n)
    ca_ryr_out = np.zeros(n)
    effcai_out = np.zeros(n)
    rho_out    = np.zeros(n)

    IP3     = 0.0
    P       = P_SS
    ca_ryr  = 0.0
    effcai  = 0.0
    rho     = float(dp.get("rho0", 0.5))

    ip3_out[0], p_out[0], ca_ryr_out[0], effcai_out[0], rho_out[0] = IP3, P, ca_ryr, effcai, rho

    for i in range(n - 1):
        dt = t[i + 1] - t[i]
        if dt <= 0:
            ip3_out[i+1], p_out[i+1], ca_ryr_out[i+1] = IP3, P, ca_ryr
            effcai_out[i+1], rho_out[i+1] = effcai, rho
            cai_total[i+1] = cai_1syn[i+1] + ca_ryr
            continue

        ca_e = max(0.0, cai_1syn[i] - CAI_REST)

        dIP3 = delta_IP3 * ca_e / (K_PLC + ca_e + 1e-12) - IP3 / TAU_IP3
        IP3 = max(0.0, IP3 + dt * dIP3)

        T = ca_e ** N_T / (K_T_NT + ca_e ** N_T + 1e-30)
        S = ca_e ** N_H / (K_H_NH + ca_e ** N_H + 1e-30)

        J_ryr  = V_RyR * P * T
        J_ip3r = V_IP3R * P * (IP3 / (K_IP3 + IP3 + 1e-12)) * (ca_e / (K_ACT + ca_e + 1e-12))

        dP = phi_serca * S * (1.0 - P) - J_ryr - J_ip3r - K_LEAK * (P - P_SS)
        P = min(1.0, max(0.0, P + dt * dP))

        decay_ryr = np.exp(-dt / TAU_RYR)
        ca_ryr = ca_ryr * decay_ryr + ETA_RYR * J_ryr * TAU_RYR * (1.0 - decay_ryr)

        eff_decay = np.exp(-dt / tau_eff)
        effcai = effcai * eff_decay + (ca_e + ca_ryr) * tau_eff * (1.0 - eff_decay)

        gp_eff = gamma_p + g_RyR * ca_ryr
        pot = 1.0 if effcai > theta_p else 0.0
        dep = 1.0 if (effcai > theta_d and pot == 0.0) else 0.0
        drho = (-rho * (1 - rho) * (0.5 - rho)
                + gp_eff * (1 - rho) * pot
                - gamma_d * rho * dep) / 70000.0
        rho = min(1.0, max(0.0, rho + dt * drho))

        cai_total[i+1]  = cai_1syn[i+1] + ca_ryr
        ip3_out[i+1]    = IP3
        p_out[i+1]      = P
        ca_ryr_out[i+1] = ca_ryr
        effcai_out[i+1] = effcai
        rho_out[i+1]    = rho

    return {
        "cai_total": cai_total,
        "priming":   ip3_out,
        "ca_er":     p_out,
        "ca_cicr":   ca_ryr_out,
        "effcai":    effcai_out,
        "rho":       rho_out,
    }


class ERDynamicsCICRModel(CICRModel):
    DESCRIPTION = "V5 ER Dynamics (IP3R/RyR Competition, 16 free)"

    FIT_PARAMS = [
        ("gamma_d_GB_GluSynapse", 50.0,  250.0),
        ("gamma_p_GB_GluSynapse", 50.0,  300.0),
        ("a00",        1.0,   2.5),
        ("a01",        1.0,   5.0),
        ("a10",        1.0,   5.0),
        ("a11",        1.0,   5.0),
        ("a20",        1.0,   2.5),
        ("a21",        1.0,   5.0),
        ("a30",        1.0,  10.0),
        ("a31",        1.0,   5.0),
        ("delta_IP3",  0.01,  10.0),
        ("phi_serca",  1e-4,  0.1),
        ("V_RyR",      1e-4,  0.1),
        ("V_IP3R",     1e-5,  0.05),
        ("g_RyR",      10.0, 50000.0),
        ("tau_eff",    50.0,  500.0),
    ]

    DEFAULT_PARAMS = {
        "gamma_d_GB_GluSynapse": 152.0,
        "gamma_p_GB_GluSynapse": 151.0,
        "a00": 1.01, "a01": 1.59, "a10": 1.12, "a11": 2.87,
        "a20": 1.87, "a21": 3.28, "a30": 2.02, "a31": 3.10,
        "delta_IP3": 1.0,
        "phi_serca": 0.005,
        "V_RyR":     0.005,
        "V_IP3R":    0.001,
        "g_RyR":     5000.0,
        "tau_eff":   300.0,
    }

    def unpack_params(self, x):
        gb = {
            'gamma_d': x[0], 'gamma_p': x[1],
            'a00': x[2], 'a01': x[3], 'a10': x[4], 'a11': x[5],
            'a20': x[6], 'a21': x[7], 'a30': x[8], 'a31': x[9],
        }
        cicr = {
            'delta_IP3': x[10], 'phi_serca': x[11],
            'V_RyR': x[12], 'V_IP3R': x[13],
            'g_RyR': x[14], 'tau_eff': x[15],
        }
        return gb, cicr

    def get_init_fn(self):
        def init_fn(cai_first, rho0):
            return (0.0, P_SS, 0.0, 0.0, rho0)
        return init_fn

    def get_debug_sim_fn(self):
        return _v5_debug

    def get_step_factory(self):
        def scan_factory(params, syn_params):
            gb, cicr = params
            c_pre_opt, c_post_opt, is_apical, cai_pre, t_pre, cai_post, t_post = syn_params

            c_pre = jnp.max(compute_effcai_piecewise_linear_jax(cai_pre, t_pre, tau_effca=cicr['tau_eff']))
            c_post = jnp.max(compute_effcai_piecewise_linear_jax(cai_post, t_post, tau_effca=cicr['tau_eff']))

            theta_d = jnp.where(is_apical,
                                gb['a20'] * c_pre + gb['a21'] * c_post,
                                gb['a00'] * c_pre + gb['a01'] * c_post)
            raw_theta_p = jnp.where(is_apical,
                                    gb['a30'] * c_pre + gb['a31'] * c_post,
                                    gb['a10'] * c_pre + gb['a11'] * c_post)
            theta_p = jnp.maximum(raw_theta_p, theta_d + 0.01)

            def scan_step(carry, inputs):
                IP3, P, ca_ryr, effcai, rho = carry
                cai_raw, dt, _nves = inputs  # _nves unused; model uses Ca-driven IP3

                ca_e = jnp.maximum(0.0, cai_raw - CAI_REST)

                dIP3 = cicr['delta_IP3'] * ca_e / (K_PLC + ca_e + 1e-12) - IP3 / TAU_IP3
                IP3_new = jnp.maximum(0.0, IP3 + dt * dIP3)

                T = ca_e ** N_T / (K_T_NT + ca_e ** N_T + 1e-30)
                S = ca_e ** N_H / (K_H_NH + ca_e ** N_H + 1e-30)

                J_ryr  = cicr['V_RyR'] * P * T
                J_ip3r = (cicr['V_IP3R'] * P
                          * (IP3_new / (K_IP3 + IP3_new + 1e-12))
                          * (ca_e / (K_ACT + ca_e + 1e-12)))

                dP = (cicr['phi_serca'] * S * (1.0 - P)
                      - J_ryr - J_ip3r
                      - K_LEAK * (P - P_SS))
                P_new = jnp.clip(P + dt * dP, 0.0, 1.0)

                decay_ryr = jnp.exp(-dt / TAU_RYR)
                ca_ryr_new = ca_ryr * decay_ryr + ETA_RYR * J_ryr * TAU_RYR * (1.0 - decay_ryr)

                eff_decay = jnp.exp(-dt / cicr['tau_eff'])
                effcai_new = effcai * eff_decay + (ca_e + ca_ryr_new) * cicr['tau_eff'] * (1.0 - eff_decay)

                gp_eff = gb['gamma_p'] + cicr['g_RyR'] * ca_ryr_new
                pot = jnp.where(effcai_new > theta_p, 1.0, 0.0)
                dep = jnp.where((effcai_new > theta_d) & (effcai_new <= theta_p), 1.0, 0.0)

                drho = (-rho * (1.0 - rho) * (0.5 - rho)
                        + gp_eff * (1.0 - rho) * pot
                        - gb['gamma_d'] * rho * dep) / 70000.0
                rho_new = jnp.where(dt > 0, jnp.clip(rho + dt * drho, 0.0, 1.0), rho)

                return (IP3_new, P_new, ca_ryr_new, effcai_new, rho_new), None

            return scan_step
        return scan_factory


if __name__ == "__main__":
    ERDynamicsCICRModel().run()
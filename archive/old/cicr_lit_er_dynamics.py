#!/usr/bin/env python3
"""
CICR Model V5.1 — ER Store Dynamics with IP3R/RyR Competition
16 free parameters (down from 20), 4 newly fixed from literature.

Equations:
  Ca_e     = max(0, cai_raw - cai_rest)
  dIP3/dt  = delta_IP3 * Ca_e/(K_PLC + Ca_e) - IP3/tau_IP3
  dP/dt    = phi_serca * Hill(Ca_e) * (1-P) - J_RyR - J_IP3R - k_leak*(P - P_ss)
  dCa_RyR/dt = eta_RyR * J_RyR - Ca_RyR/tau_RyR
  drho/dt  = GB(effcai, gamma_p_eff, gamma_d)

  J_RyR  = V_RyR * P * Ca_e^n_T / (K_T^n_T + Ca_e^n_T)
  J_IP3R = V_IP3R * P * IP3/(K_IP3+IP3) * Ca_e/(K_act+Ca_e)
  Hill(x) = x^n_H / (K_H^n_H + x^n_H)
  gamma_p_eff = gamma_p + g_RyR * Ca_RyR

Literature-fixed constants:
  n_H   = 4         SERCA Hill coeff         Satoh 2011 (in vivo n=5.7, in vitro n~2, compromise=4)
  K_H   = 0.0004    SERCA Kd = 0.4 uM       Satoh 2011, Lytton 1992
  K_act = 0.0004    IP3R Ca co-agonist       Neymotin 2015 Table 1 (kact=0.4 uM)
  K_IP3 = 1.0       IP3 half-activation      abstract units (normalized)
  K_PLC = 0.0005    PLC Ca cofactor          Hashimotodani 2005
  P_ss  = 0.15      resting store fraction
  tau_RyR = 100     RyR Ca decay             fast cytosolic clearance
  cai_rest = 70e-6  resting [Ca]_i = 70 nM

Newly fixed from literature (previously free):
  n_T     = 8       RyR Hill cooperativity   steep threshold for coincidence detection
  K_T     = 0.002   RyR half-act = 2 uM     best fit landed at 1.7 uM, fix at 2
  k_leak  = 7e-6    passive ER leak rate     Hong & Ross 2007: AP priming persists 2-3 min
                                              without mGluR -> no IP3 -> P decays only via leak
                                              1/k_leak ~ 143s matches 2-3 min retention
  tau_IP3 = 1000    IP3 decay time           Caya-Bissonnette 2023 Table S1: tau_ER = 999.25 ms
                                              sets the ~1s BTSP eligibility window
"""

import numpy as np
import jax.numpy as jnp
from cicr_common import CICRModel, compute_effcai_piecewise_linear_jax

# --- Literature-fixed constants (original set) ---
N_H = 4                # Satoh 2011
K_H = 0.0004           # Satoh 2011, Lytton 1992: SERCA Kd = 0.4 uM
K_ACT = 0.0004         # Neymotin 2015 Table 1: IP3R Ca co-agonist
K_IP3 = 1.0            # abstract/normalized
K_PLC = 0.0005         # PLC Ca cofactor
P_SS = 0.15            # resting store fraction
TAU_RYR = 100.0        # fast cytosolic clearance
CAI_REST = 70e-6       # resting [Ca]_i = 70 nM
TRACE_TAU_REF = 200.0  # reference tau for trace recording

# --- Newly fixed from literature (were free in v5.0) ---
N_T = 8                # RyR Hill cooperativity
K_T = 0.002            # RyR half-activation = 2 uM
K_LEAK = 7e-6          # Hong & Ross 2007: 1/k_leak ~ 143s
TAU_IP3 = 1000.0       # Caya-Bissonnette 2023: tau_ER = 999.25 ms


def _hill(x, K, n):
    xn = x**n
    return xn / (K**n + xn + 1e-30)


def _v5_debug(cai_1syn, t, cp, cq, is_apical, dp):
    delta_IP3 = float(dp["delta_IP3"])
    phi_serca = float(dp["phi_serca"])
    V_RyR     = float(dp["V_RyR"])
    V_IP3R    = float(dp["V_IP3R"])
    g_RyR     = float(dp["g_RyR"])
    tau_eff   = float(dp["tau_eff"])
    gamma_d   = float(dp.get("gamma_d_GB_GluSynapse", 150.0))
    gamma_p   = float(dp.get("gamma_p_GB_GluSynapse", 150.0))

    scale = tau_eff / TRACE_TAU_REF
    cp_s, cq_s = cp * scale, cq * scale
    if is_apical:
        theta_d = dp["a20"] * cp_s + dp["a21"] * cq_s
        theta_p = dp["a30"] * cp_s + dp["a31"] * cq_s
    else:
        theta_d = dp["a00"] * cp_s + dp["a01"] * cq_s
        theta_p = dp["a10"] * cp_s + dp["a11"] * cq_s
    theta_p = max(theta_p, theta_d + 0.01)

    n = len(cai_1syn)
    cai_total = np.copy(cai_1syn)
    ip3_out   = np.zeros(n)
    p_out     = np.zeros(n)
    ca_ryr_out = np.zeros(n)
    effcai_out = np.zeros(n)
    rho_out   = np.zeros(n)

    IP3, P, ca_ryr, effcai = 0.0, P_SS, 0.0, 0.0
    rho = float(dp.get("rho0", 0.5))

    ip3_out[0] = IP3; p_out[0] = P; ca_ryr_out[0] = ca_ryr
    effcai_out[0] = effcai; rho_out[0] = rho

    for i in range(n - 1):
        dt = t[i+1] - t[i]
        if dt <= 0:
            ip3_out[i+1] = IP3; p_out[i+1] = P; ca_ryr_out[i+1] = ca_ryr
            effcai_out[i+1] = effcai; rho_out[i+1] = rho
            cai_total[i+1] = cai_1syn[i+1] + ca_ryr
            continue

        ca_e = max(0.0, cai_1syn[i] - CAI_REST)

        dIP3 = delta_IP3 * ca_e / (K_PLC + ca_e + 1e-12) - IP3 / TAU_IP3
        IP3 = max(0.0, IP3 + dt * dIP3)

        T = _hill(ca_e, K_T, N_T)
        S = _hill(ca_e, K_H, N_H)
        J_ryr  = V_RyR * P * T
        J_ip3r = V_IP3R * P * (IP3 / (K_IP3 + IP3)) * (ca_e / (K_ACT + ca_e + 1e-12))

        dP = phi_serca * S * (1.0 - P) - J_ryr - J_ip3r - K_LEAK * (P - P_SS)
        P = min(1.0, max(0.0, P + dt * dP))

        decay_ryr = np.exp(-dt / TAU_RYR)
        ca_ryr = ca_ryr * decay_ryr + J_ryr * TAU_RYR * (1.0 - decay_ryr)

        eff_decay = np.exp(-dt / tau_eff)
        ca_total_i = ca_e + ca_ryr
        effcai = effcai * eff_decay + ca_total_i * tau_eff * (1.0 - eff_decay)

        gp_eff = gamma_p + g_RyR * ca_ryr
        pot = 1.0 if effcai > theta_p else 0.0
        dep = 1.0 if effcai > theta_d else 0.0
        drho = (-rho*(1-rho)*(0.5-rho) + gp_eff*(1-rho)*pot - gamma_d*rho*dep) / 70000.0
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
    DESCRIPTION = "V5.1 ER Dynamics (IP3R/RyR, 16-param literature-constrained)"

    FIT_PARAMS = [
        ("gamma_d_GB_GluSynapse", 50.0, 250.0),
        ("gamma_p_GB_GluSynapse", 50.0, 300.0),
        ("a00", 1.0, 2.5),
        ("a01", 1.0, 5.0),
        ("a10", 1.0, 5.0),
        ("a11", 1.0, 5.0),
        ("a20", 1.0, 2.5),
        ("a21", 1.0, 5.0),
        ("a30", 1.0, 10.0),
        ("a31", 1.0, 5.0),
        ("delta_IP3", 0.01, 5.0),
        ("phi_serca", 0.0001, 0.05),
        ("V_RyR", 0.0001, 0.05),
        ("V_IP3R", 0.0001, 0.01),
        ("g_RyR", 10.0, 50000.0),
        ("tau_eff", 50.0, 500.0),
    ]

#     {
#     "gamma_d_GB_GluSynapse": 238.82466466435795,
#     "gamma_p_GB_GluSynapse": 275.7381402363067,
#     "a00": 1.1167225987989537,
#     "a01": 3.716004358311825,
#     "a10": 1.4735850418779937,
#     "a11": 3.899906068377136,
#     "a20": 1.9528641629235532,
#     "a21": 4.551828307416708,
#     "a30": 8.91203652022043,
#     "a31": 4.788147047700919,
#     "delta_IP3": 2.5422401862098454,
#     "phi_serca": 0.03775530047715139,
#     "V_RyR": 0.006681962363681571,
#     "V_IP3R": 0.005795126148092674,
#     "g_RyR": 1482.2542429337445,
#     "tau_eff": 281.56253162406966
# }
    DEFAULT_PARAMS = {
        "gamma_d_GB_GluSynapse": 238.82466466435795, "gamma_p_GB_GluSynapse": 275.7381402363067,
        "a00": 1.1167225987989537, "a01": 3.716004358311825, "a10": 1.4735850418779937, "a11": 3.899906068377136,
        "a20": 1.9528641629235532, "a21": 4.551828307416708, "a30": 8.91203652022043, "a31": 4.788147047700919,
        "delta_IP3": 2.5422401862098454,
        "phi_serca": 0.03775530047715139,
        "V_RyR": 0.006681962363681571,
        "V_IP3R": 0.005795126148092674,
        "g_RyR": 1482.2542429337445,
        "tau_eff": 281.56253162406966,
    }

    def unpack_params(self, x):
        gb = {
            'gamma_d': x[0], 'gamma_p': x[1],
            'a00': x[2], 'a01': x[3], 'a10': x[4], 'a11': x[5],
            'a20': x[6], 'a21': x[7], 'a30': x[8], 'a31': x[9],
        }
        cicr = {
            'delta_IP3': x[10],
            'phi_serca': x[11],
            'V_RyR': x[12],
            'V_IP3R': x[13],
            'g_RyR': x[14],
            'tau_eff': x[15],
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
                                gb['a20']*c_pre + gb['a21']*c_post,
                                gb['a00']*c_pre + gb['a01']*c_post)
            raw_theta_p = jnp.where(is_apical,
                                    gb['a30']*c_pre + gb['a31']*c_post,
                                    gb['a10']*c_pre + gb['a11']*c_post)
            theta_p = jnp.maximum(raw_theta_p, theta_d + 0.01)

            def scan_step(carry, inputs):
                IP3, P, ca_ryr, effcai, rho = carry
                cai_raw, dt, _nves = inputs  # _nves unused; model uses Ca-driven IP3

                ca_e = jnp.maximum(0.0, cai_raw - CAI_REST)

                dIP3 = cicr['delta_IP3'] * ca_e / (K_PLC + ca_e + 1e-12) - IP3 / TAU_IP3
                IP3_new = jnp.maximum(0.0, IP3 + dt * dIP3)

                ca_e_nT = ca_e ** N_T
                K_T_nT  = K_T ** N_T
                T = ca_e_nT / (K_T_nT + ca_e_nT + 1e-30)

                ca_e_nH = ca_e ** N_H
                K_H_nH  = K_H ** N_H
                S = ca_e_nH / (K_H_nH + ca_e_nH + 1e-30)

                J_ryr  = cicr['V_RyR'] * P * T
                J_ip3r = cicr['V_IP3R'] * P * (IP3_new / (K_IP3 + IP3_new + 1e-12)) * (ca_e / (K_ACT + ca_e + 1e-12))

                dP = cicr['phi_serca'] * S * (1.0 - P) - J_ryr - J_ip3r - K_LEAK * (P - P_SS)
                P_new = jnp.clip(P + dt * dP, 0.0, 1.0)

                decay_ryr = jnp.exp(-dt / TAU_RYR)
                ca_ryr_new = ca_ryr * decay_ryr + J_ryr * TAU_RYR * (1.0 - decay_ryr)

                eff_decay = jnp.exp(-dt / cicr['tau_eff'])
                ca_total = ca_e + ca_ryr_new
                effcai_new = effcai * eff_decay + ca_total * cicr['tau_eff'] * (1.0 - eff_decay)

                gp_eff = gb['gamma_p'] + cicr['g_RyR'] * ca_ryr_new
                pot = jnp.where(effcai_new > theta_p, 1.0, 0.0)
                dep = jnp.where(effcai_new > theta_d, 1.0, 0.0)

                drho = (-rho*(1.0-rho)*(0.5-rho)
                        + gp_eff*(1.0-rho)*pot
                        - gb['gamma_d']*rho*dep) / 70000.0
                rho_new = jnp.where(dt > 0, jnp.clip(rho + dt * drho, 0.0, 1.0), rho)

                return (IP3_new, P_new, ca_ryr_new, effcai_new, rho_new), None

            return scan_step
        return scan_factory


if __name__ == "__main__":
    ERDynamicsCICRModel().run()
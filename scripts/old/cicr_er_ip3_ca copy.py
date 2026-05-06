#!/usr/bin/env python3
"""
CICR Model V5+IP3R — faithful JAX replica of GluSynapse_CICR.mod

IP3 produced as discrete bolus on vesicle release (Nves_CICR), NOT Ca-driven.
ca_ip3r feeds into effcai alongside ca_ryr. g_IP3R optionally boosts gamma_p.
"""

import numpy as np
import jax.numpy as jnp
from cicr_common import CICRModel, compute_effcai_piecewise_linear_jax

N_H = 4; K_H = 0.0004; K_ACT = 0.0003; K_IP3 = 1.0; P_SS = 0.3; CAI_REST = 70e-6
TRACE_TAU_REF = 200.0

def _hill(x, K, n):
    xn = x**n; return xn / (K**n + xn + 1e-30)

def _v5ip3r_debug(cai_1syn, t, cp, cq, is_apical, dp):
    """Numpy debug sim — matches MOD DERIVATIVE block exactly."""
    delta_IP3  = float(dp["delta_IP3"]); tau_IP3 = float(dp["tau_IP3"])
    phi_serca  = float(dp["phi_serca"]); k_leak  = float(dp["k_leak"])
    V_RyR = float(dp["V_RyR"]); K_T = float(dp["K_T"]); n_T = float(dp["n_T"])
    V_IP3R = float(dp["V_IP3R"]); g_RyR = float(dp["g_RyR"]); g_IP3R = float(dp.get("g_IP3R", 0.0))
    tau_RyR = float(dp.get("tau_RyR", 100.0)); tau_IP3R = float(dp.get("tau_IP3R", 100.0))
    tau_eff = float(dp["tau_eff"])
    gamma_d = float(dp.get("gamma_d_GB_GluSynapse", 150.0)); gamma_p = float(dp.get("gamma_p_GB_GluSynapse", 150.0))
    scale = tau_eff / TRACE_TAU_REF; cp_s, cq_s = cp * scale, cq * scale
    if is_apical: theta_d = dp["a20"]*cp_s + dp["a21"]*cq_s; theta_p = dp["a30"]*cp_s + dp["a31"]*cq_s
    else: theta_d = dp["a00"]*cp_s + dp["a01"]*cq_s; theta_p = dp["a10"]*cp_s + dp["a11"]*cq_s
    theta_p = max(theta_p, theta_d + 0.01)
    _nves_raw = dp.get("_nves", None)
    if _nves_raw is None or np.ndim(_nves_raw) == 0:
        nves = np.zeros(len(t))
    else:
        nves = np.asarray(_nves_raw, dtype=np.float64)

    n = len(cai_1syn)
    cai_total, ip3_out, p_out = np.copy(cai_1syn), np.zeros(n), np.zeros(n)
    ca_ryr_out, ca_ip3r_out, effcai_out, rho_out = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    IP3, P, ca_ryr, ca_ip3r, effcai = 0.0, P_SS, 0.0, 0.0, 0.0
    rho = float(dp.get("rho0", 0.5))
    ip3_out[0]=IP3; p_out[0]=P; rho_out[0]=rho

    for i in range(n-1):
        dt = t[i+1] - t[i]
        if dt <= 0:
            ip3_out[i+1]=IP3; p_out[i+1]=P; ca_ryr_out[i+1]=ca_ryr; ca_ip3r_out[i+1]=ca_ip3r
            effcai_out[i+1]=effcai; rho_out[i+1]=rho; cai_total[i+1]=cai_1syn[i+1]+ca_ryr+ca_ip3r; continue
        ca_e = max(0.0, cai_1syn[i] - CAI_REST)
        # IP3: bolus + decay (MOD: IP3 += delta_IP3 * Nves in NET_RECEIVE, then IP3' = -IP3/tau_IP3)
        IP3 = IP3 + delta_IP3 * nves[i]
        IP3 = max(0.0, IP3 + dt * (-IP3 / tau_IP3))
        T = _hill(ca_e, K_T, n_T); S = _hill(ca_e, K_H, N_H)
        J_ryr = V_RyR * P * T
        ipg = IP3 / (K_IP3 + IP3 + 1e-12); cag = ca_e / (K_ACT + ca_e + 1e-12)
        J_ip3r = V_IP3R * P * ipg * cag
        J_serca = phi_serca * S * (1.0 - P); J_leak = k_leak * (P - P_SS)
        P = min(1.0, max(0.0, P + dt * (J_serca - J_ryr - J_ip3r - J_leak)))
        decay_ryr = np.exp(-dt / tau_RyR); ca_ryr = ca_ryr * decay_ryr + J_ryr * tau_RyR * (1.0 - decay_ryr)
        decay_ip3r = np.exp(-dt / tau_IP3R); ca_ip3r = ca_ip3r * decay_ip3r + J_ip3r * tau_IP3R * (1.0 - decay_ip3r)
        eff_decay = np.exp(-dt / tau_eff)
        effcai = effcai * eff_decay + (ca_e + ca_ryr + ca_ip3r) * tau_eff * (1.0 - eff_decay)
        gp_eff = gamma_p + g_RyR * ca_ryr + g_IP3R * ca_ip3r
        pot = 1.0 if effcai > theta_p else 0.0; dep = 1.0 if effcai > theta_d else 0.0
        drho = (-rho*(1-rho)*(0.5-rho) + gp_eff*(1-rho)*pot - gamma_d*rho*dep) / 70000.0
        rho = min(1.0, max(0.0, rho + dt * drho))
        cai_total[i+1]=cai_1syn[i+1]+ca_ryr+ca_ip3r; ip3_out[i+1]=IP3; p_out[i+1]=P
        ca_ryr_out[i+1]=ca_ryr; ca_ip3r_out[i+1]=ca_ip3r; effcai_out[i+1]=effcai; rho_out[i+1]=rho

    return {"cai_total": cai_total, "priming": ip3_out, "ca_er": p_out,
            "ca_cicr": ca_ryr_out + ca_ip3r_out, "ca_ryr": ca_ryr_out, "ca_ip3r": ca_ip3r_out,
            "effcai": effcai_out, "rho": rho_out}


class ERDynamicsCICRModel(CICRModel):
    DESCRIPTION = "V5+IP3R ER Dynamics (discrete bolus)"
    FIT_PARAMS = [
        ("gamma_d_GB_GluSynapse", 50.0, 250.0), ("gamma_p_GB_GluSynapse", 50.0, 300.0),
        ("a00", 1.0, 5.0), ("a01", 1.0, 5.0), ("a10", 1.0, 5.0), ("a11", 1.0, 5.0),
        ("a20", 1.0, 5.0), ("a21", 1.0, 5.0), ("a30", 1.0, 10.0), ("a31", 1.0, 5.0),
        ("delta_IP3", 0.01, 10.0), ("tau_IP3", 500.0, 8000.0),
        ("phi_serca", 1e-5, 0.05), ("k_leak", 1e-6, 0.005),
        ("V_RyR", 1e-5, 0.05), ("K_T", 0.0005, 0.005), ("n_T", 6.0, 15.0),
        ("V_IP3R", 1e-5, 0.05), ("g_RyR", 10.0, 50000.0), ("g_IP3R", 0.0, 50000.0),
        ("tau_RyR", 20.0, 500.0), ("tau_IP3R", 20.0, 500.0), ("tau_eff", 50.0, 500.0),
    ]
    DEFAULT_PARAMS = {
        "gamma_d_GB_GluSynapse": 50.6589, "gamma_p_GB_GluSynapse": 261.5788,
        "a00": 1.0012, "a01": 1.5214, "a10": 2.3423, "a11": 4.8482,
        "a20": 1.0944, "a21": 1.5690, "a30": 3.5827, "a31": 4.7627,
        "delta_IP3": 9.9306, "tau_IP3": 2475.0241, "phi_serca": 1.006e-5, "k_leak": 1.141e-4,
        "V_RyR": 1.9e-3, "K_T": 1.600e-3, "n_T": 10.3077, "V_IP3R": 2.0e-4,
        "g_RyR": 2958.9394, "g_IP3R": 4031.4515, "tau_RyR": 60.4743, "tau_IP3R": 20.3999, "tau_eff": 50.0422,
    }

    def unpack_params(self, x):
        gb = {'gamma_d':x[0],'gamma_p':x[1],'a00':x[2],'a01':x[3],'a10':x[4],'a11':x[5],'a20':x[6],'a21':x[7],'a30':x[8],'a31':x[9]}
        cicr = {'delta_IP3':x[10],'tau_IP3':x[11],'phi_serca':x[12],'k_leak':x[13],
                'V_RyR':x[14],'K_T':x[15],'n_T':x[16],'V_IP3R':x[17],
                'g_RyR':x[18],'g_IP3R':x[19],'tau_RyR':x[20],'tau_IP3R':x[21],'tau_eff':x[22]}
        return gb, cicr

    def get_init_fn(self):
        def init_fn(cai_first, rho0): return (0.0, P_SS, 0.0, 0.0, 0.0, rho0)  # IP3, P, ca_ryr, ca_ip3r, effcai, rho
        return init_fn

    def get_debug_sim_fn(self): return _v5ip3r_debug

    def get_step_factory(self):
        def scan_factory(params, syn_params):
            gb, cicr = params
            c_pre_opt, c_post_opt, is_apical, cai_pre, t_pre, cai_post, t_post = syn_params
            c_pre = jnp.max(compute_effcai_piecewise_linear_jax(cai_pre, t_pre, tau_effca=cicr['tau_eff']))
            c_post = jnp.max(compute_effcai_piecewise_linear_jax(cai_post, t_post, tau_effca=cicr['tau_eff']))
            theta_d = jnp.where(is_apical, gb['a20']*c_pre+gb['a21']*c_post, gb['a00']*c_pre+gb['a01']*c_post)
            theta_p = jnp.maximum(jnp.where(is_apical, gb['a30']*c_pre+gb['a31']*c_post, gb['a10']*c_pre+gb['a11']*c_post), theta_d + 0.01)

            def scan_step(carry, inputs):
                IP3, P, ca_ryr, ca_ip3r, effcai, rho = carry
                cai_raw, dt, nves_i = inputs
                ca_e = jnp.maximum(0.0, cai_raw - CAI_REST)
                # IP3: bolus + decay
                IP3_post = IP3 + cicr['delta_IP3'] * nves_i
                IP3_new = jnp.maximum(0.0, IP3_post + dt * (-IP3_post / cicr['tau_IP3']))
                # gates
                ca_e_nT = ca_e ** cicr['n_T']; K_T_nT = cicr['K_T'] ** cicr['n_T']
                T = ca_e_nT / (K_T_nT + ca_e_nT + 1e-30)
                ca_e4 = ca_e**N_H; K_H4 = K_H**N_H
                S = ca_e4 / (K_H4 + ca_e4 + 1e-30)
                # fluxes
                J_ryr = cicr['V_RyR'] * P * T
                ipg = IP3_new / (K_IP3 + IP3_new + 1e-12); cag = ca_e / (K_ACT + ca_e + 1e-12)
                J_ip3r = cicr['V_IP3R'] * P * ipg * cag
                J_serca = cicr['phi_serca'] * S * (1.0 - P)
                J_leak = cicr['k_leak'] * (P - P_SS)
                P_new = jnp.clip(P + dt * (J_serca - J_ryr - J_ip3r - J_leak), 0.0, 1.0)
                # cytosolic Ca pools
                d_ryr = jnp.exp(-dt / cicr['tau_RyR']); ca_ryr_new = ca_ryr * d_ryr + J_ryr * cicr['tau_RyR'] * (1.0 - d_ryr)
                d_ip3r = jnp.exp(-dt / cicr['tau_IP3R']); ca_ip3r_new = ca_ip3r * d_ip3r + J_ip3r * cicr['tau_IP3R'] * (1.0 - d_ip3r)
                # effcai
                d_eff = jnp.exp(-dt / cicr['tau_eff'])
                effcai_new = effcai * d_eff + (ca_e + ca_ryr_new + ca_ip3r_new) * cicr['tau_eff'] * (1.0 - d_eff)
                # rho
                gp_eff = gb['gamma_p'] + cicr['g_RyR'] * ca_ryr_new + cicr['g_IP3R'] * ca_ip3r_new
                pot = jnp.where(effcai_new > theta_p, 1.0, 0.0); dep = jnp.where(effcai_new > theta_d, 1.0, 0.0)
                drho = (-rho*(1.0-rho)*(0.5-rho) + gp_eff*(1.0-rho)*pot - gb['gamma_d']*rho*dep) / 70000.0
                rho_new = jnp.where(dt > 0, jnp.clip(rho + dt * drho, 0.0, 1.0), rho)
                return (IP3_new, P_new, ca_ryr_new, ca_ip3r_new, effcai_new, rho_new), None
            return scan_step
        return scan_factory

if __name__ == "__main__":
    ERDynamicsCICRModel().run()
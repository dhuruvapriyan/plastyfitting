#!/usr/bin/env python3
"""
CICR Model — Unified Architecture (DAG + Priming, Fixed GB Thresholds)

State variables (carry): IP3, P, DAG, ca_cicr, effcai, rho

Equations:
  dIP3/dt    = delta_IP3 * nves - IP3 / tau_IP3
  J_serca    = phi_serca * S(ca_e) * (1 - P)
  J_cicr     = V_cicr * P * T(ca_e) * IP3/(K_IP3 + IP3)
  dP/dt      = J_serca - J_cicr - k_leak * P
  dDAG/dt    = k_dag * IP3 * S(ca_e) - DAG / tau_DAG
  dca_cicr   = J_cicr - ca_cicr / tau_cicr
  deffcai    = (ca_e + ca_cicr - effcai) / tau_eff

  gamma_p_eff = gamma_p + g_cicr * ca_cicr   (CICR amplifies potentiation)
  gamma_d_eff = gamma_d + g_dag * DAG         (DAG/eCB amplifies depression)
  pot = 1 if effcai > theta_p (FIXED)
  dep = 1 if effcai > theta_d (FIXED)
  drho/dt = [-rho(1-rho)(0.5-rho) + gamma_p_eff*(1-rho)*pot
             - gamma_d_eff*rho*dep] / 70000

P = ER store content (priming). Starts at 0, filled by SERCA.
DAG = PLC coincidence detector (requires both Ca2+ and IP3/mGluR).
  DAG → eCB → amplified depression (Nevian pathway).
Thresholds theta_d, theta_p are FIXED (Graupner-Brunel).
"""

import numpy as np
import jax.numpy as jnp
from cicr_common import CICRModel, compute_effcai_piecewise_linear_jax

# Fixed biophysical constants
N_H = 4; K_H = 0.0004; K_IP3 = 1.0; CAI_REST = 70e-6
TRACE_TAU_REF = 200.0

def _hill(x, K, n):
    xn = x**n; return xn / (K**n + xn + 1e-30)

def _debug_sim(cai_1syn, t, cp, cq, is_apical, dp):
    """Numpy debug simulation."""
    delta_IP3 = float(dp["delta_IP3"]); tau_IP3 = float(dp["tau_IP3"])
    phi_serca = float(dp["phi_serca"]); k_leak = float(dp["k_leak"])
    V_cicr = float(dp["V_cicr"]); K_T = float(dp["K_T"]); n_T = float(dp["n_T"])
    k_dag = float(dp["k_dag"]); tau_DAG = float(dp["tau_DAG"]); g_dag = float(dp["g_dag"])
    g_cicr = float(dp["g_cicr"]); tau_cicr = float(dp["tau_cicr"]); tau_eff = float(dp["tau_eff"])
    gamma_d = float(dp.get("gamma_d_GB_GluSynapse", 150.0))
    gamma_p = float(dp.get("gamma_p_GB_GluSynapse", 150.0))

    scale = tau_eff / TRACE_TAU_REF; cp_s, cq_s = cp * scale, cq * scale
    if is_apical:
        theta_d = dp["a20"]*cp_s + dp["a21"]*cq_s
        theta_p = dp["a30"]*cp_s + dp["a31"]*cq_s
    else:
        theta_d = dp["a00"]*cp_s + dp["a01"]*cq_s
        theta_p = dp["a10"]*cp_s + dp["a11"]*cq_s
    theta_p = max(theta_p, theta_d + 0.01)

    _nves_raw = dp.get("_nves", None)
    if _nves_raw is None or np.ndim(_nves_raw) == 0:
        nves = np.zeros(len(t))
    else:
        nves = np.asarray(_nves_raw, dtype=np.float64)

    n = len(cai_1syn); cai_total = np.copy(cai_1syn)
    ip3_out, p_out, dag_out = np.zeros(n), np.zeros(n), np.zeros(n)
    ca_cicr_out, effcai_out, rho_out = np.zeros(n), np.zeros(n), np.zeros(n)
    IP3, P, DAG, ca_cicr, effcai = 0.0, 0.0, 0.0, 0.0, 0.0
    rho = float(dp.get("rho0", 0.5))
    rho_out[0] = rho

    for i in range(n - 1):
        dt = t[i+1] - t[i]
        if dt <= 0:
            ip3_out[i+1]=IP3; p_out[i+1]=P; dag_out[i+1]=DAG
            ca_cicr_out[i+1]=ca_cicr; effcai_out[i+1]=effcai; rho_out[i+1]=rho
            cai_total[i+1]=cai_1syn[i+1]+ca_cicr; continue
        ca_e = max(0.0, cai_1syn[i] - CAI_REST)
        # IP3
        IP3 = IP3 + delta_IP3 * nves[i]
        IP3 = max(0.0, IP3 + dt * (-IP3 / tau_IP3))
        # Gates
        T = _hill(ca_e, K_T, n_T)
        S = _hill(ca_e, K_H, N_H)
        ipg = IP3 / (K_IP3 + IP3 + 1e-12)
        # Fluxes
        J_serca = phi_serca * S * (1.0 - P)
        J_cicr_val = V_cicr * P * T * ipg
        J_leak = k_leak * P
        # P update
        P = min(1.0, max(0.0, P + dt * (J_serca - J_cicr_val - J_leak)))
        # DAG update (PLC = IP3 * S(ca_e), coincidence detector)
        dDAG = k_dag * IP3 * S - DAG / tau_DAG
        DAG = max(0.0, DAG + dt * dDAG)
        # ca_cicr
        d_c = np.exp(-dt / tau_cicr)
        ca_cicr = ca_cicr * d_c + J_cicr_val * tau_cicr * (1.0 - d_c)
        # effcai
        d_e = np.exp(-dt / tau_eff)
        effcai = effcai * d_e + (ca_e + ca_cicr) * tau_eff * (1.0 - d_e)
        # Plasticity: CICR amplifies potentiation, DAG amplifies depression
        gp_eff = gamma_p + g_cicr * ca_cicr
        gd_eff = gamma_d + g_dag * DAG
        pot = 1.0 if effcai > theta_p else 0.0
        dep = 1.0 if effcai > theta_d else 0.0
        drho = (-rho*(1-rho)*(0.5-rho) + gp_eff*(1-rho)*pot - gd_eff*rho*dep) / 70000.0
        rho = min(1.0, max(0.0, rho + dt * drho))
        cai_total[i+1]=cai_1syn[i+1]+ca_cicr; ip3_out[i+1]=IP3; p_out[i+1]=P
        dag_out[i+1]=DAG; ca_cicr_out[i+1]=ca_cicr; effcai_out[i+1]=effcai; rho_out[i+1]=rho

    return {"cai_total": cai_total, "priming": ip3_out, "ca_er": p_out,
            "ca_cicr": ca_cicr_out, "ca_ryr": dag_out, "ca_ip3r": ca_cicr_out,
            "effcai": effcai_out, "rho": rho_out}


class LiRinzelCICRModel(CICRModel):
    DESCRIPTION = "Biophysical IP3R/CICR Model (Li-Rinzel, ER Depletion, Regenerative)"
    FIT_PARAMS = [
        # Plasticity Parameters
        ("gamma_d_GB_GluSynapse", 50.0, 250.0), ("gamma_p_GB_GluSynapse", 50.0, 300.0),
        ("a00", 1.0, 5.0), ("a01", 1.0, 5.0), ("a10", 1.0, 5.0), ("a11", 1.0, 5.0),
        ("a20", 1.0, 5.0), ("a21", 1.0, 5.0), ("a30", 1.0, 10.0), ("a31", 1.0, 5.0),
        
        # IP3 Production
        ("delta_IP3", 0.01, 10.0), ("tau_IP3", 100.0, 2000.0),
        
        # Li-Rinzel Gating Parameters (uM)
        ("d1", 0.01, 1.0), ("d2", 0.5, 5.0), ("d3", 0.05, 1.5), ("d5", 0.05, 1.0),
        ("a2", 0.05, 2.0), # Inactivation rate constant
        
        # Flux Parameters (uM/ms)
        ("V_CICR", 0.01, 10.0), 
        ("V_SERCA", 0.01, 5.0), ("K_SERCA", 0.05, 1.0),
        ("V_leak", 1e-5, 0.01),
        
        # Compartment/Scaling
        ("gamma_ER", 0.05, 0.25), # Volume ratio ER / Cytosol
        ("tau_extrusion", 10.0, 300.0), # Cytosolic clearance
        ("tau_eff", 50.0, 500.0)
    ]
#     ======================================================================
#   Method: cmaes
#   Loss:   0.0004527014
# ======================================================================

#   Best parameters:
#     gamma_d_GB_GluSynapse          =   249.9229  (default: 67.1800, +272.0%)
#     gamma_p_GB_GluSynapse          =   191.8133  (default: 292.5200, -34.4%)
#     a00                            =     2.8440  (default: 1.1800, +141.0%)
#     a01                            =     1.2974  (default: 1.2400, +4.6%)
#     a10                            =     4.9952  (default: 4.3000, +16.2%)
#     a11                            =     3.5837  (default: 4.9400, -27.5%)
#     a20                            =     1.0138  (default: 1.0500, -3.4%)
#     a21                            =     1.0256  (default: 1.0500, -2.3%)
#     a30                            =     1.0804  (default: 9.9900, -89.2%)
#     a31                            =     3.3058  (default: 5.0000, -33.9%)
#     delta_IP3                      =     7.1016  (default: 0.5000, +1320.3%)
#     tau_IP3                        =   553.6846  (default: 500.0000, +10.7%)
#     d1                             =     0.7115  (default: 0.1300, +447.3%)
#     d2                             =     0.5164  (default: 1.0490, -50.8%)
#     d3                             =     0.6019  (default: 0.9434, -36.2%)
#     d5                             =     0.0533  (default: 0.0823, -35.3%)
#     a2                             =     1.8171  (default: 0.2000, +808.6%)
#     V_CICR                         =     6.7840  (default: 0.5000, +1256.8%)
#     V_SERCA                        =     2.3493  (default: 0.9000, +161.0%)
#     K_SERCA                        =     0.8934  (default: 0.1000, +793.4%)
#     V_leak                         =     0.0016  (default: 0.0002, +685.0%)
#     gamma_ER                       =     0.1255  (default: 0.1000, +25.5%)
#     tau_extrusion                  =   163.7724  (default: 50.0000, +227.5%)
#     tau_eff                        =   129.1823  (default: 50.0000, +158.4%)

#   Protocol         Predicted  Experiment      Error
#   -------------------------------------------------
#   10Hz_10ms           1.2159      1.2013    +0.0146
#   10Hz_-10ms          0.8073      0.7922    +0.0151
# ======================================================================

# 2026-03-01 12:05:10,625 - INFO - Generating diagnostic plot → cicr_diagnostic_biophysical_ip3r/cicr_model_cmaes_20260301_114212_ref.png
# 2026-03-01 12:05:12,565 - INFO -   Computing 10Hz_10ms (single synapse 3 for plot)…
# Traceback (most recent call last):
    DEFAULT_PARAMS = {
        "gamma_d_GB_GluSynapse": 249.9229, "gamma_p_GB_GluSynapse": 191.8133,
        "a00": 2.8440, "a01": 1.2974, "a10": 4.9952, "a11": 3.5837,
        "a20": 1.0138, "a21": 1.0256, "a30": 1.0804, "a31": 3.3058,
        "delta_IP3": 7.1016, "tau_IP3": 553.6846,
        "d1": 0.7115, "d2": 0.5164, "d3": 0.6019, "d5": 0.0533, "a2": 1.8171,
        "V_CICR": 6.7840, "V_SERCA": 2.3493, "K_SERCA": 0.8934, "V_leak": 0.0016,
        "gamma_ER": 0.1255, "tau_extrusion": 163.7724, "tau_eff": 129.1823
    }

    def unpack_params(self, x):
        gb = {'gamma_d': x[0], 'gamma_p': x[1],
              'a00': x[2], 'a01': x[3], 'a10': x[4], 'a11': x[5],
              'a20': x[6], 'a21': x[7], 'a30': x[8], 'a31': x[9]}
        cicr = {'delta_IP3': x[10], 'tau_IP3': x[11],
                'd1': x[12], 'd2': x[13], 'd3': x[14], 'd5': x[15], 'a2': x[16],
                'V_CICR': x[17], 'V_SERCA': x[18], 'K_SERCA': x[19], 'V_leak': x[20],
                'gamma_ER': x[21], 'tau_extrusion': x[22], 'tau_eff': x[23]}
        return gb, cicr

    def get_init_fn(self):
        def init_fn(cai_first, rho0):
            # IP3, Ca_ER, ca_cicr, h, effcai, rho
            # ER starts mostly full (e.g. 100 uM), h starts mostly open (~0.8)
            return (0.0, 100.0, 0.0, 0.8, 0.0, rho0)
        return init_fn

    def get_debug_sim_fn(self):
        # We can implement a numpy version of the Li-Rinzel scan step here if needed later
        return _debug_sim

    def get_step_factory(self):
        def scan_factory(params, syn_params):
            gb, cicr = params
            c_pre_opt, c_post_opt, is_apical, cai_pre, t_pre, cai_post, t_post = syn_params
            c_pre = jnp.max(compute_effcai_piecewise_linear_jax(cai_pre, t_pre, tau_effca=cicr['tau_eff']))
            c_post = jnp.max(compute_effcai_piecewise_linear_jax(cai_post, t_post, tau_effca=cicr['tau_eff']))
            
            theta_d = jnp.where(is_apical, gb['a20']*c_pre + gb['a21']*c_post, gb['a00']*c_pre + gb['a01']*c_post)
            theta_p = jnp.maximum(
                jnp.where(is_apical, gb['a30']*c_pre + gb['a31']*c_post, gb['a10']*c_pre + gb['a11']*c_post),
                theta_d + 0.01)

            def scan_step(carry, inputs):
                IP3, Ca_ER, ca_cicr, h, effcai, rho = carry
                cai_raw, dt, nves_i = inputs
                
                # External Calcium (from VDCC/NMDA)
                ca_ext = jnp.maximum(0.0, cai_raw - CAI_REST)
                
                # TOTAL Cytosolic Calcium (The feedback loop!)
                ca_i = ca_ext + ca_cicr
                
                # 1. IP3 Dynamics (Coincidence detection / production)
                IP3_post = IP3 + cicr['delta_IP3'] * nves_i
                IP3_new = jnp.maximum(0.0, IP3_post + dt * (-IP3_post / cicr['tau_IP3']))
                
                # 2. Li-Rinzel IP3R Gating
                # Activation by IP3
                m_inf = IP3_new / (IP3_new + cicr['d1'] + 1e-12)
                # Activation by Cytosolic Calcium (Regenerative!)
                n_inf = ca_i / (ca_i + cicr['d5'] + 1e-12)
                
                P_open = (m_inf ** 3) * (n_inf ** 3) * (h ** 3)
                
                # Inactivation gate 'h'
                Q2 = cicr['d2'] * (IP3_new + cicr['d1']) / (IP3_new + cicr['d3'] + 1e-12)
                h_inf = Q2 / (Q2 + ca_i + 1e-12)
                tau_h = 1.0 / (cicr['a2'] * (Q2 + ca_i + 1e-12))
                
                # Update h
                eh = jnp.exp(-dt / tau_h)
                h_new = h * eh + h_inf * (1.0 - eh)
                
                # 3. Calcium Fluxes (ER to Cytosol and back)
                gradient = jnp.maximum(0.0, Ca_ER - ca_i) # Ca flows out of ER
                J_cicr = cicr['V_CICR'] * P_open * gradient
                J_leak = cicr['V_leak'] * gradient
                J_serca = cicr['V_SERCA'] * (ca_i**2) / (cicr['K_SERCA']**2 + ca_i**2 + 1e-12)
                
                # 4. Integrate ER and Cytosolic CICR pools
                dCa_ER = (J_serca - J_cicr - J_leak) / cicr['gamma_ER']
                Ca_ER_new = jnp.maximum(0.0, Ca_ER + dt * dCa_ER)
                
                dca_cicr = J_cicr + J_leak - J_serca - (ca_cicr / cicr['tau_extrusion'])
                ca_cicr_new = jnp.maximum(0.0, ca_cicr + dt * dca_cicr)
                
                # 5. Effcai & Plasticity 
                # (We keep standard STDP behavior reading from total effective calcium)
                d_eff = jnp.exp(-dt / cicr['tau_eff'])
                effcai_new = effcai * d_eff + (ca_ext + ca_cicr_new) * cicr['tau_eff'] * (1.0 - d_eff)
                
                pot = jnp.where(effcai_new > theta_p, 1.0, 0.0)
                dep = jnp.where(effcai_new > theta_d, 1.0, 0.0)
                drho = (-rho*(1.0-rho)*(0.5-rho) + gb['gamma_p']*(1.0-rho)*pot - gb['gamma_d']*rho*dep) / 70000.0
                rho_new = jnp.where(dt > 0, jnp.clip(rho + dt * drho, 0.0, 1.0), rho)
                
                return (IP3_new, Ca_ER_new, ca_cicr_new, h_new, effcai_new, rho_new), None
            return scan_step
        return scan_factory

if __name__ == "__main__":
    LiRinzelCICRModel().run()

#!/usr/bin/env python3
"""
CICR Model — Li-Rinzel IP3R with theta_p-Gated RyR Release (ReLU)

The RyR/CICR release is gated by a ReLU on (effcai - theta_p): zero below
theta_p, linearly proportional to excess above. This ties ER release to the
CaMKII activation regime and provides graded amplification — more calcium
above the potentiation threshold means more CICR.

State variables (carry): IP3, Ca_ER, ca_cicr, h, effcai, rho

Key equation:
  ryr_gate = max(0, effcai - theta_p)
  J_cicr   = V_CICR * P_open * gradient * ryr_gate

V_CICR absorbs the gain (no separate steepness parameter needed).

Fixed biophysical constants (from GluSynapse_CICR.mod CMA-ES fit + literature):
  Li-Rinzel gating:  d1=0.7115, d2=0.5164, d3=0.6019, d5=0.0533, a2=1.8171
  Compartment:       gamma_ER=0.1255, K_SERCA=0.8934, V_leak=0.0016

Free parameters: 16 (10 plasticity + 6 CICR/scaling)
"""

import numpy as np
import jax
import jax.numpy as jnp
from cicr_common import CICRModel, compute_effcai_piecewise_linear_jax

# Fixed biophysical constants
CAI_REST = 70e-6
TRACE_TAU_REF = 200.0

# Li-Rinzel IP3R gating constants (from GluSynapse_CICR.mod fitted values)
D1 = 0.7115      # IP3 dissociation constant (IP3 activation)
D2 = 0.5164      # Ca inactivation dissociation constant
D3 = 0.6019      # IP3 offset in h gate expression
D5 = 0.0533      # Ca activation dissociation constant
A2 = 1.8171      # inactivation rate constant (/mM/ms)

# Compartment / flux constants
GAMMA_ER = 0.1255   # ER/cytosol volume ratio (morphological)
K_SERCA = 0.8934    # SERCA half-activation (mM) — Lytton et al 1992
V_LEAK = 0.0016     # passive ER leak conductance (/ms)


def _debug_sim(cai_1syn, t, cp, cq, is_apical, dp):
    """Numpy debug simulation — Li-Rinzel IP3R with theta_p ReLU RyR gate."""
    delta_IP3 = float(dp["delta_IP3"]); tau_IP3 = float(dp["tau_IP3"])
    V_CICR = float(dp["V_CICR"]); V_SERCA = float(dp["V_SERCA"])
    tau_ext = float(dp["tau_extrusion"]); tau_eff = float(dp["tau_eff"])
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
    ip3_out = np.zeros(n); ca_er_out = np.zeros(n)
    ca_cicr_out = np.zeros(n); h_out = np.zeros(n)
    effcai_out = np.zeros(n); rho_out = np.zeros(n)

    # Initial state
    IP3 = 0.0; Ca_ER = 100.0; ca_cicr = 0.0; h = 0.8; effcai = 0.0
    rho = float(dp.get("rho0", 0.5))
    rho_out[0] = rho; h_out[0] = h; ca_er_out[0] = Ca_ER

    for i in range(n - 1):
        dt = t[i+1] - t[i]
        if dt <= 0:
            ip3_out[i+1] = IP3; ca_er_out[i+1] = Ca_ER
            ca_cicr_out[i+1] = ca_cicr; h_out[i+1] = h
            effcai_out[i+1] = effcai; rho_out[i+1] = rho
            cai_total[i+1] = cai_1syn[i+1] + ca_cicr
            continue

        ca_ext = max(0.0, cai_1syn[i] - CAI_REST)
        ca_i = ca_ext + ca_cicr

        # 1. IP3
        IP3 = IP3 + delta_IP3 * nves[i]
        IP3 = max(0.0, IP3 + dt * (-IP3 / tau_IP3))

        # 2. Li-Rinzel IP3R gating (fixed constants)
        m_inf = IP3 / (IP3 + D1 + 1e-12)
        n_inf = ca_i / (ca_i + D5 + 1e-12)
        P_open = (m_inf ** 3) * (n_inf ** 3) * (h ** 3)

        # h gate
        Q2 = D2 * (IP3 + D1) / (IP3 + D3 + 1e-12)
        h_inf = Q2 / (Q2 + ca_i + 1e-12)
        tau_h = 1.0 / (A2 * (Q2 + ca_i + 1e-12))
        eh = np.exp(-dt / tau_h)
        h = h * eh + h_inf * (1.0 - eh)

        # 3. RyR gate: ReLU on (effcai - theta_p)
        ryr_gate = max(0.0, effcai - theta_p)

        # 4. Fluxes (fixed K_SERCA, V_LEAK, GAMMA_ER)
        gradient = max(0.0, Ca_ER - ca_i)
        J_cicr = V_CICR * P_open * gradient * ryr_gate
        J_leak = V_LEAK * gradient
        J_serca = V_SERCA * (ca_i**2) / (K_SERCA**2 + ca_i**2 + 1e-12)

        # 5. ER and cytosolic pools
        dCa_ER = (J_serca - J_cicr - J_leak) / GAMMA_ER
        Ca_ER = max(0.0, Ca_ER + dt * dCa_ER)

        dca_cicr = J_cicr + J_leak - J_serca - (ca_cicr / tau_ext)
        ca_cicr = max(0.0, ca_cicr + dt * dca_cicr)

        # 6. effcai
        d_e = np.exp(-dt / tau_eff)
        effcai = effcai * d_e + (ca_ext + ca_cicr) * tau_eff * (1.0 - d_e)

        # 7. Plasticity
        pot = 1.0 if effcai > theta_p else 0.0
        dep = 1.0 if effcai > theta_d else 0.0
        drho = (-rho*(1-rho)*(0.5-rho) + gamma_p*(1-rho)*pot - gamma_d*rho*dep) / 70000.0
        rho = min(1.0, max(0.0, rho + dt * drho))

        cai_total[i+1] = cai_1syn[i+1] + ca_cicr
        ip3_out[i+1] = IP3; ca_er_out[i+1] = Ca_ER
        ca_cicr_out[i+1] = ca_cicr; h_out[i+1] = h
        effcai_out[i+1] = effcai; rho_out[i+1] = rho

    return {"cai_total": cai_total, "priming": ip3_out, "ca_er": ca_er_out,
            "ca_cicr": ca_cicr_out, "ca_ryr": h_out, "ca_ip3r": ca_cicr_out,
            "effcai": effcai_out, "rho": rho_out}


class LiRinzelCICRModel(CICRModel):
    DESCRIPTION = "Li-Rinzel IP3R + theta_p ReLU RyR Gate"
    FIT_PARAMS = [
        # --- Plasticity Parameters (10) ---
        ("gamma_d_GB_GluSynapse", 50.0, 250.0), ("gamma_p_GB_GluSynapse", 50.0, 300.0),
        ("a00", 1.0, 5.0), ("a01", 1.0, 5.0), ("a10", 1.0, 5.0), ("a11", 1.0, 5.0),
        ("a20", 1.0, 5.0), ("a21", 1.0, 5.0), ("a30", 1.0, 10.0), ("a31", 1.0, 5.0),

        # --- CICR Scaling Parameters (6) ---
        # IP3 production (bounds tightened from MOD file fit)
        ("delta_IP3", 1.0, 10.0), ("tau_IP3", 200.0, 1000.0),
        # Flux magnitudes (V_CICR absorbs ReLU gain)
        ("V_CICR", 1.0, 10000.0), ("V_SERCA", 0.5, 5.0),
        # Cytosolic dynamics
        ("tau_extrusion", 50.0, 300.0),
        ("tau_eff", 50.0, 500.0),
    ]
    DEFAULT_PARAMS = {
        # Plasticity (from MOD file fitted values)
        "gamma_d_GB_GluSynapse": 249.92, "gamma_p_GB_GluSynapse": 191.81,
        "a00": 1.18, "a01": 1.24, "a10": 4.30, "a11": 4.94,
        "a20": 1.05, "a21": 1.05, "a30": 9.99, "a31": 5.00,
        # CICR scaling (from MOD file fitted values)
        "delta_IP3": 7.10, "tau_IP3": 553.68,
        "V_CICR": 500.0, "V_SERCA": 2.35,
        "tau_extrusion": 163.77, "tau_eff": 129.18,
    }

    def unpack_params(self, x):
        gb = {'gamma_d': x[0], 'gamma_p': x[1],
              'a00': x[2], 'a01': x[3], 'a10': x[4], 'a11': x[5],
              'a20': x[6], 'a21': x[7], 'a30': x[8], 'a31': x[9]}
        cicr = {'delta_IP3': x[10], 'tau_IP3': x[11],
                'V_CICR': x[12], 'V_SERCA': x[13],
                'tau_extrusion': x[14], 'tau_eff': x[15]}
        return gb, cicr

    def get_init_fn(self):
        def init_fn(cai_first, rho0):
            # IP3, Ca_ER, ca_cicr, h, effcai, rho
            return (0.0, 100.0, 0.0, 0.8, 0.0, rho0)
        return init_fn

    def get_debug_sim_fn(self):
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

                # Total Cytosolic Calcium (feedback loop)
                ca_i = ca_ext + ca_cicr

                # 1. IP3 Dynamics
                IP3_post = IP3 + cicr['delta_IP3'] * nves_i
                IP3_new = jnp.maximum(0.0, IP3_post + dt * (-IP3_post / cicr['tau_IP3']))

                # 2. Li-Rinzel IP3R Gating (fixed biophysical constants)
                m_inf = IP3_new / (IP3_new + D1 + 1e-12)
                n_inf = ca_i / (ca_i + D5 + 1e-12)
                P_open = (m_inf ** 3) * (n_inf ** 3) * (h ** 3)

                # Inactivation gate h
                Q2 = D2 * (IP3_new + D1) / (IP3_new + D3 + 1e-12)
                h_inf = Q2 / (Q2 + ca_i + 1e-12)
                tau_h = 1.0 / (A2 * (Q2 + ca_i + 1e-12))
                eh = jnp.exp(-dt / tau_h)
                h_new = h * eh + h_inf * (1.0 - eh)

                # 3. RyR gate: ReLU on (effcai - theta_p)
                #    Zero below theta_p, linearly proportional to excess above.
                #    V_CICR absorbs the gain — no separate steepness param needed.
                ryr_gate = jnp.maximum(0.0, effcai - theta_p)

                # 4. Calcium Fluxes (fixed K_SERCA, V_LEAK, GAMMA_ER)
                gradient = jnp.maximum(0.0, Ca_ER - ca_i)
                J_cicr = cicr['V_CICR'] * P_open * gradient * ryr_gate
                J_leak = V_LEAK * gradient
                J_serca = cicr['V_SERCA'] * (ca_i**2) / (K_SERCA**2 + ca_i**2 + 1e-12)

                # 5. Integrate ER and Cytosolic CICR pools
                dCa_ER = (J_serca - J_cicr - J_leak) / GAMMA_ER
                Ca_ER_new = jnp.maximum(0.0, Ca_ER + dt * dCa_ER)

                dca_cicr = J_cicr + J_leak - J_serca - (ca_cicr / cicr['tau_extrusion'])
                ca_cicr_new = jnp.maximum(0.0, ca_cicr + dt * dca_cicr)

                # 6. Effcai & Plasticity
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

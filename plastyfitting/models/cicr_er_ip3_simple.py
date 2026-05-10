#!/usr/bin/env python3
"""
CICR Model — Dual Modulation (LTP boost + LTD suppression)

Key architecture: CICR provides TWO effects on plasticity:
  1. Direct potentiation drive (no threshold required)
  2. Suppression of depression rate

Modified Graupner-Brunel plasticity:
  cicr_drive = alpha_cicr * ca_cicr

  drho = (-rho*(1-rho)*(0.5-rho)
          + gamma_p*(1-rho)*(pot + cicr_drive)     ← CICR adds direct LTP
          - gamma_d*rho*dep / (1 + cicr_drive)     ← CICR suppresses LTD
         ) / tau_rho

STDP mechanism:
  Pre-before-post: high CICR → strong pot drive + suppressed dep → LTP
  Post-before-pre: low CICR → weak pot drive + full dep → LTD

CICR dynamics: triple coincidence (IP3 × Ca × Priming)

State variables (carry): IP3, P, ca_cicr, effcai, rho
Free parameters: 17 (10 plasticity + 7 CICR), tau_eff fixed at 278.318
"""

import numpy as np
import jax
import jax.numpy as jnp
from ..cicr_common import CICRModel

CAI_REST = 70e-6
K_IP3 = 0.001


def _debug_sim(cai_1syn, t, cp, cq, is_apical, dp):
    """Numpy debug simulation — Dual modulation CICR model."""
    delta_IP3 = float(dp["delta_IP3"]); tau_IP3 = float(dp["tau_IP3"])
    V_CICR = float(dp["V_CICR"]); K_ca = float(dp["K_ca"])
    alpha_cicr = float(dp["alpha_cicr"])
    tau_charge = float(dp["tau_charge"])
    tau_ext = float(dp["tau_extrusion"]); tau_eff = float(dp.get("tau_eff", 278.318))
    gamma_d = float(dp.get("gamma_d_GB_GluSynapse", 150.0))
    gamma_p = float(dp.get("gamma_p_GB_GluSynapse", 150.0))

    cp_s, cq_s = cp, cq
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
    ip3_out = np.zeros(n); P_out = np.zeros(n)
    ca_cicr_out = np.zeros(n)
    effcai_out = np.zeros(n); rho_out = np.zeros(n)

    IP3 = 0.0; P = 0.0; ca_cicr = 0.0; effcai = 0.0
    rho = float(dp.get("rho0", 0.5))
    rho_out[0] = rho
    K_ca2 = K_ca ** 2

    for i in range(n - 1):
        dt = t[i+1] - t[i]
        if dt <= 0:
            ip3_out[i+1] = IP3; P_out[i+1] = P
            ca_cicr_out[i+1] = ca_cicr
            effcai_out[i+1] = effcai; rho_out[i+1] = rho
            cai_total[i+1] = cai_1syn[i+1] + ca_cicr
            continue

        ca_ext = max(0.0, cai_1syn[i] - CAI_REST)

        # 1. IP3 dynamics
        IP3 = IP3 + delta_IP3 * nves[i]
        IP3 = max(0.0, IP3 * np.exp(-dt / tau_IP3))

        # 2. IP3R activation
        ip3_act = IP3 / (IP3 + K_IP3 + 1e-30)

        # 3. Calcium coincidence gate (Hill n=2)
        ca_act = ca_ext**2 / (K_ca2 + ca_ext**2 + 1e-30)

        # 4. Release (triple coincidence)
        J_release = V_CICR * ip3_act * ca_act * P

        # 5. Priming
        dP = ca_ext * (1 - P) / tau_charge - J_release
        P = min(1.0, max(0.0, P + dt * dP))

        # 6. Cytosolic CICR pool
        dca_cicr = J_release - ca_cicr / tau_ext
        ca_cicr = max(0.0, ca_cicr + dt * dca_cicr)

        # 7. Effective calcium integrator (baseline only)
        d_e = np.exp(-dt / tau_eff)
        effcai = effcai * d_e + ca_ext * tau_eff * (1.0 - d_e)

        # 8. DUAL MODULATION plasticity:
        pot = 1.0 if effcai > theta_p else 0.0
        dep = 1.0 if effcai > theta_d else 0.0
        cicr_drive = alpha_cicr * ca_cicr
        drho = (-rho*(1-rho)*(0.5-rho)
                + gamma_p*(1-rho)*(pot + cicr_drive)
                - gamma_d*rho*dep / (1.0 + cicr_drive)
               ) / 70000.0
        rho = min(1.0, max(0.0, rho + dt * drho))

        cai_total[i+1] = cai_1syn[i+1] + ca_cicr
        ip3_out[i+1] = IP3; P_out[i+1] = P
        ca_cicr_out[i+1] = ca_cicr
        effcai_out[i+1] = effcai; rho_out[i+1] = rho

    return {"cai_total": cai_total, "priming": ip3_out, "P": P_out,
            "ca_cicr": ca_cicr_out,
            "effcai": effcai_out, "rho": rho_out}


class SimpleCICRModel(CICRModel):
    DESCRIPTION = "Dual Modulation CICR (LTP boost + LTD suppress)"
    NEEDS_THRESHOLD_TRACES = False  # c_pre/c_post are baked scalars; no trace arrays needed
    FIT_PARAMS = [
        # --- Plasticity Parameters (10) ---
        ("gamma_d_GB_GluSynapse", 50.0, 250.0), ("gamma_p_GB_GluSynapse", 50.0, 300.0),
        ("a00", 1.0, 5.0), ("a01", 1.0, 5.0), ("a10", 1.0, 5.0), ("a11", 1.0, 5.0),
        ("a20", 1.0, 5.0), ("a21", 1.0, 5.0), ("a30", 1.0, 10.0), ("a31", 1.0, 5.0),

        # --- CICR Parameters (7) ---
        ("delta_IP3", 0.0005, 0.05),
        ("tau_IP3", 200.0, 1000.0),
        ("V_CICR", 1e-5, 0.1),
        ("K_ca", 0.0005, 0.5),
        ("alpha_cicr", 10.0, 50000.0),      # CICR → plasticity scaling
        ("tau_charge", 1.0, 100.0),
        ("tau_extrusion", 10.0, 100.0),
    ]
    DEFAULT_PARAMS = {
        "gamma_d_GB_GluSynapse": 244.27, "gamma_p_GB_GluSynapse": 201.89,
        "a00": 1.38, "a01": 2.65, "a10": 4.07, "a11": 4.20,
        "a20": 3.97, "a21": 1.47, "a30": 3.01, "a31": 2.44,
        "delta_IP3": 0.005, "tau_IP3": 330.0,
        "V_CICR": 0.01, "K_ca": 0.003,
        "alpha_cicr": 500.0,
        "tau_charge": 50.0, "tau_extrusion": 23.5,
    }

    def unpack_params(self, x):
        gb = {'gamma_d': x[0], 'gamma_p': x[1],
              'a00': x[2], 'a01': x[3], 'a10': x[4], 'a11': x[5],
              'a20': x[6], 'a21': x[7], 'a30': x[8], 'a31': x[9]}
        cicr = {'delta_IP3': x[10], 'tau_IP3': x[11],
                'V_CICR': x[12], 'K_ca': x[13], 'alpha_cicr': x[14],
                'tau_charge': x[15],
                'tau_extrusion': x[16],
                'tau_eff': 278.318}  # Fixed — use c_pre/c_post from pkl directly
        return gb, cicr

    def get_init_fn(self):
        def init_fn(cai_first, rho0):
            return (0.0, 0.0, 0.0, 0.0, rho0)
        return init_fn

    def get_debug_sim_fn(self):
        return _debug_sim

    def get_step_factory(self):
        def scan_factory(params, syn_params):
            gb, cicr = params
            c_pre, c_post, is_apical, cai_pre, t_pre, cai_post, t_post = syn_params

            # c_pre, c_post come from pkl (precomputed at tau_eff=278.318)
            theta_d = jnp.where(is_apical, gb['a20']*c_pre + gb['a21']*c_post, gb['a00']*c_pre + gb['a01']*c_post)
            theta_p = jnp.where(is_apical, gb['a30']*c_pre + gb['a31']*c_post, gb['a10']*c_pre + gb['a11']*c_post)

            K_ca2 = cicr['K_ca'] ** 2

            def scan_step(carry, inputs):
                IP3, P, ca_cicr, effcai, rho = carry
                cai_raw, dt, nves_i = inputs

                ca_ext = jnp.maximum(0.0, cai_raw - CAI_REST)

                # 1. IP3 dynamics
                IP3_post = IP3 + cicr['delta_IP3'] * nves_i
                IP3_new = jnp.maximum(0.0, IP3_post * jnp.exp(-dt / cicr['tau_IP3']))

                # 2. IP3R activation
                ip3_act = IP3_new / (IP3_new + K_IP3 + 1e-30)

                # 3. Calcium coincidence gate
                ca_act = ca_ext**2 / (K_ca2 + ca_ext**2 + 1e-30)

                # 4. Release (triple coincidence)
                J_release = cicr['V_CICR'] * ip3_act * ca_act * P

                # 5. Priming
                dP = ca_ext * (1.0 - P) / cicr['tau_charge'] - J_release
                P_new = jnp.where(dt > 0, jnp.clip(P + dt * dP, 0.0, 1.0), P)

                # 6. Cytosolic CICR pool
                dca_cicr = J_release - ca_cicr / cicr['tau_extrusion']
                ca_cicr_new = jnp.maximum(0.0, ca_cicr + dt * dca_cicr)

                # 7. Effective calcium integrator (baseline only)
                d_eff = jnp.exp(-dt / cicr['tau_eff'])
                effcai_new = effcai * d_eff + ca_ext * cicr['tau_eff'] * (1.0 - d_eff)

                # 8. DUAL MODULATION plasticity:
                #    CICR adds direct potentiation + suppresses depression
                pot = jnp.where(effcai_new > theta_p, 1.0, 0.0)
                dep = jnp.where(effcai_new > theta_d, 1.0, 0.0)
                cicr_drive = cicr['alpha_cicr'] * ca_cicr_new
                drho = (-rho*(1.0-rho)*(0.5-rho)
                        + gb['gamma_p']*(1.0-rho)*(pot + cicr_drive)
                        - gb['gamma_d']*rho*dep
                       ) / 70000.0
                rho_new = jnp.where(dt > 0, jnp.clip(rho + dt * drho, 0.0, 1.0), rho)

                return (IP3_new, P_new, ca_cicr_new, effcai_new, rho_new), None
            return scan_step
        return scan_factory

if __name__ == "__main__":
    SimpleCICRModel().run()

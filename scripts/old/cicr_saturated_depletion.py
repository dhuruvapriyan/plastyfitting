#!/usr/bin/env python3
"""
Primed-Junctional CICR Model V4.0 — Multiplicative CICR-Potentiation

6 equations, 8 CICR parameters (18 total with Graupner-Brunel).

Architecture: CICR modulates potentiation RATE, not calcium LEVEL.

Why additive CICR → c* fails:
  c*_syn peaks at ~0.15 mM but max theta_p = a*c_pre + a*c_post ~ 0.002 mM.
  c*_syn >> theta_p for BOTH protocols → both Heavisides always ON →
  GB dynamics always favor potentiation (gamma_p*(1-rho) > gamma_d*rho).
  Adding CICR to c* only makes this worse.

Multiplicative solution:
  gamma_p_eff = gamma_p + g_cicr * Ca_CICR

  At -10ms: Ca_CICR ~ 0.01 (weak trigger) → gamma_p_eff low → LTD
  At +10ms: Ca_CICR ~ 0.33 (strong trigger) → gamma_p_eff high → LTP
  33x Ca_CICR discrimination cleanly separates the protocols.

  Biophysical basis: CaMKII (potentiation) requires high peak Ca2+ which
  ER release provides. Calcineurin (depression) responds to moderate
  sustained Ca2+ from NMDARs. Two pathways, two calcium sources.

Equations:
  1. Ca_e = max(0, Ca_i - Ca_rest)
  2. T = Ca_e^n_T / (K_T^n_T + Ca_e^n_T)
  3. dc*/dt = Ca_e - c*/tau_eff               [synaptic integrator only]
  4. dP/dt = k_fill*max(0,c*-theta_d)*(1-P)   [theta_d-gated filling]
           - P/tau_leak - V_CICR*T*P           [leak + depletion]
  5. d(Ca_CICR)/dt = V_CICR*P*T - Ca_CICR/tau_CICR  [CICR release]
  6. drho/dt ~ (gamma_p + g_cicr*Ca_CICR)*(1-rho)*H(c*-theta_p)
             - gamma_d*rho*H(c*-theta_d)      [both thresholds use c*]
"""

import numpy as np
import jax.numpy as jnp
from cicr_common import CICRModel, compute_effcai_piecewise_linear_jax


def _apply_primed_junctional_debug(cai_1syn, t, cp, cq, is_apical, dp):
    k_fill    = float(dp["k_fill"])
    tau_leak  = float(dp["tau_leak"])
    K_trig    = float(dp["K_trig"])
    n_trig    = float(dp["n_trig"])
    Vmax_CICR = float(dp["Vmax_CICR"])
    tau_CICR  = float(dp["tau_CICR"])
    g_cicr    = float(dp["g_cicr"])
    tau_eff   = float(dp["tau_eff"])
    cai_rest  = 70e-6

    theta_d = float(dp.get("theta_d", 0.0))
    theta_p = float(dp.get("theta_p", 0.0))
    gamma_d = float(dp.get("gamma_d", 77.0))
    gamma_p = float(dp.get("gamma_p", 236.0))

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

    for i in range(n - 1):
        dt = t[i + 1] - t[i]
        if dt <= 0.0:
            prime_out[i + 1]   = P
            ca_cicr_out[i + 1] = ca_cicr
            effcai_out[i + 1]  = effcai
            continue

        ca_e = max(0.0, cai_1syn[i] - cai_rest)

        T = (ca_e**n_trig) / (K_trig**n_trig + ca_e**n_trig + 1e-12)

        eff_decay = np.exp(-dt / tau_eff)
        effcai = effcai * eff_decay + ca_e * tau_eff * (1.0 - eff_decay)

        fill_drive = max(0.0, effcai - theta_d)
        A = k_fill * fill_drive
        B = A + (1.0 / tau_leak) + Vmax_CICR * T
        P_inf = A / (B + 1e-12)
        P = P_inf + (P - P_inf) * np.exp(-dt * B)

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
        "effcai":    effcai_out,
    }


class PrimedJunctionalCICRModel(CICRModel):
    DESCRIPTION = "Primed-Junctional CICR V4 (multiplicative gamma_p modulation)"

    FIT_PARAMS = [
        ("gamma_d", 50.0, 300.0),
        ("gamma_p", 50.0, 600.0),
        ("a00", 1.0, 5.0),
        ("a01", 1.0, 5.0),
        ("a10", 1.0, 5.0),
        ("a11", 1.0, 5.0),
        ("a20", 1.0, 5.0),
        ("a21", 1.0, 5.0),
        ("a30", 1.0, 5.0),
        ("a31", 1.0, 5.0),
        ("k_fill", 0.001, 10.0),
        ("tau_leak", 30000.0, 300000.0),
        ("K_trig", 0.001, 0.005),
        ("n_trig", 4.0, 15.0),
        ("Vmax_CICR", 0.0001, 0.1),
        ("tau_CICR", 10.0, 1000.0),
        ("g_cicr", 10.0, 10000.0),
        ("tau_eff", 50.0, 500.0),
    ]

    DEFAULT_PARAMS = {
        "gamma_d": 150.0,
        "gamma_p": 80.0,
        "a00": 1.003, "a01": 1.530,
        "a10": 1.424, "a11": 4.366,
        "a20": 2.160, "a21": 1.004,
        "a30": 4.738, "a31": 4.227,
        "k_fill": 0.05,
        "tau_leak": 120000.0,
        "K_trig": 0.0022,
        "n_trig": 8.0,
        "Vmax_CICR": 0.005,
        "tau_CICR": 250.0,
        "g_cicr": 500.0,
        "tau_eff": 244.0,
    }

    def unpack_params(self, x):
        gb = {
            'gamma_d': x[0], 'gamma_p': x[1],
            'a00': x[2], 'a01': x[3], 'a10': x[4], 'a11': x[5],
            'a20': x[6], 'a21': x[7], 'a30': x[8], 'a31': x[9],
        }
        cicr = {
            'k_fill': x[10], 'tau_leak': x[11],
            'K_trig': x[12], 'n_trig': x[13], 'Vmax_CICR': x[14],
            'tau_CICR': x[15], 'g_cicr': x[16], 'tau_eff': x[17],
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

                T = (ca_e**cicr['n_trig']) / (
                    cicr['K_trig']**cicr['n_trig'] + ca_e**cicr['n_trig'] + 1e-12)

                eff_decay = jnp.exp(-dt / cicr['tau_eff'])
                effcai_new = effcai * eff_decay + ca_e * cicr['tau_eff'] * (1.0 - eff_decay)

                fill_drive = jnp.maximum(0.0, effcai_new - theta_d)
                A = cicr['k_fill'] * fill_drive
                B = A + (1.0 / cicr['tau_leak']) + cicr['Vmax_CICR'] * T
                P_inf = A / (B + 1e-12)
                P_new = P_inf + (P - P_inf) * jnp.exp(-dt * B)

                J_rel = cicr['Vmax_CICR'] * P_new * T
                decay_cicr = jnp.exp(-dt / cicr['tau_CICR'])
                ca_cicr_new = ca_cicr * decay_cicr + J_rel * cicr['tau_CICR'] * (1.0 - decay_cicr)

                gamma_p_eff = gb['gamma_p'] + cicr['g_cicr'] * ca_cicr_new

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
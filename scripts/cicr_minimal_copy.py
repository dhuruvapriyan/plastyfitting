#!/usr/bin/env python3
"""
Minimal CICR Priming Model (JAX).

Architecture based on:
  - Hong & Ross (2007): ER priming via CaMKII, τ_P ~ 2-3 min
  - Caya-Bissonnette (2023): CICR sustained release ~ 0.5-1s (STAPCD)
  - Chindemi et al. (2022): GB plasticity framework (θ_d, θ_p thresholds)

Design Principles:
  1. Priming variable P charges continuously from raw synaptic calcium (ca_ext),
     dP/dt = delta_P * ca_ext * (1-P) - P/tau_P. Since +10ms has ~46% higher
     Ca peaks than -10ms, P accumulates faster for +10ms, compounding over
     10 spikes.
  2. When P > alpha_thresh * θ_p → sustained CICR event (slow Ca² release from
     ER over ~500ms-1s). P is consumed (ER emptied). Trigger tied to θ_p.
  3. Refractory gate h_ref prevents immediate re-triggering.
  4. CICR calcium (ca_cicr) adds to effcai, boosting subsequent pairing events
     from θ_d zone into θ_p zone → LTP.
  5. Without CICR: both +10ms and -10ms produce similar effcai ~ θ_d → LTD.
     With CICR (only +10ms): effcai + ca_cicr > θ_p → LTP.

State variables (carry): P, ca_cicr, h_ref, effcai, rho  (5 states)

Free parameters: 10 (GB plasticity) + 5 (CICR) = 15 total
  CICR params: delta_P, tau_P, alpha_thresh, A_cicr, tau_cicr
"""

import numpy as np
import jax.numpy as jnp
from cicr_common import CICRModel

# Fixed biophysical constants
CAI_REST = 70e-6       # Resting cytosolic calcium (mM)

# Refractory gate constants (from Caya-Bissonnette 2023)
TAU_REF = 2000.0       # Refractory recovery time constant (ms) ~ 2s
K_H_REF = 0.0005       # Inactivation half-point (mM) = 0.5 µM


# ──────────────────────────────────────────────────────────────────────
#  Numpy debug simulation
# ──────────────────────────────────────────────────────────────────────

def _debug_sim(cai_1syn, t, cp, cq, is_apical, dp):
    """Numpy debug simulation — Minimal CICR Priming Model."""
    gamma_d = float(dp.get("gamma_d_GB_GluSynapse", 150.0))
    gamma_p = float(dp.get("gamma_p_GB_GluSynapse", 150.0))
    tau_eff = 278.318

    # Thresholds (already pre-multiplied in GB fits, or use full alpha formulas)
    cp_s, cq_s = cp, cq
    if is_apical:
        theta_d = dp.get("a20", 1.0) * cp_s + dp.get("a21", 1.0) * cq_s
        theta_p = dp.get("a30", 1.0) * cp_s + dp.get("a31", 1.0) * cq_s
    else:
        theta_d = dp.get("a00", 1.0) * cp_s + dp.get("a01", 1.0) * cq_s
        theta_p = dp.get("a10", 1.0) * cp_s + dp.get("a11", 1.0) * cq_s

    # CICR parameters
    delta_P   = float(dp.get("delta_P", 0.02))        # P increment per θ_d crossing
    tau_P     = float(dp.get("tau_P", 150000.0))       # Priming decay (ms) ~ 2.5 min
    alpha_thresh = float(dp.get("alpha_thresh", 0.1))  # CICR trigger: P > alpha_thresh * θ_p
    A_cicr    = float(dp.get("A_cicr", 0.01))          # CICR release amplitude (mM)
    tau_cicr  = float(dp.get("tau_cicr", 500.0))       # CICR release/decay time (ms)

    P_thresh = alpha_thresh * theta_p  # CICR trigger threshold tied to θ_p

    n = len(cai_1syn)
    cai_total = np.copy(cai_1syn)
    P_out = np.zeros(n)
    ca_cicr_out = np.zeros(n)
    h_ref_out = np.zeros(n)
    effcai_out = np.zeros(n)
    rho_out = np.zeros(n)

    # Initial state
    P = 0.0
    ca_cicr = 0.0
    h_ref = 1.0
    effcai = 0.0
    rho = float(dp.get("rho0", 0.5))
    rho_out[0] = rho
    h_ref_out[0] = 1.0

    for i in range(n - 1):
        dt = t[i + 1] - t[i]
        if dt <= 0:
            P_out[i + 1] = P
            ca_cicr_out[i + 1] = ca_cicr
            h_ref_out[i + 1] = h_ref
            effcai_out[i + 1] = effcai
            rho_out[i + 1] = rho
            continue

        ca_ext = max(0.0, cai_1syn[i] - CAI_REST)

        # ── 1. Priming: continuous calcium-driven charging (ER store filling) ──
        # P charges proportionally to ca_ext amplitude. +10ms has ~46% higher
        # Ca peaks → P accumulates faster, compounding over 10 spikes.
        dP = delta_P * ca_ext * (1.0 - P) - P / tau_P
        P_new = max(0.0, min(1.0, P + dt * dP))

        # ── 2. CICR trigger: when P > alpha_thresh * θ_p AND h_ref > 0.5 ──
        P_peak = P_new  # save pre-trigger peak for visualization (reset happens below)
        if P_new > P_thresh and h_ref > 0.5:
            # Sustained release: inject Ca into ca_cicr pool
            ca_cicr_inject = A_cicr * h_ref  # Amplitude modulated by refractory state
            ca_cicr = ca_cicr + ca_cicr_inject
            # Consume priming (ER empties)
            P_new = 0.0
            # Slam refractory gate shut
            h_ref = 0.0

        # ── 3. ca_cicr decay (slow release profile, τ ~ 500ms) ──
        ca_cicr = max(0.0, ca_cicr * np.exp(-dt / tau_cicr))

        # ── 4. Refractory gate recovery ──
        # h_inf depends on ca_cicr: stays shut while ca_cicr is high
        k_h2 = K_H_REF ** 2
        h_inf = k_h2 / (k_h2 + ca_cicr ** 2 + 1e-30)
        dh = (h_inf - h_ref) / TAU_REF
        h_ref = min(1.0, max(0.0, h_ref + dt * dh))

        # ── 5. Effective calcium integrator (ca_ext + ca_cicr) ──
        d_e = np.exp(-dt / tau_eff)
        effcai = effcai * d_e + (ca_ext + ca_cicr) * tau_eff * (1.0 - d_e)

        # ── 6. Plasticity (Graupner-Brunel) ──
        pot = 1.0 if effcai > theta_p else 0.0
        dep = 1.0 if effcai > theta_d else 0.0
        drho = (-rho * (1 - rho) * (0.5 - rho)
                + gamma_p * (1 - rho) * pot
                - gamma_d * rho * dep) / 70000.0
        rho = min(1.0, max(0.0, rho + dt * drho))

        P = P_new
        P_out[i + 1] = P_peak  # pre-trigger peak — shows accumulation, not post-reset 0
        ca_cicr_out[i + 1] = ca_cicr
        h_ref_out[i + 1] = h_ref
        effcai_out[i + 1] = effcai
        rho_out[i + 1] = rho

    return {
        "cai_total": cai_total, "priming": P_out, "ca_er": np.zeros(n),
        "ca_cicr": ca_cicr_out, "ca_ryr": np.zeros(n), "ca_ip3r": np.zeros(n),
        "h_ref": h_ref_out, "effcai": effcai_out, "rho": rho_out,
    }


# ──────────────────────────────────────────────────────────────────────
#  JAX Model Class
# ──────────────────────────────────────────────────────────────────────

class CICRPrimingModel(CICRModel):
    DESCRIPTION = "Minimal CICR Priming (P → sustained ER release)"
    NEEDS_THRESHOLD_TRACES = False  # c_pre/c_post are baked scalars; no trace arrays needed

    FIT_PARAMS = [
        # --- GB Plasticity Parameters (10) ---
        ("gamma_d_GB_GluSynapse", 50.0, 250.0),
        ("gamma_p_GB_GluSynapse", 50.0, 300.0),
        ("a00", 1.0, 5.0), ("a01", 1.0, 5.0),       # θ_d basal  — upper bound keeps θ_d reachable by effcai
        ("a10", 1.0, 5.0), ("a11", 1.0, 5.0),     # θ_p basal — lower bound > θ_d max, forces CICR for LTP
        ("a20", 1.0, 5.0), ("a21", 1.0, 5.0),       # θ_d apical — same reachability constraint
        ("a30", 1.0, 5.0), ("a31", 1.0, 5.0),     # θ_p apical — same constraint
        # --- CICR Priming Parameters (5) ---
        ("delta_P", 0.1, 50.0),           # P charging rate (1/(mM·ms)) — continuous, with (1-P) saturation
        ("tau_P", 60000.0, 300000.0),     # Priming decay time constant (ms) [1-5 min]
        ("alpha_thresh", 0.5, 10.0),      # CICR trigger: P > alpha_thresh * θ_p — wider range for discrimination
        ("A_cicr", 0.0005, 0.005),        # CICR release amplitude (mM) ~ order of cai_CR peak
        ("tau_cicr", 100.0, 2000.0),      # CICR release decay time (ms) [100ms-2s]
    ]

    DEFAULT_PARAMS = {
        # GB plasticity (from previous fit — will be re-fit)
        "gamma_d_GB_GluSynapse": 244.27, "gamma_p_GB_GluSynapse": 201.89,
        "a00": 1.38, "a01": 2.65, "a10": 4.07, "a11": 4.20,
        "a20": 3.0, "a21": 1.47, "a30": 5.0, "a31": 3.5,
        # CICR priming defaults
        "delta_P": 5.0,           # P charging rate (1/(mM·ms))
        "tau_P": 150000.0,        # ~2.5 min decay (Hong & Ross: 2-3 min)
        "alpha_thresh": 0.5,      # CICR trigger: P > 0.5 * θ_p (raised lower bound)
        "A_cicr": 0.002,          # 2 µM CICR release amplitude (~ cai_CR peak)
        "tau_cicr": 500.0,        # 500 ms sustained release (Caya-Bissonnette ~1s)
    }

    def unpack_params(self, x):
        gb = {
            'gamma_d': x[0], 'gamma_p': x[1],
            'a00': x[2], 'a01': x[3], 'a10': x[4], 'a11': x[5],
            'a20': x[6], 'a21': x[7], 'a30': x[8], 'a31': x[9],
        }
        cicr = {
            'tau_eff': 278.318,       # Locked to NEURON trace extraction tau
            'delta_P': x[10],
            'tau_P': x[11],
            'alpha_thresh': x[12],
            'A_cicr': x[13],
            'tau_cicr': x[14],
        }
        return gb, cicr

    def get_init_fn(self):
        def init_fn(cai_first, rho0):
            # Carry: (P, ca_cicr, h_ref, effcai, rho)
            return (0.0, 0.0, 1.0, 0.0, rho0)
        return init_fn

    def get_debug_sim_fn(self):
        return _debug_sim

    def prepare_plot_pair(self, pair_item):
        """Swap c_pre/c_post → cai_CR-based thresholds, matching setup_jax."""
        if pair_item.get("c_pre_cr") is not None:
            return {**pair_item, "c_pre": pair_item["c_pre_cr"], "c_post": pair_item["c_post_cr"]}
        return pair_item

    def setup_jax(self, *args, **kwargs):
        """Swap c_pre/c_post to cai_CR-based versions.

        This model integrates cai_CR (not shaft_cai) in its effcai equation,
        so thresholds must also be derived from cai_CR for consistency.
        """
        super().setup_jax(*args, **kwargs)
        import logging
        _logger = logging.getLogger(__name__)
        for proto, cd in self.collated_data.items():
            if cd.get("c_pre_cr") is not None:
                _logger.info(f"  {proto}: swapping c_pre/c_post → cai_CR-based thresholds")
                self.collated_data[proto] = {
                    **cd,
                    "c_pre": cd["c_pre_cr"],
                    "c_post": cd["c_post_cr"],
                }

    def get_step_factory(self):
        def scan_factory(params, syn_params):
            gb, cicr = params
            c_pre, c_post, is_apical, cai_pre, t_pre, cai_post, t_post = syn_params

            tau_effca    = cicr['tau_eff']
            delta_P      = cicr['delta_P']
            tau_P        = cicr['tau_P']
            alpha_thresh = cicr['alpha_thresh']
            A_cicr       = cicr['A_cicr']
            tau_cicr     = cicr['tau_cicr']

            # Thresholds from c_pre, c_post (baked at tau_eff=278.318)
            theta_d = jnp.where(
                is_apical,
                gb['a20'] * c_pre + gb['a21'] * c_post,
                gb['a00'] * c_pre + gb['a01'] * c_post,
            )
            theta_p = jnp.where(
                is_apical,
                gb['a30'] * c_pre + gb['a31'] * c_post,
                gb['a10'] * c_pre + gb['a11'] * c_post,
            )

            # Pre-compute refractory gate constant
            k_h2 = K_H_REF ** 2

            # CICR trigger threshold tied to θ_p
            P_thresh = alpha_thresh * theta_p

            def scan_step(carry, inputs):
                # Carry: (P, ca_cicr, h_ref, effcai, rho)
                P, ca_cicr, h_ref, effcai, rho = carry
                cai_raw, dt, nves_i = inputs

                ca_ext = jnp.maximum(0.0, cai_raw - CAI_REST)

                # ── 1. Priming: continuous calcium-driven charging ──
                # P charges proportionally to ca_ext amplitude.
                dP = delta_P * ca_ext * (1.0 - P) - P / tau_P
                P_updated = jnp.clip(P + dt * dP, 0.0, 1.0)

                # ── 2. CICR trigger: P > alpha_thresh * θ_p ──
                trigger_P = jnp.where(P_updated > P_thresh, 1.0, 0.0)
                trigger_h = jnp.where(h_ref > 0.5, 1.0, 0.0)
                trigger = trigger_P * trigger_h

                # CICR injection: amplitude scaled by h_ref
                ca_cicr_inject = A_cicr * h_ref * trigger

                # Consume P and slam h_ref shut on trigger
                P_new = jnp.where(trigger > 0.5, 0.0, P_updated)
                h_ref_post_trigger = jnp.where(trigger > 0.5, 0.0, h_ref)

                # ── 3. ca_cicr decay (exponential, sustained release) ──
                ca_cicr_new = jnp.maximum(
                    0.0,
                    (ca_cicr + ca_cicr_inject) * jnp.exp(-dt / tau_cicr),
                )

                # ── 4. Refractory gate recovery ──
                h_inf = k_h2 / (k_h2 + ca_cicr_new ** 2 + 1e-30)
                dh = (h_inf - h_ref_post_trigger) / TAU_REF
                h_ref_new = jnp.where(
                    dt > 0,
                    jnp.clip(h_ref_post_trigger + dt * dh, 0.0, 1.0),
                    h_ref_post_trigger,
                )

                # ── 5. Effective calcium integrator (ca_ext + ca_cicr) ──
                decay_eff = jnp.exp(-dt / tau_effca)
                effcai_new = jnp.where(
                    dt > 0,
                    effcai * decay_eff + (ca_ext + ca_cicr_new) * tau_effca * (1.0 - decay_eff),
                    effcai,
                )

                # ── 6. Plasticity (Graupner-Brunel) ──
                pot = jnp.where(effcai_new > theta_p, 1.0, 0.0)
                dep = jnp.where(effcai_new > theta_d, 1.0, 0.0)
                drho = (
                    -rho * (1.0 - rho) * (0.5 - rho)
                    + pot * gb['gamma_p'] * (1.0 - rho)
                    - dep * gb['gamma_d'] * rho
                ) / 70000.0
                rho_new = jnp.where(
                    dt > 0,
                    jnp.clip(rho + dt * drho, 0.0, 1.0),
                    rho,
                )

                return (P_new, ca_cicr_new, h_ref_new, effcai_new, rho_new), None

            return scan_step
        return scan_factory


if __name__ == "__main__":
    CICRPrimingModel().run()

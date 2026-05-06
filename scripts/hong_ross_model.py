#!/usr/bin/env python3
"""
Hong & Ross (2007) Inspired ER Priming Model (JAX).

Biological principles:
1. Action potentials (large Ca2+ peaks) non-linearly fill the ER stores.
2. The filled state (Priming, P_ER) decays very slowly (2-3 mins).
3. A triggered IP3-mediated CICR release requires the ER to be primed.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import expit
from cicr_common import CICRModel

# Fixed biophysical constants
CAI_REST = 70e-6       # Resting cytosolic calcium (mM)
TAU_REF = 2000.0       # Refractory recovery time constant (ms)
K_H_REF = 0.0005       # Inactivation half-point (mM) = 0.5 uM
_TAU_EFF_BAKED = 278.318

def _apply_hong_ross_debug(cai_1syn, t, cp, cq, is_apical, dp):
    """
    Plain-NumPy debug simulation of the Hong & Ross priming model.
    """
    n = len(cai_1syn)
    dt = t[1] - t[0] if n > 1 else 0.1
    cai_total = np.zeros(n)
    P_out = np.zeros(n)
    IP3_out = np.zeros(n)
    ca_cicr_out = np.zeros(n)
    h_ref_out = np.zeros(n)
    effcai_out = np.zeros(n)
    rho_out = np.zeros(n)

    # Initial state
    P_ER = 0.0
    IP3 = 0.0
    ca_cicr = 0.0
    h_ref = 1.0
    effcai = 0.0
    rho = float(dp.get("rho0", 0.5))

    # Parameters
    gamma_d = dp["gamma_d"]
    gamma_p = dp["gamma_p"]
    
    # Thresholds
    if is_apical:
        theta_d = dp["a20"] * cp + dp["a21"] * cq
        theta_p = dp["a30"] * cp + dp["a31"] * cq
    else:
        theta_d = dp["a00"] * cp + dp["a01"] * cq
        theta_p = dp["a10"] * cp + dp["a11"] * cq
        
    tau_eff = dp["tau_eff"]
    kappa = dp["kappa"]
    tau_P = dp["tau_P"]
    tau_IP3 = dp["tau_IP3"]
    k_IP3 = dp["k_IP3"]
    A_cicr = dp["A_cicr"]
    tau_cicr = dp["tau_cicr"]
    J_thresh = dp["J_thresh"]

    # Simple inst-diff for triggering calculation
    cai_diff = np.diff(cai_1syn, prepend=cai_1syn[0])
    cai_diff[cai_diff < 0] = 0

    for i in range(n):
        # 1. Calcium input
        ca_ext = max(0.0, cai_1syn[i] - CAI_REST)
        I_ca = cai_diff[i]
        
        # 2. Local Synaptic IP3 Trace
        # If there's a sharp influx, it produces IP3 (synaptic event placeholder)
        if I_ca > 1e-4:
            IP3 += k_IP3 * I_ca
        IP3 -= (IP3 / tau_IP3) * dt
        IP3 = max(0.0, IP3)
        
        # 3. Non-linear ER Filling (Global Priming)
        dP = kappa * (ca_ext**2) * (1.0 - P_ER) - (P_ER / tau_P)
        P_ER = max(0.0, min(1.0, P_ER + dt * dP))
        
        # 4. CICR Triggering (Overlap of Synaptic Trace and Global Trace)
        J_drive = IP3 * P_ER
        trigger = 0.0
        
        # Massive release only when both IP3 and P_ER are high, and not refractory
        if J_drive > J_thresh and h_ref > 0.5:
            trigger = 1.0
            
        ca_cicr_inject = A_cicr * h_ref * trigger
        
        # Consume ER priming and shut refractory gate if triggered
        if trigger > 0.5:
            P_ER = 0.0
            IP3 = 0.0  # Consume local receptor trace too
            h_ref = 0.0
            
        # 5. CICR decay
        ca_cicr = max(0.0, (ca_cicr + ca_cicr_inject) * np.exp(-dt / tau_cicr))
        
        # 6. Refractory recovery
        k_h2 = K_H_REF**2
        h_inf = k_h2 / (k_h2 + ca_cicr**2 + 1e-30)
        dh = (h_inf - h_ref) / TAU_REF
        h_ref = max(0.0, min(1.0, h_ref + dt * dh))
        
        # 7. Effective Calcium Integrator
        d_e = np.exp(-dt / tau_eff)
        effcai = effcai * d_e + (ca_ext + ca_cicr) * tau_eff * (1.0 - d_e)
        
        # 8. GB Plasticity
        pot = 1.0 if effcai > theta_p else 0.0
        dep = 1.0 if effcai > theta_d else 0.0
        drho = (-rho * (1.0 - rho) * (0.5 - rho) + gamma_p * (1.0 - rho) * pot - gamma_d * rho * dep) / 70000.0
        rho = max(0.0, min(1.0, rho + dt * drho))

        cai_total[i] = cai_1syn[i] + ca_cicr
        P_out[i] = P_ER
        IP3_out[i] = IP3
        ca_cicr_out[i] = ca_cicr
        h_ref_out[i] = h_ref
        effcai_out[i] = effcai
        rho_out[i] = rho

    return {
        "cai_total": cai_total,
        "priming": P_out,      # We can plot ER Priming on the priming trace
        "ca_er": IP3_out,      # Hijack ca_er trace to debug IP3
        "ca_cicr": ca_cicr_out,
        "h_ref": h_ref_out,
        "effcai": effcai_out,
        "rho": rho_out
    }

class HongRossModel(CICRModel):
    DESCRIPTION = "Hong & Ross ER Priming (Non-linear Filling & Synaptic IP3)"
    NEEDS_THRESHOLD_TRACES = False
        
    FIT_PARAMS = [
            ("gamma_d", 50.0, 500.0), ("gamma_p", 150.0, 600.0),
            ("a00", 1.0, 5.0), ("a01", 1.0, 5.0),
            ("a10", 1.0, 5.0), ("a11", 1.0, 5.0),
            ("a20", 1.0, 5.0), ("a21", 1.0, 5.0),
            ("a30", 1.0, 5.0), ("a31", 1.0, 5.0),
            ("tau_eff", 50.0, 500.0),
            
            # Hong & Ross Priming Parameters
            ("kappa", 100.0, 50000.0),      # ER Filling non-linear rate multiplier

            ("tau_P", 30000.0, 180000.0),   # Priming decay (30s to 3 mins)
            ("tau_IP3", 100.0, 3000.0),   # IP3 decay time (100ms to 3s)
            ("k_IP3", 0.1, 100.0),        # IP3 production per syn spike
            ("A_cicr", 0.0005, 0.05),     # Amplitude of IP3 CICR (mM)
            ("tau_cicr", 100.0, 1500.0),    # CICR flux decay time (ms)
            ("J_thresh", 1e-6, 0.1),      # P_ER * IP3 needed to pop IP3Rs (lowered to allow trigger)
        ]
    
    DEFAULT_PARAMS = {
        "gamma_d": 181.4, "gamma_p": 209.9,
        "a00": 1.03, "a01": 1.94, "a10": 1.92, "a11": 3.56,
        "a20": 3.16, "a21": 2.69, "a30": 7.73, "a31": 2.74,
        "tau_eff": 278.318,
        "kappa": 5000.0,
        "tau_P": 150000.0,
        "tau_IP3": 1000.0,
        "k_IP3": 10.0,
        "A_cicr": 0.002,
        "tau_cicr": 500.0,
        "J_thresh": 0.0005
    }

    def unpack_params(self, x):
        gb = {'gamma_d': x[0], 'gamma_p': x[1], 'a00': x[2], 'a01': x[3],
              'a10': x[4], 'a11': x[5], 'a20': x[6], 'a21': x[7], 'a30': x[8], 'a31': x[9]}
        hr = {
            'tau_eff': x[10], 'kappa': x[11], 'tau_P': x[12], 'tau_IP3': x[13],
            'k_IP3': x[14], 'A_cicr': x[15], 'tau_cicr': x[16], 'J_thresh': x[17]
        }
        return gb, hr

    def get_debug_sim_fn(self):
        return _apply_hong_ross_debug

    def get_init_fn(self):
        def init_fn(cai_first, rho0):
            # Carry: (c_prev, effcai, rho, IP3, P_ER, ca_cicr, h_ref)
            return (cai_first, 0.0, rho0, 0.0, 0.0, 0.0, 1.0)
        return init_fn
        
    def prepare_plot_pair(self, pair_item):
        """Swap c_pre/c_post → cai_CR-based thresholds for plotting."""
        if pair_item.get("c_pre_cr") is not None:
            return {**pair_item, "c_pre": pair_item["c_pre_cr"], "c_post": pair_item["c_post_cr"]}
        return pair_item

    def setup_jax(self, *args, **kwargs):
        """Swap c_pre/c_post to cai_CR-based versions for the whole dataset."""
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
            gb, hr = params
            c_pre_baked, c_post_baked, is_apical, _cai_pre, t_pre, _cai_post, t_post = syn_params

            tau_effca = hr['tau_eff']
            kappa = hr['kappa']
            tau_P = hr['tau_P']
            tau_IP3 = hr['tau_IP3']
            k_IP3 = hr['k_IP3']
            A_cicr = hr['A_cicr']
            tau_cicr = hr['tau_cicr']
            J_thresh = hr['J_thresh']

            k_h2 = K_H_REF ** 2

            def compute_peak_effcai(cai_trace, t_trace):
                dt_trace = jnp.diff(t_trace, prepend=t_trace[0])
                dt_trace = jnp.where(dt_trace <= 0, 1e-6, dt_trace)
                
                def step_fn(carry, inputs):
                    c_prev, effcai, IP3, P_ER, ca_cicr, h_ref = carry
                    cai_raw, dt = inputs
                    
                    ca_ext = jnp.maximum(0.0, cai_raw - CAI_REST)
                    I_ca = jnp.maximum(0.0, cai_raw - c_prev)

                    IP3_jump = jnp.where(I_ca > 1e-4, IP3 + k_IP3 * I_ca, IP3)
                    IP3_new = jnp.maximum(0.0, IP3_jump - (IP3_jump / tau_IP3) * dt)

                    dP = kappa * (ca_ext**2) * (1.0 - P_ER) - (P_ER / tau_P)
                    P_ER_new = jnp.clip(P_ER + dt * dP, 0.0, 1.0)

                    J_drive = IP3_new * P_ER_new
                    trig_J = jnp.where(J_drive > J_thresh, 1.0, 0.0)
                    trig_h = jnp.where(h_ref > 0.5, 1.0, 0.0)
                    trigger = trig_J * trig_h

                    ca_cicr_inject = A_cicr * h_ref * trigger

                    P_ER_post = jnp.where(trigger > 0.5, 0.0, P_ER_new)
                    IP3_post = jnp.where(trigger > 0.5, 0.0, IP3_new)
                    h_ref_post = jnp.where(trigger > 0.5, 0.0, h_ref)

                    ca_cicr_new = jnp.maximum(0.0, (ca_cicr + ca_cicr_inject) * jnp.exp(-dt / tau_cicr))

                    h_inf = k_h2 / (k_h2 + ca_cicr_new**2 + 1e-30)
                    dh = (h_inf - h_ref_post) / TAU_REF
                    h_ref_new = jnp.where(dt > 0, jnp.clip(h_ref_post + dt * dh, 0.0, 1.0), h_ref_post)

                    decay_eff = jnp.exp(-dt / tau_effca)
                    effcai_new = jnp.where(
                        dt > 0,
                        effcai * decay_eff + (ca_ext + ca_cicr_new) * tau_effca * (1.0 - decay_eff),
                        effcai
                    )
                    
                    return (cai_raw, effcai_new, IP3_post, P_ER_post, ca_cicr_new, h_ref_new), effcai_new

                init = (cai_trace[0], 0.0, 0.0, 0.0, 0.0, 1.0)
                _, eff_trace = jax.lax.scan(step_fn, init, (cai_trace, dt_trace))
                return jnp.max(eff_trace)
            
            c_pre = compute_peak_effcai(_cai_pre, t_pre)
            c_post = compute_peak_effcai(_cai_post, t_post)

            theta_d = jnp.where(is_apical, gb['a20']*c_pre + gb['a21']*c_post, gb['a00']*c_pre + gb['a01']*c_post)
            theta_p = jnp.where(is_apical, gb['a30']*c_pre + gb['a31']*c_post, gb['a10']*c_pre + gb['a11']*c_post)

            def scan_step(carry, step_inputs):
                c_prev, effcai, rho, IP3, P_ER, ca_cicr, h_ref = carry
                cai_raw, dt, nves_i = step_inputs
                
                ca_ext = jnp.maximum(0.0, cai_raw - CAI_REST)
                I_ca = jnp.maximum(0.0, cai_raw - c_prev)

                # 1. IP3 Synaptic Trace
                IP3_jump = jnp.where(I_ca > 1e-4, IP3 + k_IP3 * I_ca, IP3)
                IP3_new = jnp.maximum(0.0, IP3_jump - (IP3_jump / tau_IP3) * dt)

                # 2. Non-linear ER Filling
                dP = kappa * (ca_ext**2) * (1.0 - P_ER) - (P_ER / tau_P)
                P_ER_new = jnp.clip(P_ER + dt * dP, 0.0, 1.0)

                # 3. CICR Triggering
                J_drive = IP3_new * P_ER_new
                trig_J = jnp.where(J_drive > J_thresh, 1.0, 0.0)
                trig_h = jnp.where(h_ref > 0.5, 1.0, 0.0)
                trigger = trig_J * trig_h

                ca_cicr_inject = A_cicr * h_ref * trigger

                # Consume ER priming and slam refractory gate shut on trigger
                P_ER_post = jnp.where(trigger > 0.5, 0.0, P_ER_new)
                IP3_post = jnp.where(trigger > 0.5, 0.0, IP3_new)
                h_ref_post = jnp.where(trigger > 0.5, 0.0, h_ref)

                # 4. CICR decay (exponential)
                ca_cicr_new = jnp.maximum(0.0, (ca_cicr + ca_cicr_inject) * jnp.exp(-dt / tau_cicr))

                # 5. Refractory gate recovery
                h_inf = k_h2 / (k_h2 + ca_cicr_new**2 + 1e-30)
                dh = (h_inf - h_ref_post) / TAU_REF
                h_ref_new = jnp.where(dt > 0, jnp.clip(h_ref_post + dt * dh, 0.0, 1.0), h_ref_post)

                # 6. Effective calcium integrator
                decay_eff = jnp.exp(-dt / tau_effca)
                effcai_new = jnp.where(
                    dt > 0,
                    effcai * decay_eff + (ca_ext + ca_cicr_new) * tau_effca * (1.0 - decay_eff),
                    effcai
                )

                # 7. Plasticity (Graupner-Brunel)
                pot = jnp.where(effcai_new > theta_p, 1.0, 0.0)
                dep = jnp.where(effcai_new > theta_d, 1.0, 0.0)
                drho = (-rho * (1.0 - rho) * (0.5 - rho) + pot * gb['gamma_p'] * (1.0 - rho) - dep * gb['gamma_d'] * rho) / 70000.0
                rho_new = jnp.where(dt > 0, jnp.clip(rho + dt * drho, 0.0, 1.0), rho)

                next_carry = (cai_raw, effcai_new, rho_new, IP3_post, P_ER_post, ca_cicr_new, h_ref_new)
                return next_carry, None

            return scan_step
        return scan_factory

if __name__ == "__main__":
    HongRossModel().run()

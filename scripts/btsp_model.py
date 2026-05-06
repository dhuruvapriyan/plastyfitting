#!/usr/bin/env python3
"""
BTSP Phenomenological Model (JAX).

Based on Caya-Bissonnette et al. 2023 (Equations 1-5).
Implements Short-Term Associative Plasticity of Ca2+ Dynamics (STAPCD).
The ER acts as an eligibility trace. An initial event "loads" the ER state (e).
A subsequent event occurring while (e) is high gets its calcium amplitude
and decay time amplified, triggering LTP via the Graupner-Brunel thresholds.
"""

import numpy as np
import jax.numpy as jnp
from jax.scipy.special import expit
from cicr_common import CICRModel

def _apply_btsp_debug(cai_1syn, t, cp, cq, is_apical, dp):
    """
    Plain-NumPy debug simulation of the phenomenological BTSP model.
    """
    n = len(t)
    dt = t[1] - t[0] if n > 1 else 0.1
    c = np.zeros(n)
    e = np.zeros(n)
    r = np.zeros(n)
    A_ER = np.ones(n)
    T_ER = np.ones(n)

    # Simplified instantaneous differences for peak detection (I_syn + I_AP)
    cai_drv = np.diff(cai_1syn, prepend=cai_1syn[0])
    cai_drv[cai_drv < 0] = 0

    c_cur = dp.get("min_ca", 70e-6)
    e_cur = 0.0
    r_cur = 0.0

    tau_c = dp.get("tau_eff", 278.318)
    tau_ER = dp.get("tau_ER", 1000.0)
    tau_r = dp.get("tau_r", 650.0)
    k_e = dp.get("k_e", 1.0)
    k_f = dp.get("k_f", 1.0)
    k_t = dp.get("k_t", 1.0)
    beta = dp.get("beta", 0.002)
    k_r = dp.get("k_r", 80.0)

    for i in range(n):
        I_ca = cai_drv[i]
        
        # When an event arrives, jump the ER state and Refractory state
        if I_ca > 1e-4:
            e_cur += k_e * I_ca
            r_cur += k_r * I_ca
            
        # 1. Modulators
        A_cur = 1.0 + k_f * e_cur
        sigma_r = 1.0 / (1.0 + np.exp(-beta * r_cur))
        T_cur = 1.0 + k_t * e_cur * sigma_r

        # 2. Cytosolic Ca dynamics
        c_next = c_cur - (c_cur / (tau_c * T_cur))*dt + (A_cur * I_ca)
        c_cur = c_next

        # 3. ER and Refractory trace decay
        e_cur -= (e_cur / tau_ER) * dt
        r_cur -= (r_cur / tau_r) * dt

        c[i] = c_cur
        e[i] = e_cur
        r[i] = r_cur
        A_ER[i] = A_cur
        T_ER[i] = T_cur

    return {
        "cai_total": c,
        "priming": e,  # ER Eligibility trace
        "ca_er": r,    # Refractory trace
        "ca_cicr": A_ER * T_ER  # Overall amplification factor for visualization
    }

class BTSPModel(CICRModel):
    DESCRIPTION = "BTSP Phenomenological Model (Caya-Bissonnette 2023)"
    NEEDS_THRESHOLD_TRACES = False
        
    FIT_PARAMS = [
            ("gamma_d", 50.0, 500.0), ("gamma_p", 150.0, 600.0),
            ("a00", 1.0, 50.0), ("a01", 1.0, 50.0),
            ("a10", 1.0, 50.0), ("a11", 1.0, 50.0),
            ("a20", 1.0, 50.0), ("a21", 1.0, 50.0),
            ("a30", 1.0, 50.0), ("a31", 1.0, 50.0),
            ("tau_eff", 50.0, 500.0),
            
            # BTSP Specific Parameters
            ("tau_ER", 100.0, 3000.0),  # ER eligibility decay (ms)
            ("tau_r", 100.0, 2000.0),   # Refractory decay (ms)
            ("k_e", 0.01, 100.0),        # ER Jump strength
            ("k_f", 0.0, 50.0),         # Amplitude modulation strength
            ("k_t", 0.0, 50.0),         # Time modulation strength
            ("beta", 0.0001, 1.0),      # Sigmoid sensitivity
            ("k_r", 0.1, 1000.0),        # Refractory jump strength
        ]
    
    DEFAULT_PARAMS = {
        "gamma_d": 181.4, "gamma_p": 209.9,
        "a00": 1.03, "a01": 1.94, "a10": 1.92, "a11": 3.56,
        "a20": 3.16, "a21": 2.69, "a30": 7.73, "a31": 2.74,
        "tau_eff": 278.318,
        "tau_ER": 1000.0, "tau_r": 650.0, "k_e": 10.0,
        "k_f": 1.0, "k_t": 1.0, "beta": 0.01, "k_r": 80.0
    }

    def unpack_params(self, x):
        gb = {'gamma_d': x[0], 'gamma_p': x[1], 'a00': x[2], 'a01': x[3],
              'a10': x[4], 'a11': x[5], 'a20': x[6], 'a21': x[7], 'a30': x[8], 'a31': x[9]}
        btsp = {
            'tau_eff': x[10], 'tau_ER': x[11], 'tau_r': x[12], 'k_e': x[13],
            'k_f': x[14], 'k_t': x[15], 'beta': x[16], 'k_r': x[17]
        }
        return gb, btsp

    def get_debug_sim_fn(self):
        return _apply_btsp_debug

    def get_init_fn(self):
        def init_fn(cai_first, rho0):
            # Carry: (c_raw, effcai, rho, e_eligibility, r_refractory)
            return (cai_first, 0.0, rho0, 0.0, 0.0)
        return init_fn

    def get_step_factory(self):
        _TAU_EFF_BAKED = 278.318
        
        def scan_factory(params, syn_params):
            gb, btsp = params
            c_pre_baked, c_post_baked, is_apical, _cai_pre, t_pre, _cai_post, t_post = syn_params

            tau_effca = btsp['tau_eff']
            tau_ER = btsp['tau_ER']
            tau_r = btsp['tau_r']
            k_e = btsp['k_e']
            k_f = btsp['k_f']
            k_t = btsp['k_t']
            beta = btsp['beta']
            k_r = btsp['k_r']

            tau_scale = tau_effca / _TAU_EFF_BAKED
            c_pre  = c_pre_baked  * tau_scale
            c_post = c_post_baked * tau_scale

            theta_d = jnp.where(is_apical,
                                 gb['a20']*c_pre + gb['a21']*c_post,
                                 gb['a00']*c_pre + gb['a01']*c_post)
            theta_p = jnp.where(is_apical,
                                 gb['a30']*c_pre + gb['a31']*c_post,
                                 gb['a10']*c_pre + gb['a11']*c_post)

            dt = 0.1

            def scan_step(carry, step_inputs):
                c_prev, effcai, rho, e_prev, r_prev = carry
                t_cur, cai_cur_pre, cai_cur_post = step_inputs
                cai_raw_cur = jnp.where(is_apical, cai_cur_post, cai_cur_pre)
                
                # Derive driving current (spike arrival) from raw calcium slope
                I_ca = jnp.maximum(cai_raw_cur - c_prev, 0.0)
                
                # 1. ER state jump on activity
                # In continuous time, we jump immediately if there's influx
                e_jump = jnp.where(I_ca > 1e-4, e_prev + k_e * I_ca, e_prev)
                r_jump = jnp.where(I_ca > 1e-4, r_prev + k_r * I_ca, r_prev)

                # 2. Compute Modulators
                A_ER = 1.0 + k_f * e_jump
                sigma_r = expit(beta * r_jump)  # JAX optimized sigmoid = 1 / (1 + exp(-x))
                T_ER = 1.0 + k_t * e_jump * sigma_r
                
                # 3. Cytosolic Ca dynamics (which becomes the effective calcium for plasticity)
                # Instead of standard effcai tracking, we treat effcai as the modulated calcium 'c'
                effcai_next = effcai - (effcai / (tau_effca * T_ER)) * dt + (A_ER * I_ca)

                # 4. Graupner-Brunel Plasticity Rules (using modulated effcai)
                pow1 = jnp.where(effcai_next > theta_d, 1.0, 0.0)
                pow2 = jnp.where(effcai_next > theta_p, 1.0, 0.0)
                drho = (-rho * (1.0 - rho) * (0.5 - rho)
                        + gb['gamma_p'] * (1.0 - rho) * pow2
                        - gb['gamma_d'] * rho * pow1) / (1e3 * 70.0) # tau_ind=70s
                rho_next = rho + dt * drho

                # 5. Decay the eligibility and refractory traces
                e_next = jnp.where(dt > 0, e_jump - (e_jump / tau_ER) * dt, e_jump)
                r_next = jnp.where(dt > 0, r_jump - (r_jump / tau_r) * dt, r_jump)

                next_carry = (cai_raw_cur, effcai_next, rho_next, e_next, r_next)
                return next_carry, None

            return scan_step
        return scan_factory

if __name__ == "__main__":
    BTSPModel().run()

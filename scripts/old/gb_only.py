#!/usr/bin/env python3
"""
GB-Only Baseline Model (Continuous JAX).
Turns off CICR completely to fit only the 10 Graupner-Brunel parameters.
Use this to find a solid baseline before freezing GB and tuning CICR.
"""

import numpy as np
import jax.numpy as jnp
from cicr_common import CICRModel, compute_effcai_piecewise_linear_jax

def _apply_gb_only_debug(cai_1syn, t, cp, cq, is_apical, dp):
    """Plain-NumPy debug simulation. Returns zero for all CICR variables."""
    n = len(cai_1syn)
    cai_total = np.copy(cai_1syn)
    
    return {
        "cai_total": cai_total,
        "priming":   np.zeros(n), # No IP3
        "ca_er":     np.zeros(n), # No ER pool
        "ca_cicr":   np.zeros(n), # No CICR flux
    }

class GBOnlyModel(CICRModel):
    DESCRIPTION = "GB-Only Baseline (No CICR)"
        
    FIT_PARAMS = [
            # Only the 10 GB parameters are exposed to the optimizer
            ("gamma_d", 50.0, 500.0), ("gamma_p", 150.0, 500.0),
            ("a00", 1.0, 50.0), ("a01", 1.0, 50.0),
            ("a10", 1.0, 50.0), ("a11", 1.0, 50.0),
            ("a20", 1.0, 50.0), ("a21", 1.0, 50.0),
            ("a30", 1.0, 50.0), ("a31", 1.0, 50.0),
            # ("tau_eff", 50.0, 500.0),
        ]
    
    # Best parameters:
    # gamma_d                        =   192.3981  (default: 181.4000, +6.1%)
    # gamma_p                        =   466.5131  (default: 209.9000, +122.3%)
    # a00                            =    47.4882  (default: 1.0300, +4510.5%)
    # a01                            =    16.4683  (default: 1.9400, +748.9%)
    # a10                            =     1.8911  (default: 1.9200, -1.5%)
    # a11                            =     2.3481  (default: 3.5600, -34.0%)
    # a20                            =    16.1300  (default: 3.1600, +410.4%)
    # a21                            =    40.7094  (default: 2.6900, +1413.4%)
    # a30                            =     1.3948  (default: 7.7300, -82.0%)
    # a31                            =     2.4314  (default: 2.7400, -11.3%)
    # tau_eff                        =   288.8803  (default: 278.3178, +3.8%)

    DEFAULT_PARAMS = {
        "gamma_d": 192.3981, "gamma_p": 466.5131,
        "a00": 47.4882, "a01": 16.4683, "a10": 1.8911, "a11": 2.3481,
        "a20": 16.1300, "a21": 40.7094, "a30": 1.3948, "a31": 2.4314,
        "tau_eff": 278.3178,
    }

    def unpack_params(self, x):
        # Unpack only the GB variables, return empty dict for CICR (except tau_eff which is needed for integration)
        gb = {'gamma_d': x[0], 'gamma_p': x[1], 'a00': x[2], 'a01': x[3],
              'a10': x[4], 'a11': x[5], 'a20': x[6], 'a21': x[7], 'a30': x[8], 'a31': x[9]}
        cicr = {'tau_eff': x[10]}
        return gb, cicr

    def get_debug_sim_fn(self):
        return _apply_gb_only_debug

    def get_step_factory(self):
        def scan_factory(params, syn_params):
            gb, cicr = params
            c_pre_opt, c_post_opt, is_apical, cai_pre, t_pre, cai_post, t_post = syn_params
            
            tau_effca = cicr['tau_eff']
            
            # Compute cpre and cpost dynamically using piecewise linear integrator
            c_pre = jnp.max(compute_effcai_piecewise_linear_jax(cai_pre, t_pre, tau_effca=tau_effca))
            c_post = jnp.max(compute_effcai_piecewise_linear_jax(cai_post, t_post, tau_effca=tau_effca))
            
            # GB Thresholds
            theta_d = jnp.where(is_apical, gb['a20']*c_pre + gb['a21']*c_post, gb['a00']*c_pre + gb['a01']*c_post)
            theta_p = jnp.where(is_apical, gb['a30']*c_pre + gb['a31']*c_post, gb['a10']*c_pre + gb['a11']*c_post)
            
            cai_rest = 70e-6 
            
            def scan_step(carry, inputs):
                # We unpack carry, but ignore all CICR states (index 0, 1, 2, 3, 4)
                _, _, _, _, _, effcai, rho = carry
                cai_raw, dt = inputs
                
                # 1. Graupner-Brunel Plasticity Integrator (NO CICR CALCIUM ADDED)
                decay_eff = jnp.exp(-dt / tau_effca)
                
                # Notice `ca_cicr_new` is completely removed from this calculation
                effcai_new = effcai * decay_eff + (cai_raw - cai_rest) * tau_effca * (1.0 - decay_eff)
                
                pot = jnp.where(effcai_new > theta_p, 1.0, 0.0)
                dep = jnp.where(effcai_new > theta_d, 1.0, 0.0)
                
                # JAX handles cubic terms cleanly.
                drho = (-rho*(1.0-rho)*(0.5-rho) + pot*gb['gamma_p']*(1.0-rho) - dep*gb['gamma_d']*rho) / 70000.0
                rho_new  = jnp.where(dt > 0, jnp.clip(rho + dt * drho, 0.0, 1.0), rho)
                
                # Return zeros for the CICR state variables to keep shapes consistent
                return (0.0, 0.0, 0.0, 0.0, 0.0, effcai_new, rho_new), None
            return scan_step
        return scan_factory

if __name__ == "__main__":
    GBOnlyModel().run()
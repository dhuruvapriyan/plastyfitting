#!/usr/bin/env python3
"""
GB-Only Baseline Model (Continuous JAX).
Turns off CICR completely to fit only the Graupner-Brunel parameters.
"""

import logging
import numpy as np
import jax.numpy as jnp
from plastyfitting.cicr_common import CICRModel, compute_effcai_piecewise_linear_jax, _CICR_MIN_CA

# --- Persistent run log (appends across runs, one file per model) ---
_LOG_FILE = "shaft_cai_optimization.log"
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
_fh = logging.FileHandler(_LOG_FILE, mode="a")
_fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(_fh)
logger.info("=" * 70)
logger.info("shaft_cai_cpre_cpost.py loaded — new run starting")
logger.info("=" * 70)

# Fixed biophysical constants
CAI_REST = 70e-6           # Resting cytosolic calcium (mM)
SHAFT_CAI_TAU_REF = 278.318  # tau_eff at which shaft_cai Cpre/Cpost were pre-computed

def _debug_sim(cai_1syn, t, cp, cq, is_apical, dp):
    """Numpy debug simulation — GB-Only Baseline (shaft_cai Cpre/Cpost)."""
    gamma_d = float(dp.get("gamma_d_GB_GluSynapse", 150.0))
    gamma_p = float(dp.get("gamma_p_GB_GluSynapse", 150.0))
    tau_eff = float(dp.get("tau_eff", SHAFT_CAI_TAU_REF))

    # Scale pre-baked shaft_cai Cpre/Cpost to the current tau_eff
    scale = tau_eff / SHAFT_CAI_TAU_REF
    cp_s, cq_s = cp * scale, cq * scale
    if is_apical:
        theta_d = dp["a20"]*cp_s + dp["a21"]*cq_s
        theta_p = dp["a30"]*cp_s + dp["a31"]*cq_s
    else:
        theta_d = dp["a00"]*cp_s + dp["a01"]*cq_s
        theta_p = dp["a10"]*cp_s + dp["a11"]*cq_s
    theta_p = max(theta_p, theta_d + 0.01)

    n = len(cai_1syn)
    cai_total = np.copy(cai_1syn)
    effcai_out = np.zeros(n); rho_out = np.zeros(n)

    effcai = 0.0
    rho = float(dp.get("rho0", 0.5))
    rho_out[0] = rho

    for i in range(n - 1):
        dt = t[i+1] - t[i]
        if dt <= 0:
            effcai_out[i+1] = effcai; rho_out[i+1] = rho
            continue

        ca_ext = max(0.0, cai_1syn[i] - _CICR_MIN_CA)  # Subtract resting baseline
        # Effective calcium integrator
        d_e = np.exp(-dt / tau_eff)
        effcai = effcai * d_e + ca_ext * tau_eff * (1.0 - d_e)

        # Plasticity (Graupner-Brunel)
        pot = 1.0 if effcai > theta_p else 0.0
        dep = 1.0 if effcai > theta_d else 0.0
        drho = (-rho*(1-rho)*(0.5-rho) + gamma_p*(1-rho)*pot - gamma_d*rho*dep) / 70000.0
        rho = min(1.0, max(0.0, rho + dt * drho))

        effcai_out[i+1] = effcai; rho_out[i+1] = rho

    # Return empty arrays for things we don't simulate to avoid breaking plotting scripts
    return {"cai_total": cai_total, "priming": np.zeros(n), "ca_er": np.zeros(n),
            "ca_cicr": np.zeros(n), "ca_ryr": np.zeros(n), "ca_ip3r": np.zeros(n),
            "h_ref": np.zeros(n), "effcai": effcai_out, "rho": rho_out}


class GBOnlyModel(CICRModel):
    DESCRIPTION = "GB-Only Baseline (No CICR)"
    FIT_PARAMS = [
        # --- Plasticity Parameters (10) ---
        ("gamma_d_GB_GluSynapse", 50.0, 250.0), ("gamma_p_GB_GluSynapse", 50.0, 300.0),
        ("a00", 1.0, 5.0), ("a01", 1.0, 5.0), ("a10", 1.0, 5.0), ("a11", 1.0, 5.0),
        ("a20", 1.0, 5.0), ("a21", 1.0, 5.0), ("a30", 1.0, 10.0), ("a31", 1.0, 5.0),
        # --- effcai time constant (1) ---
        ("tau_eff", 50.0, 500.0),
    ]
    DEFAULT_PARAMS = {
        # Plasticity (from previous ReLU fit)
        "gamma_d_GB_GluSynapse": 244.27, "gamma_p_GB_GluSynapse": 201.89,
        "a00": 1.38, "a01": 2.65, "a10": 4.07, "a11": 4.20,
        "a20": 3.97, "a21": 1.47, "a30": 3.01, "a31": 2.44,
        "tau_eff": SHAFT_CAI_TAU_REF,
    }

    def unpack_params(self, x):
        gb = {'gamma_d': x[0], 'gamma_p': x[1],
              'a00': x[2], 'a01': x[3], 'a10': x[4], 'a11': x[5],
              'a20': x[6], 'a21': x[7], 'a30': x[8], 'a31': x[9]}
        cicr = {'tau_eff': x[10]}
        return gb, cicr

    def get_init_fn(self):
        def init_fn(cai_first, rho0):
            # Return same number of states to align with cicr_common if needed
            # IP3, Ca_ER, ca_cicr, h_ref, effcai, rho
            return (0.0, 0.4, 0.0, 1.0, 0.0, rho0)
        return init_fn

    def get_debug_sim_fn(self):
        return _debug_sim

    def setup_jax(self, *args, **kwargs):
        """After standard collation, swap 'cai' → 'shaft_cai' so the effcai
        integrator runs on shaft calcium rather than cai_CR."""
        logger.info("GBOnlyModel.setup_jax: collating data and building JAX scan (may take a few minutes for large datasets)…")
        super().setup_jax(*args, **kwargs)
        logger.info("GBOnlyModel.setup_jax: collation done, swapping cai → shaft_cai")
        for proto, cd in self.collated_data.items():
            if cd.get("shaft_cai") is None:
                raise KeyError(
                    f"Protocol '{proto}': 'shaft_cai' not available in collated data. "
                    "Ensure new-format simulation pkls (with shaft_cai) are used.")
            n_pairs = cd["cai"].shape[0]
            logger.info(f"  {proto}: {n_pairs} pairs, cai shape {tuple(cd['cai'].shape)}")
            self.collated_data[proto] = {**cd, "cai": cd["shaft_cai"]}
        logger.info("GBOnlyModel.setup_jax: shaft_cai swap complete — starting JAX JIT compilation (first call may take 2–5 min)…")

    def get_step_factory(self):
        def scan_factory(params, syn_params):
            gb, cicr = params
            c_pre, c_post, is_apical, cai_pre, t_pre, cai_post, t_post = syn_params

            tau_effca = cicr['tau_eff']

            # Scale pre-baked shaft_cai Cpre/Cpost to the current tau_eff.
            # They were computed at SHAFT_CAI_TAU_REF; linear rescaling matches
            # the proportional change in the effcai integrator's steady-state.
            scale = tau_effca / SHAFT_CAI_TAU_REF
            c_pre_s, c_post_s = c_pre * scale, c_post * scale

            theta_d = jnp.where(is_apical, gb['a20']*c_pre_s + gb['a21']*c_post_s, gb['a00']*c_pre_s + gb['a01']*c_post_s)
            theta_p = jnp.maximum(
                jnp.where(is_apical, gb['a30']*c_pre_s + gb['a31']*c_post_s, gb['a10']*c_pre_s + gb['a11']*c_post_s),
                theta_d + 0.01)

            def scan_step(carry, inputs):
                IP3, Ca_ER, ca_cicr, h_ref, effcai, rho = carry
                cai_raw, dt, nves_i = inputs
                
                ca_ext = jnp.maximum(0.0, cai_raw - _CICR_MIN_CA)  # Subtract resting baseline

                # 1. Graupner-Brunel Plasticity Integrator
                decay_eff = jnp.exp(-dt / tau_effca)
                
                effcai_new = effcai * decay_eff + ca_ext * tau_effca * (1.0 - decay_eff)
                
                pot = jnp.where(effcai_new > theta_p, 1.0, 0.0)
                dep = jnp.where(effcai_new > theta_d, 1.0, 0.0)
                
                drho = (-rho*(1.0-rho)*(0.5-rho) + pot*gb['gamma_p']*(1.0-rho) - dep*gb['gamma_d']*rho) / 70000.0
                rho_new = jnp.where(dt > 0, jnp.clip(rho + dt * drho, 0.0, 1.0), rho)
                
                # Return the dummy CICR states and updated GB states
                return (IP3, Ca_ER, ca_cicr, h_ref, effcai_new, rho_new), None
            return scan_step
        return scan_factory

if __name__ == "__main__":
    GBOnlyModel().run()

#!/usr/bin/env python3
"""
Shared JAX-accelerated base class and utilities for CICR parameter fitting.

Provides:
  - Batched tensor collation for protocols
  - Pure functional JAX factories for ODE integration
  - Abstract CICRModel base class with vectorized optimization (DE, CMA-ES, NN)
"""

import os, sys, json, time, pickle, logging, argparse
from functools import partial
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

EXPERIMENTAL_TARGETS = {
    "2Hz_5ms":    0.9886,
    "5Hz_5ms":    1.0161,
    "10Hz_10ms":  1.2013,
    "10Hz_-10ms": 0.7922,
    "50Hz_10ms":  1.06,
    "10Hz_5ms":   1.2038,    #1.2038,0.0644
}

EXPERIMENTAL_ERRORS = {
    "2Hz_5ms":    0.04,
    "5Hz_5ms":    0.0727,
    "10Hz_10ms":  0.0626,
    "10Hz_-10ms": 0.0259,
    "50Hz_10ms":  0.09,
    "10Hz_5ms":   0.0644,   
}

BASE_DIR = "/project/rrg-emuller/dhuruva/plastyfitting"
L5_TRACE_DIR = os.path.join(BASE_DIR, "trace_results/CHINDEMI_PARAMS")
L23_TRACE_DIR = os.path.join(BASE_DIR, "trace_results/L23PC_Chindemi_params")
L5_BASIS_DIR = os.path.join(BASE_DIR, "basis_results")
L23_BASIS_DIR = os.path.join(BASE_DIR, "basis_results_L23PC_L5TTPC")
THRESHOLD_TRACE_DIR = os.path.join(BASE_DIR, "trace_results/CHINDEMI_PARAMS/threshold_traces_out")

PROTOCOL_PATHWAY = {
    "2Hz_5ms": "L5TTPC", "5Hz_5ms": "L5TTPC",
    "10Hz_10ms": "L5TTPC", "10Hz_-10ms": "L5TTPC",
    "10Hz_5ms": "L5TTPC",
    "50Hz_10ms": "L23PC", 
}

@partial(jax.jit, static_argnums=(3, 4))
def compute_effcai_piecewise_linear_jax(cai_trace, t, tau_effca=278.318, min_ca=70e-6, effcai0=0.0):
    dt_trace = jnp.diff(t, prepend=t[0])
    
    def scan_step(effcai, inputs):
        f0, f1, dt = inputs
        dt = jnp.where(dt <= 0, 1e-6, dt) 
        a = (f1 - f0) / dt
        decay = jnp.exp(-dt / tau_effca)
        
        term1 = f0 * tau_effca * (1.0 - decay)
        term2 = a * (tau_effca * dt - (tau_effca**2) * (1.0 - decay))
        
        effcai_new = jnp.where(
            inputs[2] <= 0,
            effcai,
            effcai * decay + term1 + term2
        )
        return effcai_new, effcai_new

    f0_arr = cai_trace[:-1] - min_ca
    f1_arr = cai_trace[1:] - min_ca
    dt_arr = dt_trace[1:]

    _, effcai_history = jax.lax.scan(scan_step, effcai0, (f0_arr, f1_arr, dt_arr))
    effcai_full = jnp.concatenate([jnp.array([effcai0]), effcai_history])
    return effcai_full


def _jax_peak_effcai_zoh(cai_trace, tau_effca, dt, min_ca=70e-6):
    """JAX ZOH peak effcai — exact discrete replica of NEURON effcai_GB ODE.

    deffcai/dt = -effcai/tau + (cai_CR - min_ca)

    ZOH: effcai[n+1] = effcai[n]*exp(-dt/tau) + max(cai-min_ca,0)*tau*(1-exp(-dt/tau))

    Args:
        cai_trace  : 1D JAX array of calcium values
        tau_effca  : time constant (ms), may be a traced JAX scalar
        dt         : fixed timestep (ms)
        min_ca     : resting calcium baseline (mM)
    Returns:
        peak effcai scalar
    """
    decay  = jnp.exp(-dt / tau_effca)
    factor = tau_effca * (1.0 - decay)

    def step(carry, ca):
        eff, peak = carry
        ca_ext  = jnp.maximum(ca - min_ca, 0.0)
        eff_new = eff * decay + ca_ext * factor
        peak_new = jnp.maximum(peak, eff_new)
        return (eff_new, peak_new), None  # no output history — saves ~15000×8B per synapse per vmap

    (_, peak), _ = jax.lax.scan(step, (0.0, 0.0), cai_trace[1:])
    return peak


def _build_dt(t_raw, n_ca_pts):
    t = np.asarray(t_raw, dtype=np.float64)
    if len(t) == n_ca_pts:
        dt = np.diff(t)
        if np.all(dt > 0):
            return dt
    # Infer uniform dt from total time span (e.g. 1501 pts over 1500ms → 1ms)
    inferred_dt = (t[-1] - t[0]) / (n_ca_pts - 1)
    return np.full(n_ca_pts - 1, inferred_dt)


def _build_t_uniform(t_raw, n_ca_pts):
    t = np.asarray(t_raw, dtype=np.float64)
    if len(t) == n_ca_pts and np.all(np.diff(t) > 0):
        return t
    inferred_dt = (t[-1] - t[0]) / (n_ca_pts - 1)
    return np.arange(n_ca_pts, dtype=np.float64) * inferred_dt


def compute_cpre_cpost_zoh(thresh_data, gids, tau_effca, ca_key="cai_CR", min_ca=70e-6):
    """Peak effcai via zero-order-hold integration of fixed-dt cai_CR.

    Replicates NEURON GluSynapse.mod effcai ODE exactly:
        deffcai/dt = -effcai/tau + (cai_CR - min_ca_CR)

    ZOH discrete solution per step dt with constant input u:
        effcai[n+1] = effcai[n]*exp(-dt/tau) + u*tau*(1 - exp(-dt/tau))
    """
    ca_pre  = thresh_data["pre"][ca_key]
    ca_post = thresh_data["post"][ca_key]
    n_pre   = len(next(iter(ca_pre.values())))
    n_post  = len(next(iter(ca_post.values())))
    dt_pre  = _build_dt(thresh_data["pre"]["t"],  n_pre)
    dt_post = _build_dt(thresh_data["post"]["t"], n_post)

    def _peak(cai_arr, dt_arr):
        effcai = 0.0
        peak   = 0.0
        for dt, ca in zip(dt_arr, np.asarray(cai_arr, dtype=np.float64)[1:]):
            decay  = np.exp(-dt / tau_effca)
            ca_ext = ca - min_ca
            if ca_ext > 0:
                effcai = effcai * decay + ca_ext * tau_effca * (1.0 - decay)
            else:
                effcai *= decay
            if effcai > peak:
                peak = effcai
        return peak

    c_pre  = np.array([_peak(ca_pre[g],  dt_pre)  for g in gids])
    c_post = np.array([_peak(ca_post[g], dt_post) for g in gids])
    return c_pre, c_post


def compute_cpre_cpost_analytical(thresh_data, gids, tau_effca, ca_key="cai_CR", min_ca=70e-6):
    """Analytical scaling estimate: C_pre(τ) = c_max · τ / (τ + τ_half).

    τ_half is the half-width (FWHM) of the (cai_CR - min_ca) transient.
    """
    ca_pre  = thresh_data["pre"][ca_key]
    ca_post = thresh_data["post"][ca_key]
    n_pre   = len(next(iter(ca_pre.values())))
    n_post  = len(next(iter(ca_post.values())))
    t_pre   = _build_t_uniform(thresh_data["pre"]["t"],  n_pre)
    t_post  = _build_t_uniform(thresh_data["post"]["t"], n_post)

    def _scale(cai_arr, t_arr):
        ca    = np.maximum(np.asarray(cai_arr, dtype=np.float64) - min_ca, 0.0)
        c_max = float(np.max(ca))
        if c_max <= 0:
            return 0.0
        above    = np.where(ca >= c_max / 2.0)[0]
        tau_half = float(t_arr[above[-1]] - t_arr[above[0]]) if len(above) >= 2 else t_arr[1] - t_arr[0]
        return c_max * tau_effca * tau_half / (tau_effca + tau_half)

    c_pre  = np.array([_scale(ca_pre[g],  t_pre)  for g in gids])
    c_post = np.array([_scale(ca_post[g], t_post) for g in gids])
    return c_pre, c_post


# def compute_cpre_cpost_from_shaft_cai(thresh_data, gids, tau_effca=278.318, ca_key="shaft_cai"):
#     """Compute Cpre and Cpost by running the effcai integrator over shaft_cai
#     traces from a single-pulse threshold simulation.

#     shaft_cai records calcium only in the spine shaft, i.e. the compartment
#     upstream of the synaptic density. Using it (instead of cai_CR) gives a
#     cleaner, CICR-free baseline for the GB thresholds.

#     Args:
#         thresh_data : Loaded threshold pickle dict with top-level keys
#                       'pre' and 'post'.  Each must contain:
#                         - 'shaft_cai': {gid (int/np.int64): ndarray}
#                         - 't'        : 1-D time array (ms)
#         gids        : Ordered sequence of synapse GIDs whose Cpre/Cpost
#                       values are required.
#         tau_effca   : effcai time constant (ms).  Default matches NEURON.

#     Returns:
#         c_pre  : np.ndarray shape (n_syn,)  — peak effcai from pre shaft_cai
#         c_post : np.ndarray shape (n_syn,)  — peak effcai from post shaft_cai

#     Raises:
#         KeyError : if 'pre' or 'post' sections are missing required keys, or
#                    if any GID is absent from shaft_cai dicts.
#     """
#     # --- Validate top-level structure ---
#     for side in ("pre", "post"):
#         if side not in thresh_data:
#             raise KeyError(f"thresh_data missing required key '{side}'")
#         for required in (ca_key, "t"):
#             if required not in thresh_data[side]:
#                 raise KeyError(
#                     f"thresh_data['{side}'] missing required key '{required}'")

#     t_pre_raw  = np.asarray(thresh_data["pre"]["t"],  dtype=np.float64)
#     t_post_raw = np.asarray(thresh_data["post"]["t"], dtype=np.float64)
#     shaft_pre  = thresh_data["pre"][ca_key]
#     shaft_post = thresh_data["post"][ca_key]

#     # shaft_cai / cai_CR may be at fixed dt=0.1ms (15001 pts) while 't' tracks
#     # CVODE steps (863 pts). Reconstruct the correct dt array in either case.
#     def _get_dt(t_raw, n_pts, dt_ms=0.1):
#         if len(t_raw) == n_pts:
#             return np.diff(np.asarray(t_raw, dtype=np.float64))
#         return np.full(n_pts - 1, dt_ms)

#     n_pre  = len(next(iter(shaft_pre.values())))
#     n_post = len(next(iter(shaft_post.values()))    )
#     dt_pre  = _get_dt(t_pre_raw,  n_pre)
#     dt_post = _get_dt(t_post_raw, n_post)

#     def _peak_effcai_np(cai_arr, dt_arr, tau):
#         """Pure-numpy effcai integrator — no JIT, no dispatch overhead."""
#         effcai = 0.0
#         peak   = 0.0
#         for dt, ca in zip(dt_arr, cai_arr[1:]):
#             if dt <= 0:
#                 continue
#             ca_ext = max(0.0, ca - 70e-6)  # 70e-6 is _CICR_MIN_CA
#             decay  = np.exp(-dt / tau)
#             effcai = effcai * decay + ca_ext * tau * (1.0 - decay)
#             if effcai > peak:
#                 peak = effcai
#         return peak

#     c_pre  = np.zeros(len(gids), dtype=np.float64)
#     c_post = np.zeros(len(gids), dtype=np.float64)

#     for i, gid in enumerate(gids):
#         if gid not in shaft_pre:
#             raise KeyError(
#                 f"GID {gid} not found in thresh_data['pre']['shaft_cai']")
#         if gid not in shaft_post:
#             raise KeyError(
#                 f"GID {gid} not found in thresh_data['post']['shaft_cai']")

#         c_pre[i]  = _peak_effcai_np(np.asarray(shaft_pre[gid],  dtype=np.float64), dt_pre,  tau_effca)
#         c_post[i] = _peak_effcai_np(np.asarray(shaft_post[gid], dtype=np.float64), dt_post, tau_effca)

#     return c_pre, c_post

def compute_cpre_cpost_from_shaft_cai(
    thresh_data,
    gids,
    tau_effca: float = 278.318,
    ca_key: str = "cai_CR",          # ← changed default: cai_CR matches GluSynapse.mod
    min_ca: float = 70e-6,           # ← min_ca_CR from GluSynapse.mod (mM)
):
    """Compute Cpre and Cpost by running the effcai integrator over calcium
    traces from a single-pulse threshold simulation.

    Replicates the NEURON GluSynapse.mod effcai_GB ODE exactly:

        effcai_GB' = -effcai_GB / tau_effca_GB + (cai_CR - min_ca_CR)

    The default ca_key is 'cai_CR' (spine-head calcium, post-CICR), which is
    the signal that actually drives effcai_GB in the mod file.  Pass
    ca_key='shaft_cai' only if you deliberately want a CICR-free estimate
    (note: results will then systematically underestimate NEURON values).

    CVODE compatibility
    -------------------
    cai_CR is recorded at fixed dt=0.1 ms (15 001 pts for a 1500 ms window),
    while 't' may be a short CVODE step array (~863 pts).  The function
    detects this mismatch and falls back to a uniform 0.1 ms grid rather than
    mis-pairing CVODE steps with the fixed-dt calcium array.  If 't' and the
    calcium array have the same length (both fixed-dt or both CVODE), the
    actual dt values are used directly — no interpolation, no resampling.

    Args:
        thresh_data : Loaded threshold pickle dict with top-level keys
                      'pre' and 'post'.  Each must contain:
                        - ca_key : {gid (int/np.int64): ndarray}
                        - 't'    : 1-D time array (ms)
        gids        : Ordered sequence of synapse GIDs.
        tau_effca   : effcai time constant (ms).  Default = tau_effca_GB.
        ca_key      : Which calcium signal to integrate.  Default 'cai_CR'.
        min_ca      : Resting calcium baseline (mM).  Default = min_ca_CR.

    Returns:
        c_pre  : np.ndarray shape (n_syn,)  — peak effcai from pre  traces
        c_post : np.ndarray shape (n_syn,)  — peak effcai from post traces

    Raises:
        KeyError : if required keys are missing or any GID is absent.
    """
    # ------------------------------------------------------------------ #
    # 1. Validate structure
    # ------------------------------------------------------------------ #
    for side in ("pre", "post"):
        if side not in thresh_data:
            raise KeyError(f"thresh_data missing required key '{side}'")
        for required in (ca_key, "t"):
            if required not in thresh_data[side]:
                raise KeyError(
                    f"thresh_data['{side}'] missing required key '{required}'"
                )

    # ------------------------------------------------------------------ #
    # 2. Build dt arrays — CVODE-safe
    #
    #    Calcium arrays (cai_CR) are recorded at a fixed dt by NEURON's
    #    Vector.record mechanism (e.g. 1 ms → 1501 pts, or 0.1 ms → 15001).
    #    The 't' array may be a CVODE step log with far fewer points.
    #    When lengths mismatch, infer the uniform dt from the time span and
    #    number of calcium samples rather than assuming a hardcoded value.
    # ------------------------------------------------------------------ #
    def _build_dt(t_raw: np.ndarray, n_ca_pts: int) -> np.ndarray:
        """Return a dt array of length (n_ca_pts - 1).

        If t_raw matches the calcium array length, use actual diffs (handles
        both fixed-dt and CVODE-recorded calcium correctly).  Otherwise infer
        the uniform dt from the time span: dt = (t[-1]-t[0]) / (n_ca_pts-1).
        This correctly handles both 1ms (1501 pts) and 0.1ms (15001 pts)
        recordings without hardcoding a value.
        """
        t = np.asarray(t_raw, dtype=np.float64)
        if len(t) == n_ca_pts:
            dt = np.diff(t)
            if np.all(dt > 0):
                return dt
        # Length mismatch or degenerate steps → infer from span
        inferred_dt = (t[-1] - t[0]) / (n_ca_pts - 1)
        return np.full(n_ca_pts - 1, inferred_dt)

    ca_pre  = thresh_data["pre"][ca_key]
    ca_post = thresh_data["post"][ca_key]

    n_pre  = len(next(iter(ca_pre.values())))
    n_post = len(next(iter(ca_post.values())))

    dt_pre  = _build_dt(thresh_data["pre"]["t"],  n_pre)
    dt_post = _build_dt(thresh_data["post"]["t"], n_post)

    # ------------------------------------------------------------------ #
    # 3. ZOH effcai integrator — exact discrete form of the NEURON ODE
    #
    #    For constant input u over interval dt:
    #        effcai[n+1] = effcai[n] * exp(-dt/tau)
    #                    + u * tau * (1 - exp(-dt/tau))
    #
    #    This is the zero-order-hold (ZOH) solution, equivalent to NEURON's
    #    euler method at the limit of small dt, but exact for any dt.
    # ------------------------------------------------------------------ #
    def _peak_effcai(cai_arr: np.ndarray, dt_arr: np.ndarray, tau: float) -> float:
        effcai = 0.0
        peak   = 0.0
        for dt, ca in zip(dt_arr, cai_arr[1:]):
            if dt <= 0:
                continue
            ca_ext = ca - min_ca
            if ca_ext <= 0:          # below baseline → pure decay, skip exp
                effcai *= np.exp(-dt / tau)
                continue
            decay  = np.exp(-dt / tau)
            effcai = effcai * decay + ca_ext * tau * (1.0 - decay)
            if effcai > peak:
                peak = effcai
        return peak

    # ------------------------------------------------------------------ #
    # 4. Compute per-synapse Cpre / Cpost
    # ------------------------------------------------------------------ #
    c_pre  = np.zeros(len(gids), dtype=np.float64)
    c_post = np.zeros(len(gids), dtype=np.float64)

    for i, gid in enumerate(gids):
        if gid not in ca_pre:
            raise KeyError(
                f"GID {gid} not found in thresh_data['pre']['{ca_key}']"
            )
        if gid not in ca_post:
            raise KeyError(
                f"GID {gid} not found in thresh_data['post']['{ca_key}']"
            )

        c_pre[i]  = _peak_effcai(
            np.asarray(ca_pre[gid],  dtype=np.float64), dt_pre,  tau_effca
        )
        c_post[i] = _peak_effcai(
            np.asarray(ca_post[gid], dtype=np.float64), dt_post, tau_effca
        )

    return c_pre, c_post


# Fixed biophysical constants for CICR
_CICR_MIN_CA = 70e-6    # Resting cytosolic calcium (mM)
_CICR_K_IP3 = 0.001     # IP3R half-activation for IP3 (mM) = 1 µM


def compute_effcai_with_cicr_jax(cai_trace, t, cicr_params, nves, tau_effca=278.318):
    """Compute effcai with priming + coincidence detection CICR, at JAX speed.

    Triple coincidence detector for STDP timing sensitivity:
        J_release = V_CICR * ip3_act * ca_act * P
    where ca_act = ca_ext^2 / (K_ca^2 + ca_ext^2)

    Args:
        cai_trace: Raw test-pulse calcium trace, shape [T].
        t:         Time array, shape [T].
        cicr_params: Dict with keys: delta_IP3, tau_IP3, V_CICR, K_ca,
                     tau_charge, tau_extrusion.
        nves:      Vesicle release events, shape [T] (1.0 at spike times).
        tau_effca: effcai time constant (ms).

    Returns:
        effcai_full: Array of shape [T] with the CICR-boosted effcai trace.
    """
    dt_trace = jnp.diff(t, prepend=t[0])

    delta_IP3 = cicr_params['delta_IP3']
    tau_IP3 = cicr_params['tau_IP3']
    V_CICR = cicr_params['V_CICR']
    K_ca2 = cicr_params['K_ca'] ** 2
    tau_charge = cicr_params['tau_charge']
    tau_ext = cicr_params['tau_extrusion']

    # carry = (IP3, P, ca_cicr, effcai)
    init_carry = (0.0, 0.0, 0.0, 0.0)

    def scan_step(carry, inputs):
        IP3, P, ca_cicr, effcai = carry
        cai_raw, dt, nves_i = inputs

        ca_ext = jnp.maximum(0.0, cai_raw - _CICR_MIN_CA)

        # 1. IP3 dynamics (cnexp decay + bolus)
        IP3_post = IP3 + delta_IP3 * nves_i
        IP3_new = jnp.maximum(0.0, IP3_post * jnp.exp(-dt / tau_IP3))

        # 2. IP3R activation (saturating)
        ip3_act = IP3_new / (IP3_new + _CICR_K_IP3 + 1e-30)

        # 3. Calcium coincidence gate (Hill n=2)
        ca_act = ca_ext**2 / (K_ca2 + ca_ext**2 + 1e-30)

        # 4. Release (triple coincidence: IP3 × Ca × Priming)
        J_release = V_CICR * ip3_act * ca_act * P

        # 5. Priming (charged by VDCC Ca, consumed by release)
        dP = ca_ext * (1.0 - P) / tau_charge - J_release
        P_new = jnp.clip(P + dt * dP, 0.0, 1.0)

        # 6. Cytosolic CICR pool
        dca_cicr = J_release - ca_cicr / tau_ext
        ca_cicr_new = jnp.maximum(0.0, ca_cicr + dt * dca_cicr)

        # 7. effcai integrator
        d_eff = jnp.exp(-dt / tau_effca)
        effcai_new = effcai * d_eff + (ca_ext + ca_cicr_new) * tau_effca * (1.0 - d_eff)

        return (IP3_new, P_new, ca_cicr_new, effcai_new), effcai_new

    _, effcai_history = jax.lax.scan(scan_step, init_carry, (cai_trace[1:], dt_trace[1:], nves[1:]))
    effcai_full = jnp.concatenate([jnp.array([0.0]), effcai_history])
    return effcai_full


def _load_pkl(pkl_path, needs_threshold_traces=True):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    t = np.asarray(data["t"], dtype=np.float64)
    cai_raw = data["cai_CR"]

    # ------------------------------------------------------------------ #
    # Format detection: new format has cai_CR as a dict {gid: ndarray}.  #
    # Old format has cai_CR as a plain ndarray (n_syn, T).               #
    # ------------------------------------------------------------------ #
    if isinstance(cai_raw, dict):
        # --- New format (dict-keyed by GID) ---
        gids = list(cai_raw.keys())
        n_syn = len(gids)
        cai = np.stack([np.asarray(cai_raw[g], dtype=np.float64) for g in gids], axis=0)

        # c_pre / c_post: DO NOT read from data["c_pre"]/["c_post"] here.
        # Those are raw NEURON Cpre/Cpost values (not peak effcai from
        # single-pulse shaft_cai) and will be ~5x too large, giving wrong
        # thresholds. The correct values are computed below from the threshold
        # traces pkl via compute_cpre_cpost_from_shaft_cai.
        c_pre  = np.zeros(n_syn, dtype=np.float64)  # placeholder; overwritten below
        c_post = np.zeros(n_syn, dtype=np.float64)

        # rho0 from rho_GB dict (first time-point of each synapse trace)
        rho_gb_dict = data.get("rho_GB", {})
        rho0 = np.array(
            [float(np.asarray(rho_gb_dict[g])[0]) if g in rho_gb_dict else 0.0
             for g in gids],
            dtype=np.float64,
        )

        # Location from syn_props
        syn_props = data.get("syn_props", {})
        loc_list  = syn_props.get("loc", ["basal"] * n_syn)
        is_apical = np.array([loc == "apical" for loc in loc_list], dtype=bool)

        # shaft_cai — available in new-format pkls
        shaft_raw = data.get("shaft_cai", {})
        if not isinstance(shaft_raw, dict) or len(shaft_raw) == 0:
            raise KeyError(
                f"'shaft_cai' not found or empty in simulation pkl: {pkl_path}")
        shaft_cai = np.stack(
            [np.asarray(shaft_raw[g], dtype=np.float64) for g in gids], axis=0)

    else:
        # --- Old format (ndarray cai_CR) ---
        cai = np.asarray(cai_raw, dtype=np.float64)
        if cai.ndim == 1:
            cai = cai.reshape(1, -1)
        elif cai.shape[0] == len(t) and cai.shape[1] != len(t):
            cai = cai.T
        n_syn = cai.shape[0]
        gids  = list(range(n_syn))

        rho0 = np.zeros(n_syn, dtype=np.float64)
        if "rho_GB" in data:
            rho_gb = np.asarray(data["rho_GB"], dtype=np.float64)
            if rho_gb.ndim == 1:
                rho_gb = rho_gb.reshape(1, -1)
            elif rho_gb.shape[0] == len(t) and rho_gb.shape[1] != len(t):
                rho_gb = rho_gb.T
            rho0 = rho_gb[:, 0].copy()

        synprops  = data.get("synprop", {})
        c_pre     = np.asarray(synprops.get("Cpre",  np.zeros(n_syn)), dtype=np.float64)
        c_post    = np.asarray(synprops.get("Cpost", np.zeros(n_syn)), dtype=np.float64)
        loc_list  = synprops.get("loc", ["basal"] * n_syn)
        is_apical = np.array([loc == "apical" for loc in loc_list], dtype=bool)
        shaft_cai = None  # not available in old format

    # Build nves: 1.0 per prespike, deterministic mean-field approximation.
    nves = np.zeros(len(t), dtype=np.float64)
    prespikes = data.get("prespikes", [])
    if len(prespikes) > 0:
        for spk_t in np.asarray(prespikes, dtype=np.float64):
            idx = np.searchsorted(t, spk_t)
            if idx < len(t):
                nves[idx] = 1.0

    # Load threshold traces
    pair_name = Path(pkl_path).parent.parent.name
    # Highest priority: sibling threshold_traces_out/ inside whichever trace root we loaded from.
    # This makes precomputed directories self-contained: if the caller baked threshold traces
    # alongside the simulation traces, they are found here without any extra arguments.
    local_threshold_path = Path(pkl_path).parent.parent.parent / "threshold_traces_out" / f"{pair_name}_threshold_traces.pkl"
    # Then cpre_cpost_traces (canonical new path)
    cpre_cpost_path = Path(BASE_DIR) / "trace_results/CHINDEMI_PARAMS/cpre_cpost_traces" / f"{pair_name}_threshold_traces.pkl"
    # Then the configured threshold trace dir
    threshold_path = Path(THRESHOLD_TRACE_DIR) / f"{pair_name}_threshold_traces.pkl"
    # Finally legacy fallback
    legacy_threshold_path = Path(BASE_DIR) / "trace_results/cpre_cpost_cai_raw_traces" / f"{pair_name}_threshold_traces.pkl"

    found_path = next(
        (p for p in (local_threshold_path, cpre_cpost_path, threshold_path, legacy_threshold_path)
         if p.exists()),
        None,
    )

    if found_path is not None:
        with open(found_path, "rb") as ft:
            thresh_data = pickle.load(ft)

        # Extract GIDs in order from the pre section
        pre_keys = list(thresh_data["pre"]["cai_CR"].keys())
        gids_ordered = pre_keys[:n_syn]

        # --- c_pre / c_post ---
        # When tau_eff is a fitted parameter (needs_threshold_traces=True) we
        # must recompute Cpre/Cpost at each candidate tau from cai_CR.
        # When tau_eff is fixed the caller has already loaded the correct values
        # from synprop['Cpre'/'Cpost'] (old-format path) which match NEURON.
        if needs_threshold_traces:
            c_pre, c_post = compute_cpre_cpost_from_shaft_cai(
                thresh_data, gids_ordered, ca_key="cai_CR")
            logger.info(
                f"  {Path(pkl_path).parent.name}: c_pre={c_pre}  c_post={c_post} "
                f"(peak effcai from cai_CR threshold traces)")

        # --- c_pre_cr / c_post_cr: mirror of c_pre/c_post ---
        c_pre_cr, c_post_cr = c_pre.copy(), c_post.copy()

        # --- Build cai_pre / cai_post from shaft_cai threshold traces ---
        # Skip if the calling model doesn't use them (saves memory + time).
        if not needs_threshold_traces:
            logger.info(
                f"  {Path(pkl_path).parent.name}: skipping cai_pre/cai_post "
                f"(NEEDS_THRESHOLD_TRACES=False)")
            dummy_trace = np.zeros((n_syn, 1), dtype=np.float64)
            dummy_t     = np.zeros(1, dtype=np.float64)
            return {"cai": cai, "t": t, "c_pre": c_pre, "c_post": c_post,
                    "c_pre_cr": c_pre_cr, "c_post_cr": c_post_cr,
                    "is_apical": is_apical, "rho0": rho0, "nves": nves,
                    "shaft_cai": shaft_cai,
                    "cai_pre": dummy_trace, "t_pre": dummy_t,
                    "cai_post": dummy_trace, "t_post": dummy_t}

        # Always use cai_CR: this is the signal that drives effcai_GB in NEURON
        # and is confirmed to match the ground truth ZOH integration.
        t_pre_raw  = np.asarray(thresh_data["pre"]["t"],  dtype=np.float64)
        t_post_raw = np.asarray(thresh_data["post"]["t"], dtype=np.float64)

        def _match_t_np(t_raw, n_pts):
            if len(t_raw) == n_pts and np.all(np.diff(t_raw) > 0):
                return t_raw
            inferred_dt = (t_raw[-1] - t_raw[0]) / (n_pts - 1)
            return np.arange(n_pts, dtype=np.float64) * inferred_dt

        n_cai_pre  = len(thresh_data["pre"]["cai_CR"][gids_ordered[0]])
        n_cai_post = len(thresh_data["post"]["cai_CR"][gids_ordered[0]])
        t_pre  = _match_t_np(t_pre_raw,  n_cai_pre)
        t_post = _match_t_np(t_post_raw, n_cai_post)
        cai_pre  = np.zeros((n_syn, n_cai_pre),  dtype=np.float64)
        cai_post = np.zeros((n_syn, n_cai_post), dtype=np.float64)
        for idx, s_key in enumerate(gids_ordered):
            cai_pre[idx]  = np.asarray(thresh_data["pre"]["cai_CR"][s_key])
            cai_post[idx] = np.asarray(thresh_data["post"]["cai_CR"][s_key])

        return {"cai": cai, "t": t, "c_pre": c_pre, "c_post": c_post,
                "c_pre_cr": c_pre_cr, "c_post_cr": c_post_cr,
                "is_apical": is_apical, "rho0": rho0, "nves": nves,
                "shaft_cai": shaft_cai,
                "cai_pre": cai_pre, "t_pre": t_pre, "cai_post": cai_post, "t_post": t_post}

    raise FileNotFoundError(
        f"No threshold traces found for pair '{pair_name}'.\n"
        f"  Searched:\n    {cpre_cpost_path}\n    {threshold_path}\n    {legacy_threshold_path}")


def _load_basis(pre_gid, post_gid, basis_dir):
    csv_path = os.path.join(basis_dir, f"basis_{pre_gid}_{post_gid}.csv")
    if not os.path.exists(csv_path): return None
    df = pd.read_csv(csv_path)
    configs = df["config"].apply(lambda x: [int(i) for i in x.split(",")])
    n_syn = len(configs.iloc[0])
    baseline_row = df[configs.apply(lambda x: sum(x) == 0)]
    if baseline_row.empty: return None
    
    baseline_mean = float(baseline_row["mean"].values[0])
    singleton_means = np.zeros(n_syn, dtype=np.float64)
    for i in range(n_syn):
        row = df[configs.apply(lambda x: sum(x) == 1 and x[i] == 1)]
        if row.empty: return None
        singleton_means[i] = row["mean"].values[0]
    return {"baseline_mean": baseline_mean, "singleton_means": singleton_means, "n_syn": n_syn}

def preload_all_data(max_pairs=None, protocols=None, needs_threshold_traces=True,
                     l5_trace_dir=None):
    """Load all protocol data from disk.

    Args:
        l5_trace_dir: Override for L5TTPC trace directory.  When provided,
                      L5 pairs are loaded from this directory instead of the
                      default L5_TRACE_DIR.  Threshold traces are still
                      resolved via BASE_DIR (they are found by pair-name, so
                      they need not live inside l5_trace_dir).
    """
    if protocols is None: protocols = list(EXPERIMENTAL_TARGETS.keys())
    protocol_data = {proto: [] for proto in protocols}

    def _load_dir(trace_dir, basis_dir, protos):
        protos = [p for p in protos if p in protocol_data]
        if not protos or not Path(trace_dir).exists(): return
        dirs = sorted(d for d in Path(trace_dir).iterdir() if d.is_dir())
        n_dirs = len(dirs)
        logger.info(f"Loading from {trace_dir} ({n_dirs} pair dirs, protocols={protos})")
        loaded = 0
        for pair_dir in dirs:
            if max_pairs and all(len(protocol_data[p]) >= max_pairs for p in protos if p in protocol_data): break
            parts = pair_dir.name.split("-")
            if len(parts) != 2: continue
            basis = _load_basis(int(parts[0]), int(parts[1]), basis_dir)
            if not basis: continue
            for proto in protos:
                if max_pairs and len(protocol_data[proto]) >= max_pairs: continue
                pkl_path = pair_dir / proto / "simulation_traces.pkl"
                if pkl_path.exists():
                    try:
                        pd_item = _load_pkl(str(pkl_path), needs_threshold_traces=needs_threshold_traces)
                    except Exception as exc:
                        logger.warning(f"  Skipping {pair_dir.name}/{proto}: {exc}")
                        continue
                    if pd_item["cai"].shape[0] == basis["n_syn"]:
                        pd_item.update(basis)
                        protocol_data[proto].append(pd_item)
                        loaded += 1
                        logger.info(
                            f"  [{loaded}] Loaded {pair_dir.name}/{proto} "
                            f"(n_syn={basis['n_syn']}, T={pd_item['cai'].shape[1]})")

    effective_l5_dir = l5_trace_dir if l5_trace_dir is not None else L5_TRACE_DIR
    _load_dir(effective_l5_dir, L5_BASIS_DIR, [p for p, pw in PROTOCOL_PATHWAY.items() if pw == "L5TTPC"])
    _load_dir(L23_TRACE_DIR, L23_BASIS_DIR, ["50Hz_10ms"])
    return protocol_data

def _interpolate_pair(pair, interp_dt):
    """Interpolate cai traces to uniform grid with interp_dt (ms) spacing."""
    t_old = pair['t']
    t_new = np.arange(t_old[0], t_old[-1], interp_dt)
    n_syn = pair['cai'].shape[0]
    cai_new = np.zeros((n_syn, len(t_new)))
    for s in range(n_syn):
        cai_new[s] = np.interp(t_new, t_old, pair['cai'][s])
    # Re-bin nves: assign each spike event to nearest index in new grid
    nves_old = pair.get('nves', np.zeros(len(t_old)))
    nves_new = np.zeros(len(t_new))
    for i, v in enumerate(nves_old):
        if v > 0:
            idx = int(np.argmin(np.abs(t_new - t_old[i])))
            nves_new[idx] += v
    return {**pair, 'cai': cai_new, 't': t_new, 'nves': nves_new}

def collate_protocol_to_jax(pairs_list, dt_step=1, interp_dt=None):
    n_pairs = len(pairs_list)
    if n_pairs == 0: return None
    if interp_dt is not None:
        pairs_list = [_interpolate_pair(p, interp_dt) for p in pairs_list]
    elif dt_step > 1:
        def _downsample(p, step):
            # Bin-sum nves so spikes between sub-sampled points aren't lost
            nves_full = p.get('nves', np.zeros(p['cai'].shape[1]))
            n = len(nves_full)
            # Pad to a multiple of step
            pad = (-n) % step
            nves_padded = np.pad(nves_full, (0, pad))
            nves_ds = nves_padded.reshape(-1, step).sum(axis=1)
            sc = p.get('shaft_cai')
            sc_ds = sc[:, ::step] if sc is not None else None
            return {**p, 'cai': p['cai'][:, ::step], 't': p['t'][::step], 'nves': nves_ds, 'shaft_cai': sc_ds}
        pairs_list = [_downsample(p, dt_step) for p in pairs_list]
    max_time = max(len(p['t']) for p in pairs_list)
    max_syn = max(p['cai'].shape[0] for p in pairs_list)
    
    cai = np.zeros((n_pairs, max_syn, max_time))
    t = np.zeros((n_pairs, max_time))
    c_pre, c_post, rho0, singletons = (np.zeros((n_pairs, max_syn)) for _ in range(4))
    c_pre_cr, c_post_cr = np.zeros((n_pairs, max_syn)), np.zeros((n_pairs, max_syn))
    is_apical, valid = (np.zeros((n_pairs, max_syn), dtype=bool) for _ in range(2))
    baseline = np.zeros(n_pairs)

    # shaft_cai — only present in new-format pairs; require ALL-or-NONE
    has_shaft = [p.get("shaft_cai") is not None for p in pairs_list]
    if any(has_shaft) and not all(has_shaft):
        raise ValueError(
            "Mixed data: some pairs have 'shaft_cai' (new format) and some do not (old format). "
            "Cannot collate mixed-format batches.")
    shaft_cai = np.zeros((n_pairs, max_syn, max_time)) if all(has_shaft) else None

    # Pre/Post Traces (assuming pre and post traces have max_time limits, or we use their own full lengths)
    max_t_pre = max(p.get('cai_pre', np.zeros((1,1))).shape[1] for p in pairs_list)
    max_t_post = max(p.get('cai_post', np.zeros((1,1))).shape[1] for p in pairs_list)
    
    # We pad the threshold traces
    cai_pre = np.zeros((n_pairs, max_syn, max_t_pre))
    cai_post = np.zeros((n_pairs, max_syn, max_t_post))
    t_pre = np.zeros((n_pairs, max_t_pre))
    t_post = np.zeros((n_pairs, max_t_post))
    nves = np.zeros((n_pairs, max_time))
    
    for i, p in enumerate(pairs_list):
        ns, nt = p['cai'].shape[0], p['cai'].shape[1]
        cai[i, :ns, :nt] = p['cai']
        if nt < max_time: cai[i, :ns, nt:] = p['cai'][:, -1:]
        t[i, :nt] = p['t']
        if nt < max_time: t[i, nt:] = p['t'][-1]
        c_pre[i, :ns], c_post[i, :ns] = p['c_pre'], p['c_post']
        if 'c_pre_cr' in p:
            c_pre_cr[i, :ns], c_post_cr[i, :ns] = p['c_pre_cr'], p['c_post_cr']
        is_apical[i, :ns], rho0[i, :ns] = p['is_apical'], p['rho0']
        baseline[i], singletons[i, :ns], valid[i, :ns] = p['baseline_mean'], p['singleton_means'], True
        nv = p.get('nves', np.zeros(nt))
        nves[i, :nt] = nv[:nt]
        if shaft_cai is not None:
            sc = p['shaft_cai']  # (n_syn, T)
            shaft_cai[i, :ns, :nt] = sc
            if nt < max_time: shaft_cai[i, :ns, nt:] = sc[:, -1:]

        # Load threshold traces if they were extracted successfully
        if 'cai_pre' in p:
            nt_pre = p['cai_pre'].shape[1]
            cai_pre[i, :ns, :nt_pre] = p['cai_pre']
            if nt_pre < max_t_pre: cai_pre[i, :ns, nt_pre:] = p['cai_pre'][:, -1:]
            
            t_pre[i, :nt_pre] = p['t_pre']
            if nt_pre < max_t_pre: t_pre[i, nt_pre:] = p['t_pre'][-1]
            
            nt_post = p['cai_post'].shape[1]
            cai_post[i, :ns, :nt_post] = p['cai_post']
            if nt_post < max_t_post: cai_post[i, :ns, nt_post:] = p['cai_post'][:, -1:]
            
            t_post[i, :nt_post] = p['t_post']
            if nt_post < max_t_post: t_post[i, nt_post:] = p['t_post'][-1]
        
    return {
        'cai': jnp.array(cai), 't': jnp.array(t),
        'c_pre': jnp.array(c_pre), 'c_post': jnp.array(c_post),
        'c_pre_cr': jnp.array(c_pre_cr), 'c_post_cr': jnp.array(c_post_cr),
        'is_apical': jnp.array(is_apical), 'rho0': jnp.array(rho0),
        'baseline': jnp.array(baseline), 'singletons': jnp.array(singletons),
        'valid': jnp.array(valid),
        'nves': jnp.array(nves),
        'shaft_cai': jnp.array(shaft_cai) if shaft_cai is not None else None,
        'cai_pre': jnp.array(cai_pre), 't_pre': jnp.array(t_pre),
        'cai_post': jnp.array(cai_post), 't_post': jnp.array(t_post)
    }

class CICRModel(ABC):
    FIT_PARAMS = []
    DEFAULT_PARAMS = {}
    DESCRIPTION = ""
    NEEDS_THRESHOLD_TRACES = True  # Set False when tau_eff is fixed (skip large cai_pre/cai_post arrays)

    def __init__(self):
        self.PARAM_NAMES = [p[0] for p in self.FIT_PARAMS]
        self.PARAM_BOUNDS = [(p[1], p[2]) for p in self.FIT_PARAMS]
        self.LOWER = np.array([b[0] for b in self.PARAM_BOUNDS])
        self.UPPER = np.array([b[1] for b in self.PARAM_BOUNDS])
        self.DEFAULT_X0 = np.array([self.DEFAULT_PARAMS[n] for n in self.PARAM_NAMES])

    @abstractmethod
    def unpack_params(self, x): ...

    @abstractmethod
    def get_step_factory(self): ...

    def get_debug_sim_fn(self): return None

    def prepare_plot_pair(self, pair_item):
        """Optional hook: transform a raw protocol pair before diagnostic plotting.

        Override in subclasses that swap c_pre/c_post (e.g. to cai_CR-based
        values) so that diagnostic plots use the same thresholds as the JAX
        model.  The default implementation is the identity.
        """
        return pair_item

    def get_init_fn(self):
        def default_init(cai_first, rho0): return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, rho0)
        return default_init

    def setup_jax(self, protocol_data, targets, lambda_reg=0.0, dt_step=1, interp_dt=None, use_loss_v2=False, use_loss_basic=False):
        self.targets_dict = targets
        self.proto_names = list(targets.keys())
        self.target_vals = jnp.array(list(targets.values()))
        self.target_errs = jnp.array([EXPERIMENTAL_ERRORS.get(p, 0.1) for p in self.proto_names])
        
        # Apply specific weights to protocols (e.g. 1.2 for 10Hz_-10ms LTD protocol)
        weights_list = [1.2 if p == "10Hz_-10ms" else 1.0 for p in self.proto_names]
        self.weights = jnp.array(weights_list)
        
        self.lambda_reg = lambda_reg
        
        raw_collated = {p: collate_protocol_to_jax(data, dt_step=dt_step, interp_dt=interp_dt) for p, data in protocol_data.items() if p in targets}
        self.collated_data = {p: d for p, d in raw_collated.items() if d is not None}
        self.proto_names = [p for p in self.proto_names if p in self.collated_data]
        self.target_vals = jnp.array([targets[p] for p in self.proto_names])
        self.weights = jnp.ones(len(self.proto_names))

        # Replace large threshold trace arrays with tiny dummies when not needed
        if not self.NEEDS_THRESHOLD_TRACES:
            for proto, cd in self.collated_data.items():
                n_pairs, max_syn = cd['c_pre'].shape
                self.collated_data[proto] = {
                    **cd,
                    'cai_pre': jnp.zeros((n_pairs, max_syn, 1)),
                    'cai_post': jnp.zeros((n_pairs, max_syn, 1)),
                    't_pre': jnp.zeros((n_pairs, 1)),
                    't_post': jnp.zeros((n_pairs, 1)),
                }

        step_factory = self.get_step_factory()
        init_fn = self.get_init_fn()

        def sim_synapse(cai_trace, t_trace, nves_trace, c_pre, c_post, is_apical, rho0, sm, bmean, valid, cai_pre, t_pre, cai_post, t_post, params):
            dt_trace = jnp.diff(t_trace, prepend=t_trace[0])
            dt_trace = jnp.where(dt_trace <= 0, 1e-6, dt_trace)
            init = init_fn(cai_trace[0], rho0)
            
            # Pack extended syn_params
            syn_params = (c_pre, c_post, is_apical, cai_pre, t_pre, cai_post, t_post)
            scan_fn = step_factory(params, syn_params)
            
            final, _ = jax.lax.scan(scan_fn, init, (cai_trace, dt_trace, nves_trace))
            return jnp.where(valid & (final[-1] >= 0.5), sm - bmean, 0.0)

        # nves_trace is per-pair (not per-synapse), so in_axes has None for the synapse vmap dim
        vmap_syn = jax.vmap(sim_synapse, in_axes=(0, None, None, 0, 0, 0, 0, 0, None, 0, 0, None, 0, None, None))
        
        def sim_pair(cai_p, t_p, nves_p, cpre_p, cpost_p, isapi_p, rho0_p, bmean, sm_p, valid_p, 
                     caipre_p, tpre_p, caipost_p, tpost_p, params):
                     
            contribs = vmap_syn(cai_p, t_p, nves_p, cpre_p, cpost_p, isapi_p, rho0_p, sm_p, bmean, valid_p, 
                                caipre_p, tpre_p, caipost_p, tpost_p, params)
                                
            epsp_after = bmean + jnp.sum(contribs)
            contribs_before = jnp.where(valid_p & (rho0_p >= 0.5), sm_p - bmean, 0.0)
            epsp_before = bmean + jnp.sum(contribs_before)
            return jnp.where(epsp_before > 0, epsp_after / epsp_before, jnp.nan)

        vmap_pair = jax.vmap(sim_pair, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None))
        
        def sim_protocol(proto_data, params):
            ratios = vmap_pair(proto_data['cai'], proto_data['t'], proto_data['nves'],
                               proto_data['c_pre'], proto_data['c_post'], 
                               proto_data['is_apical'], proto_data['rho0'], proto_data['baseline'], 
                               proto_data['singletons'], proto_data['valid'], 
                               proto_data['cai_pre'], proto_data['t_pre'], proto_data['cai_post'], proto_data['t_post'],
                               params)
            return jnp.nanmean(ratios)

        def forward_single(x_array, collated):
            params = self.unpack_params(x_array)
            return jnp.stack([sim_protocol(collated[p], params) for p in self.proto_names])

        @jax.jit
        def forward_batch(x_matrix, collated):
            return jax.vmap(forward_single, in_axes=(0, None))(x_matrix, collated)

        idx_ltp = self.proto_names.index("10Hz_10ms") if "10Hz_10ms" in self.proto_names else -1
        idx_ltd = self.proto_names.index("10Hz_-10ms") if "10Hz_-10ms" in self.proto_names else -1

        @jax.jit
        def objective_single(x_array, collated):
            preds = forward_single(x_array, collated)

            if use_loss_basic:
                # Pure weighted MSE — no sep_penalty, no regularization
                return jnp.sum(self.weights * (preds - self.target_vals) ** 2)

            if use_loss_v2:
                # error = |R_insilico - R_invitro| / Standard_Error(R_invitro)
                base_loss = jnp.sum(self.weights * (jnp.abs(preds - self.target_vals) / self.target_errs))
            else:
                base_loss = jnp.sum(self.weights * (preds - self.target_vals)**2)

            sep_penalty = 0.0
            if idx_ltp >= 0 and idx_ltd >= 0:
                actual_sep = preds[idx_ltp] - preds[idx_ltd]
                sep_penalty = 20.0 * jnp.maximum(0.0, 0.41 - actual_sep)**2  # 20x weight forces LTP/LTD separation
            loss = base_loss + sep_penalty
            if self.lambda_reg > 0: loss += self.lambda_reg * jnp.sum((x_array - self.DEFAULT_X0)**2)
            return loss

        @jax.jit
        def objective_batch(x_matrix, collated):
            return jax.vmap(objective_single, in_axes=(0, None))(x_matrix, collated)

        self.forward_batch = partial(forward_batch, collated=self.collated_data)
        self.objective_single = partial(objective_single, collated=self.collated_data)
        self.objective_batch = partial(objective_batch, collated=self.collated_data)

    def run_de(self, max_iter=1000, seed=42, popsize=15, patience=100, loss_tol=0.001, **kw):
        from scipy.optimize import differential_evolution
        logger.info(f"Running DE Vectorized (maxiter={max_iter}, popsize={popsize}, patience={patience}, loss_tol={loss_tol})")
        def vectorized_obj(x_matrix): return np.array(self.objective_batch(jnp.array(x_matrix.T)))

        stagnation = {"best": np.inf, "count": 0, "gen": 0}
        def _callback(intermediate_result):
            f = intermediate_result.fun
            stagnation["gen"] += 1
            
            # Print progress every 10 generations
            if stagnation["gen"] % 10 == 0:
                logger.info(f"DE gen {stagnation['gen']}: loss={f:.6f} (best={stagnation['best']:.6f}, stagnation={stagnation['count']})")
            
            # Absolute convergence threshold
            if f < loss_tol:
                logger.info(f"Early stopping: Reached target loss ({f:.6f} < {loss_tol})")
                return True
                
            if stagnation["best"] - f > 1e-8:
                stagnation["best"], stagnation["count"] = f, 0
            else:
                stagnation["count"] += 1
                if stagnation["count"] >= patience:
                    logger.info(f"Early stopping: no improvement for {patience} steps.")
                    return True
            return False

        n_pop = popsize * len(self.PARAM_NAMES)
        rng = np.random.default_rng(seed)
        init_pop = self.LOWER + rng.random((n_pop, len(self.PARAM_NAMES))) * (self.UPPER - self.LOWER)
        init_pop[0] = np.clip(self.DEFAULT_X0, self.LOWER, self.UPPER)
        for i, seed_dict in enumerate(getattr(self, 'SEED_PARAMS', []), start=1):
            if i < n_pop:
                x_seed = np.array([seed_dict.get(n, self.DEFAULT_X0[j]) for j, n in enumerate(self.PARAM_NAMES)])
                init_pop[i] = np.clip(x_seed, self.LOWER, self.UPPER)

        result = differential_evolution(vectorized_obj, self.PARAM_BOUNDS, maxiter=max_iter, popsize=popsize,
                                        tol=0, atol=0, seed=seed, disp=True, polish=True, vectorized=True,
                                        callback=_callback)#, init=init_pop)
        return {"method": "differential_evolution", "x": result.x.tolist(), "fun": float(result.fun)}

    def run_cmaes(self, max_iter=1000, seed=42, popsize=15, patience=100, **kw):
        import cma
        logger.info(f"Running CMA-ES (maxiter={max_iter}, popsize={popsize})")
        
        # Scale to bounds strictly
        sigma_0 = np.mean((self.UPPER - self.LOWER) / 4.0)
        options = {
            'bounds': [self.LOWER.tolist(), self.UPPER.tolist()],
            'maxiter': max_iter,
            'popsize': popsize * len(self.PARAM_NAMES),
            'seed': seed,
            'verbose': -1,
            'tolflatfitness': 10000,
            'tolfun': 0,
            'tolfunhist': 0,
            'ftarget': 1e-6
        }
        
        x0 = np.clip(self.DEFAULT_X0, self.LOWER, self.UPPER)
        es = cma.CMAEvolutionStrategy(x0, sigma_0, options)
        
        eval_chunk_size = getattr(self, 'EVAL_CHUNK_SIZE', 4)
        def vectorized_obj(x_list):
            X_mat = np.array(x_list)
            n = len(X_mat)
            results = []
            for start in range(0, n, eval_chunk_size):
                chunk = jnp.array(X_mat[start:start + eval_chunk_size])
                results.append(np.array(self.objective_batch(chunk)))
            return np.concatenate(results).tolist()

        best_f = np.inf
        stag_count = 0
        
        while not es.stop():
            X = es.ask()
            f_vals = vectorized_obj(X)
            es.tell(X, f_vals)
            
            current_best = np.min(f_vals)
            if best_f - current_best > 1e-8:
                best_f = current_best
                stag_count = 0
            else:
                stag_count += 1
                
            es.disp()
            if stag_count >= patience:
                logger.info(f"Early stopping CMA-ES: no improvement for {patience} steps.")
                break
                
        if es.stop():
            logger.info(f"CMA-ES auto-stopped because: {es.stop()}")
            
        res = es.result
        return {"method": "cmaes", "x": res.xbest.tolist(), "fun": float(res.fbest)}
        
    def run_optax(self, max_iter=1000, seed=42, **kw):
        import optax
        logger.info(f"Running Optax AdamW (maxiter={max_iter})")
        
        # Soften objective wrapper
        # We need a pure scalar function of a 1D array
        idx_ltp = self.proto_names.index("10Hz_10ms") if "10Hz_10ms" in self.proto_names else -1
        idx_ltd = self.proto_names.index("10Hz_-10ms") if "10Hz_-10ms" in self.proto_names else -1

        @jax.jit
        def loss_fn(x_unconstrained):
            # Sigmoid map unconstrained x onto bounds
            lower, upper = jnp.array(self.LOWER), jnp.array(self.UPPER)
            x_val = lower + (upper - lower) * jax.nn.sigmoid(x_unconstrained)
            
            # objective_single natively expects a 1D array of length n_params
            return self.objective_single(x_val)

        optimizer = optax.adamw(learning_rate=0.01)
        
        # Inverse sigmoid for initial params
        x0_val = np.clip(self.DEFAULT_X0, self.LOWER + 1e-6, self.UPPER - 1e-6)
        x_init = np.log((x0_val - self.LOWER) / (self.UPPER - x0_val))
        
        params = jnp.array(x_init)
        opt_state = optimizer.init(params)
        
        value_and_grad_fn = jax.value_and_grad(loss_fn)
        
        best_x = params
        best_f = np.inf
        stag_count = 0
        
        for i in range(max_iter):
            loss_val, grads = value_and_grad_fn(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            
            if best_f - loss_val > 1e-8:
                best_f = float(loss_val)
                best_x = params
                stag_count = 0
            else:
                stag_count += 1
                
            if i % 100 == 0:
                logger.info(f"Iter {i}: Loss = {loss_val:.6f}")
                
            if stag_count >= 100:
                logger.info(f"Optax Early Stopping after {i} iterations.")
                break
                
        lower, upper = self.LOWER, self.UPPER
        final_x = lower + (upper - lower) * np.array(jax.nn.sigmoid(best_x))
        return {"method": "optax", "x": final_x.tolist(), "fun": best_f}

    def run_optuna(self, max_iter=1000, seed=42, **kw):
        import optuna
        optuna.logging.set_verbosity(optuna.logging.INFO)
        logger.info(f"Running Optuna TPE (maxiter={max_iter})")
        
        def objective(trial):
            x = []
            for i, name in enumerate(self.PARAM_NAMES):
                x.append(trial.suggest_float(name, self.LOWER[i], self.UPPER[i]))
            loss = float(self.objective_single(jnp.array(x)))
            return loss

        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction='minimize', sampler=sampler)
        
        # Enqueue the default starting guess
        study.enqueue_trial({name: self.DEFAULT_X0[i] for i, name in enumerate(self.PARAM_NAMES)})
        
        study.optimize(objective, n_trials=max_iter)
        
        best = study.best_trial
        best_x = [best.params[name] for name in self.PARAM_NAMES]
        return {"method": "optuna", "x": best_x, "fun": best.value}

    def run_pso(self, max_iter=1000, seed=42, popsize=15, **kw):
        import pyswarms as ps
        logger.info(f"Running PySwarms PSO (maxiter={max_iter}, popsize={popsize})")
        
        options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
        n_particles = popsize * len(self.PARAM_NAMES)
        bounds = (self.LOWER, self.UPPER)
        
        def f_wrapper(x_matrix):
            return np.array(self.objective_batch(jnp.array(x_matrix)))
            
        optimizer = ps.single.GlobalBestPSO(n_particles=n_particles, dimensions=len(self.PARAM_NAMES),
                                            options=options, bounds=bounds)
                                            
        cost, pos = optimizer.optimize(f_wrapper, iters=max_iter)
        return {"method": "pso", "x": pos.tolist(), "fun": float(cost)}

    def print_results(self, opt_result):
        x = np.array(opt_result["x"])
        preds = np.array(self.forward_batch(jnp.array([x])))[0]
        print(f"\n{'='*70}\n  Method: {opt_result['method']}\n  Loss:   {opt_result['fun']:.10f}\n{'='*70}")
        print("\n  Best parameters:")
        for i, name in enumerate(self.PARAM_NAMES):
            d = self.DEFAULT_PARAMS[name]; b = x[i]
            pct = (b - d) / d * 100 if d != 0 else 0
            print(f"    {name:30s} = {b:10.4f}  (default: {d:.4f}, {pct:+.1f}%)")
        print(f"\n  {'Protocol':<15s} {'Predicted':>10s} {'Experiment':>11s} {'Error':>10s}")
        print(f"  {'-'*49}")
        for p, pred, targ in zip(self.proto_names, preds, self.target_vals):
            print(f"  {p:<15s} {pred:10.4f} {targ:11.4f} {pred-targ:+10.4f}")
        print(f"{'='*70}\n")

    def run(self):
        parser = argparse.ArgumentParser(description=f"JAX GB Params ({self.DESCRIPTION})")
        parser.add_argument("--method", choices=["de", "cmaes", "optax", "optuna", "pso"], default="de")
        parser.add_argument("--max-iter", type=int, default=1000)
        parser.add_argument("--protocols", nargs="+", choices=list(EXPERIMENTAL_TARGETS.keys()))
        parser.add_argument("--max-pairs", type=int, default=None)
        parser.add_argument("--dt-step", type=int, default=10, help="Subsample every N CVODE time points (default 10 suits CVODE traces; use --interp-dt for uniform resampling)")
        parser.add_argument("--popsize", type=int, default=15)
        parser.add_argument("--early-stopping", type=int, default=100, help="Patience for early stopping")
        parser.add_argument("--interp-dt", type=float, default=None, help="Interpolate traces to this dt (ms) for CICR accuracy")
        parser.add_argument("--eval", type=str, default=None, help="JSON file or string of params to evaluate instead of fitting")
        parser.add_argument("--loss-v2", action="store_true", help="Use error = |R_insilico - R_invitro| / StdErr")
        parser.add_argument("--loss-basic", action="store_true", help="Pure weighted MSE: sum_p w_p*(pred_p - target_p)^2, no sep_penalty")
        parser.add_argument("--n-plots", type=int, default=1, help="Number of random diagnostic plots to generate")
        parser.add_argument("--verbose", action="store_true", help="Enable INFO-level logging (loading progress etc.)")
        args = parser.parse_args()

        if args.verbose:
            logging.getLogger().setLevel(logging.INFO)

        targets = {p: v for p, v in EXPERIMENTAL_TARGETS.items() if p in args.protocols} if args.protocols else EXPERIMENTAL_TARGETS
        protocol_data = preload_all_data(max_pairs=args.max_pairs, protocols=list(targets.keys()),
                                         needs_threshold_traces=self.NEEDS_THRESHOLD_TRACES)
        self.setup_jax(protocol_data, targets, dt_step=args.dt_step, interp_dt=args.interp_dt, use_loss_v2=args.loss_v2, use_loss_basic=args.loss_basic)
        
        t0 = time.time()
        ts = time.strftime("%Y%m%d_%H%M%S")
        model_tag = self.DESCRIPTION.lower().split("(")[0].strip().replace(" ", "_")

        if args.eval:
            if os.path.isfile(args.eval):
                with open(args.eval) as f: ep = json.load(f)
            else:
                ep = json.loads(args.eval)
            if "best_parameters" in ep:
                ep = {k: (v["value"] if isinstance(v, dict) else v) for k, v in ep["best_parameters"].items()}
            dp = dict(self.DEFAULT_PARAMS)
            dp.update(ep)
            x_eval = np.array([dp[n] for n in self.PARAM_NAMES])
            loss = float(self.objective_single(jnp.array(x_eval)))
            res = {"method": "eval", "x": x_eval.tolist(), "fun": loss, "time": time.time() - t0}
        else:
            if args.method == "cmaes":
                res = self.run_cmaes(max_iter=args.max_iter, popsize=args.popsize, patience=args.early_stopping)
            elif args.method == "optax":
                res = self.run_optax(max_iter=args.max_iter)
            elif args.method == "optuna":
                res = self.run_optuna(max_iter=args.max_iter)
            elif args.method == "pso":
                res = self.run_pso(max_iter=args.max_iter, popsize=args.popsize)
            else:
                res = self.run_de(max_iter=args.max_iter, popsize=args.popsize, patience=args.early_stopping)
            res["time"] = time.time() - t0
            
            out_json = f"best_params_{model_tag}_{ts}.json"
            best_dict = {n: float(v) for n, v in zip(self.PARAM_NAMES, res["x"])}
            with open(out_json, "w") as f:
                json.dump(best_dict, f, indent=4)
            logger.info(f"Saved best parameters to {out_json}")

        self.print_results(res)
        from plot_cicr_diagnostic import pick_random_pair_syn
        
        loaded_protos = [p for p in ["10Hz_10ms", "10Hz_-10ms"] if p in protocol_data]
        
        # Always generate at least the reference plot #84/3
        plot_out = f"cicr_diagnostic_{model_tag}_{args.method}_{ts}_ref.png"
        logger.info(f"Generating diagnostic plot → {plot_out}")
        self.plot_diagnostic_from_results(pair_idx=84, syn_idx=3, output=plot_out, protocols=loaded_protos, _best_x=res["x"], loaded_protocol_data=protocol_data)
        
        # Generate N additional random plots
        for i in range(args.n_plots):
            p_idx, s_idx = pick_random_pair_syn(protocol_data, loaded_protos)
            rand_out = f"cicr_diagnostic_{model_tag}_{args.method}_{ts}_rand{i+1}_p{p_idx}_s{s_idx}.png"
            logger.info(f"Generating random diagnostic plot {i+1}/{args.n_plots} → {rand_out}")
            self.plot_diagnostic_from_results(pair_idx=p_idx, syn_idx=s_idx, output=rand_out, protocols=loaded_protos, _best_x=res["x"], loaded_protocol_data=protocol_data)

    def plot_diagnostic_from_results(self, pair_idx=0, syn_idx=0, output="cicr_diagnostic.png", protocols=("10Hz_10ms", "10Hz_-10ms"), _best_x=None, loaded_protocol_data=None):
        from plot_cicr_diagnostic import compute_single, plot_diagnostic
        dp = dict(self.DEFAULT_PARAMS)
        if _best_x is not None:
            for name, val in zip(self.PARAM_NAMES, _best_x): dp[name] = float(val)
        
        if loaded_protocol_data is not None:
            protocol_data = loaded_protocol_data
        else:
            protocol_data = preload_all_data(protocols=list(protocols))
        results = {}
        for proto in protocols:
            pairs = protocol_data.get(proto, [])
            if not pairs: continue
            
            logger.info(f"  Computing {proto} (single synapse {syn_idx} for plot)…")
            plot_pair = self.prepare_plot_pair(pairs[pair_idx])
            res_single = compute_single(plot_pair, syn_idx, dp, debug_sim_fn=self.get_debug_sim_fn())
            results[proto] = res_single
            
            n_syn = pairs[pair_idx]["cai"].shape[0]
            avg = { 'cai_total': 0, 'cai_raw': 0, 'priming': 0, 'ca_er': 0, 'ca_cicr': 0, 'effcai_no': 0, 'effcai_ci': 0, 'rho_no': 0, 'rho_ci': 0 }
            for s in range(n_syn):
                r = compute_single(plot_pair, s, dp, debug_sim_fn=self.get_debug_sim_fn())
                avg['cai_total'] += r['cai_total'].max()
                avg['cai_raw'] += r['cai_raw'].max()
                avg['priming'] += r['priming'].max()
                avg['ca_er'] += r['ca_er'].max()
                avg['ca_cicr'] += r['ca_cicr'].max()
                avg['effcai_no'] += r['effcai_no'].max()
                avg['effcai_ci'] += r['effcai_ci'].max()
                avg['rho_no'] += r['rho_no'][-1]
                avg['rho_ci'] += r['rho_ci'][-1]

            def f(k): return f"{avg[k]/n_syn:.6f}"
            print(f"\n  --- {proto} (AVERAGES ACROSS {n_syn} SYNAPSES) ---")
            print(f"  cai_raw max:         {f('cai_raw')} mM")
            print(f"  cai_total max:       {f('cai_total')} mM")
            print(f"  Latent State max:    {f('priming')} au")
            
            # Smart Unit Detection
            er_val = avg['ca_er']/n_syn
            if er_val > 0.1:
                print(f"  ER metric max:       {er_val:.6f} au")
            else:
                print(f"  ca_er max:           {er_val:.6f} mM")
                
            print(f"  ca_cicr max:         {f('ca_cicr')} mM")
            print(f"  effcai max no/ci:    {f('effcai_no')} / {f('effcai_ci')}")
            print(f"  rho final no/ci:     {f('rho_no')} / {f('rho_ci')}")

        if results:
            plot_diagnostic(results, protocols=list(results.keys()), pair_idx=pair_idx, syn_idx=syn_idx, param_label=self.DESCRIPTION, output=output)
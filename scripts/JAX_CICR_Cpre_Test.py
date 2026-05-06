#!/usr/bin/env python3
import json
import numpy as np
from cicr_common import _load_pkl
from cicr_er_ip3_simple import SimpleCICRModel, _debug_sim

# V12 Parameters you requested
p_str = '{"enable_CICR_GluSynapse": 1, "gamma_d_GB_GluSynapse": 82.19974298716522, "gamma_p_GB_GluSynapse": 241.7999404447529, "a00": 1.6583210931204957, "a01": 1.5136531069201258, "a10": 3.809033325093047, "a11": 4.5762615641623, "a20": 3.352173921151783, "a21": 3.3765942854214686, "a30": 1.0011444205701536, "a31": 1.2569437734232813, "delta_IP3_CICR_GluSynapse": 0.001386261047598346, "tau_IP3_CICR_GluSynapse": 304.9378710595687, "V_IP3R_CICR_GluSynapse": 0.027532028218439433, "V_RyR_CICR_GluSynapse": 0.002743740615742242, "V_SERCA_CICR_GluSynapse": 0.01329259228286096, "K_SERCA_CICR_GluSynapse": 0.006286133934321784, "V_leak_CICR_GluSynapse": 0.00014020222355700998, "tau_extrusion_CICR_GluSynapse": 34.209295623307625, "tau_effca_GB_GluSynapse": 170.14893667200263, "tau_ref_CICR_GluSynapse": 796.6915676506252, "K_h_ref_CICR_GluSynapse": 0.0007856151840756975}'
p = json.loads(p_str)

# Map MOD variables to Python variables
dp = SimpleCICRModel.DEFAULT_PARAMS.copy()
for k, v in p.items():
    k_clean = k.replace("_CICR_GluSynapse", "").replace("_GB_GluSynapse", "")
    dp[k_clean] = v

# Load the raw cai test pulse traces from 180164-197248
data = _load_pkl("/project/rrg-emuller/dhuruva/plastyfitting/trace_results/Chindemi_params/180164-197248/10Hz_10ms/simulation_traces.pkl")

# Synapse 0 (gid 348477224 equivalent)
cai_pre = data["cai_pre"][0]
t_pre = data["t_pre"]

# 1. Simulate Test 1 equivalent (CICR machinery OFF)
# Just use the raw JAX piecewise integrator baseline, exactly what we modified JAX to do for you!
from cicr_common import compute_effcai_piecewise_linear_jax
import jax.numpy as jnp
eff_no_cicr = compute_effcai_piecewise_linear_jax(jnp.array(cai_pre), jnp.array(t_pre), tau_effca=dp["tau_effca"])
print("--- JAX EXACT INTEGRATION OF RAW CALCIUM TRACE (Base, CICR OFF) ---")
print(f"-> JAX Cpre:  {float(jnp.max(eff_no_cicr)):.6f}")

print("\n--- WHAT HAPPENS IF CICR KICKS IN DURING TEST PULSE (CICR ON) ---")
# 2. Emulate the NEURON test pulse by injecting 1 spike into the Python `_debug_sim`
dp["_nves"] = np.zeros(len(t_pre))
spike_idx = np.argmax(np.diff(cai_pre))  # Test pulse occurs when Ca rises sharply
dp["_nves"][spike_idx] = 1.0  # Fire 1 IP3 bolus

# Run JAX _debug_sim which simulates the full IP3 -> RyR -> CICR release cycle!
res = _debug_sim(cai_pre, t_pre, cp=0.1, cq=0.0001, is_apical=False, dp=dp)

print(f"-> JAX Cpre (with CICR active!): {np.max(res['effcai']):.6f}")
print("   (Notice how triggering CICR blows up the test pulse Cpre to values > 0.9 !)")

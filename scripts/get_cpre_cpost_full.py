#!/usr/bin/env python3
"""Compute Cpre/Cpost using the NumPy _debug_sim (priming CICR model)."""
import json
import numpy as np
import jax.numpy as jnp
from cicr_common import _load_pkl, compute_effcai_piecewise_linear_jax
from cicr_er_ip3_simple import SimpleCICRModel, _debug_sim

p_str = '{"enable_CICR_GluSynapse": 1, "gamma_d_GB_GluSynapse": 82.19974298716522, "gamma_p_GB_GluSynapse": 241.7999404447529, "a00": 1.6583210931204957, "a01": 1.5136531069201258, "a10": 3.809033325093047, "a11": 4.5762615641623, "a20": 3.352173921151783, "a21": 3.3765942854214686, "a30": 1.0011444205701536, "a31": 1.2569437734232813, "delta_IP3_CICR_GluSynapse": 0.001386261047598346, "tau_IP3_CICR_GluSynapse": 304.9378710595687, "V_CICR_CICR_GluSynapse": 0.001, "tau_charge_CICR_GluSynapse": 500.0, "tau_extrusion_CICR_GluSynapse": 34.209295623307625, "tau_effca_GB_GluSynapse": 170.14893667200263}'
p = json.loads(p_str)

dp = SimpleCICRModel.DEFAULT_PARAMS.copy()
for k, v in p.items():
    k_clean = k.replace("_CICR_GluSynapse", "").replace("_GB_GluSynapse", "")
    dp[k_clean] = v

data = _load_pkl("/project/rrg-emuller/dhuruva/plastyfitting/trace_results/Chindemi_params/180164-197248/10Hz_10ms/simulation_traces.pkl")
cai_pre = data["cai_pre"]
t_pre = data["t_pre"]
cai_post = data["cai_post"]
t_post = data["t_post"]
n_syn = len(cai_pre)
base_gid = 348477224

print("--- Priming CICR Model: Cpre/Cpost via _debug_sim ---")

cpre_dict = {}
cpost_dict = {}

for i in range(n_syn):
    # PRE TEST PULSE
    dp["_nves"] = np.zeros(len(t_pre))
    
    peak_ca_pre = np.max(cai_pre[i])
    base_ca_pre = np.min(cai_pre[i])
    if peak_ca_pre - base_ca_pre > 1e-5:
        spike_idx = np.argmax(np.diff(cai_pre[i]))
        dp["_nves"][spike_idx] = 1.0
        
    res_pre = _debug_sim(cai_pre[i], t_pre, cp=0.1, cq=0.0001, is_apical=False, dp=dp)
    cpre_dict[base_gid + i] = float(np.max(res_pre["effcai"]))
    
    # POST TEST PULSE (no presynaptic spike → no IP3 → no CICR)
    dp["_nves"] = np.zeros(len(t_post))
    res_post = _debug_sim(cai_post[i], t_post, cp=0.1, cq=0.0001, is_apical=False, dp=dp)
    cpost_dict[base_gid + i] = float(np.max(res_post["effcai"]))

print(f"-> CICR Cpre (NumPy _debug_sim):  {cpre_dict}")
print(f"-> CICR Cpost (NumPy _debug_sim): {cpost_dict}")

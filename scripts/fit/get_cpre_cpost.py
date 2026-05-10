import json
import numpy as np
import jax.numpy as jnp
from plastyfitting.cicr_common import _load_pkl, compute_effcai_piecewise_linear_jax
from plastyfitting.models.cicr_er_ip3_simple import SimpleCICRModel, _debug_sim

p_str = '{"enable_CICR_GluSynapse": 1, "gamma_d_GB_GluSynapse": 82.19974298716522, "gamma_p_GB_GluSynapse": 241.7999404447529, "a00": 1.6583210931204957, "a01": 1.5136531069201258, "a10": 3.809033325093047, "a11": 4.5762615641623, "a20": 3.352173921151783, "a21": 3.3765942854214686, "a30": 1.0011444205701536, "a31": 1.2569437734232813, "delta_IP3_CICR_GluSynapse": 0.001386261047598346, "tau_IP3_CICR_GluSynapse": 304.9378710595687, "V_IP3R_CICR_GluSynapse": 0.027532028218439433, "V_RyR_CICR_GluSynapse": 0.002743740615742242, "V_SERCA_CICR_GluSynapse": 0.01329259228286096, "K_SERCA_CICR_GluSynapse": 0.006286133934321784, "V_leak_CICR_GluSynapse": 0.00014020222355700998, "tau_extrusion_CICR_GluSynapse": 34.209295623307625, "tau_effca_GB_GluSynapse": 170.14893667200263, "tau_ref_CICR_GluSynapse": 796.6915676506252, "K_h_ref_CICR_GluSynapse": 0.0007856151840756975}'
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

print("--- Test 2: CICR Dynamics (CICR ON - V12 Params) (Simulated with JAX) ---")

cpre_dict = {}
cpost_dict = {}

# Emulate NEURON's test pulse (1 test spike injected) inside JAX _debug_sim
for i in range(n_syn):
    # PRE TEST PULSE
    dp["_nves"] = np.zeros(len(t_pre))
    spike_idx = np.argmax(np.diff(cai_pre[i]))
    dp["_nves"][spike_idx] = 1.0
    res_pre = _debug_sim(cai_pre[i], t_pre, cp=0.1, cq=0.0001, is_apical=False, dp=dp)
    cpre_dict[base_gid + i] = float(np.max(res_pre["effcai"]))
    
    # POST TEST PULSE
    dp["_nves"] = np.zeros(len(t_post))
    spike_idx_post = np.argmax(np.diff(cai_post[i]))
    dp["_nves"][spike_idx_post] = 1.0  # Even post-test pulse triggers 1 spike locally sometimes
    res_post = _debug_sim(cai_post[i], t_post, cp=0.1, cq=0.0001, is_apical=False, dp=dp)
    cpost_dict[base_gid + i] = float(np.max(res_post["effcai"]))

print(f"-> CICR Cpre (JAX full emulation):  {cpre_dict}")
print(f"-> CICR Cpost (JAX full emulation): {cpost_dict}")

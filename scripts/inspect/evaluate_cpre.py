import json
import numpy as np
import jax.numpy as jnp
from plastyfitting.cicr_common import _load_pkl
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
n_syn = len(cai_pre)

for i in range(1): # Just first synapse
    # 1. Run JAX _debug_sim to get ca_cicr_CICR trace over time
    dp["_nves"] = np.zeros(len(t_pre))
    
    # NEURON c_pre_finder creates a massive calcium spike at t=200 by firing presynaptically
    peak_ca_pre = np.max(cai_pre[i])
    base_ca_pre = np.min(cai_pre[i])
    if peak_ca_pre - base_ca_pre > 1e-5:
        spike_idx = np.argmax(np.diff(cai_pre[i]))
        dp["_nves"][spike_idx] = 1.0 # 1 spike triggers 1 IP3 bolus
        
    res_pre = _debug_sim(cai_pre[i], t_pre, cp=0.1, cq=0.0001, is_apical=False, dp=dp)
    
    print(f"JAX _debug_sim raw Cpre (effcai peak natively integrated): {np.max(res_pre['effcai'])}")
    
    # Can we just use `res_pre['effcai']` as the Cpre inside the pipeline if enable_CICR=1?
    # Yes, _debug_sim literally calculates:
    # d_effcai = (-effcai / dp['tau_effca'] + ca_ext + dp['enable_CICR'] * ca_cicr) * dt
    # This IS the term you are asking for.

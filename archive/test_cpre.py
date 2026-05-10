import json
import numpy as np
from cicr_common import _load_pkl
from cicr_er_ip3_simple import SimpleCICRModel, _debug_sim
from prefire_parser import parse_params # wait I don't need this

data = _load_pkl("/project/rrg-emuller/dhuruva/plastyfitting/trace_results/Chindemi_params/180164-197248/10Hz_10ms/simulation_traces.pkl")
cai_pre = data["cai_pre"][0]
t_pre = data["t_pre"]

# User V12 params
p_str = '{"enable_CICR_GluSynapse": 1, "gamma_d_GB_GluSynapse": 82.19974298716522, "gamma_p_GB_GluSynapse": 241.7999404447529, "a00": 1.6583210931204957, "a01": 1.5136531069201258, "a10": 3.809033325093047, "a11": 4.5762615641623, "a20": 3.352173921151783, "a21": 3.3765942854214686, "a30": 1.0011444205701536, "a31": 1.2569437734232813, "delta_IP3_CICR_GluSynapse": 0.001386261047598346, "tau_IP3_CICR_GluSynapse": 304.9378710595687, "V_IP3R_CICR_GluSynapse": 0.027532028218439433, "V_RyR_CICR_GluSynapse": 0.002743740615742242, "V_SERCA_CICR_GluSynapse": 0.01329259228286096, "K_SERCA_CICR_GluSynapse": 0.006286133934321784, "V_leak_CICR_GluSynapse": 0.00014020222355700998, "tau_extrusion_CICR_GluSynapse": 34.209295623307625, "tau_effca_GB_GluSynapse": 170.14893667200263, "tau_ref_CICR_GluSynapse": 796.6915676506252, "K_h_ref_CICR_GluSynapse": 0.0007856151840756975}'
p = json.loads(p_str)

dp = SimpleCICRModel.DEFAULT_PARAMS.copy()
for k, v in p.items():
    k_clean = k.replace("_CICR_GluSynapse", "").replace("_GB_GluSynapse", "")
    dp[k_clean] = v
dp["tau_extrusion"] = dp.get("tau_extrusion", 10.0)

# Simulate what JAX CICR would do
# _nves is needed for IP3 production. A single test pulse (pre or post) is 1 spike.
dp["_nves"] = np.zeros(len(t_pre))
dp["_nves"][np.abs(t_pre - 50.0).argmin()] = 1.0 # assuming the test pulse spike happens at somewhere, wait actually find the peak of cai_pre
spike_idx = np.argmax(np.diff(cai_pre))
dp["_nves"][spike_idx] = 1.0

res = _debug_sim(cai_pre, t_pre, 0, 0, False, dp)
print("max effcai with CICR ON:", np.max(res["effcai"]))


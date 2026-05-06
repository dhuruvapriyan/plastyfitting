import numpy as np
import jax.numpy as jnp
from cicr_er_ip3_simple import SimpleCICRModel, _debug_sim
import json

p_str = '{"enable_CICR_GluSynapse": 1, "gamma_d_GB_GluSynapse": 82.19974298716522, "gamma_p_GB_GluSynapse": 241.7999404447529, "a00": 1.6583210931204957, "a01": 1.5136531069201258, "a10": 3.809033325093047, "a11": 4.5762615641623, "a20": 3.352173921151783, "a21": 3.3765942854214686, "a30": 1.0011444205701536, "a31": 1.2569437734232813, "delta_IP3_CICR_GluSynapse": 0.001386261047598346, "tau_IP3_CICR_GluSynapse": 304.9378710595687, "V_IP3R_CICR_GluSynapse": 0.027532028218439433, "V_RyR_CICR_GluSynapse": 0.002743740615742242, "V_SERCA_CICR_GluSynapse": 0.01329259228286096, "K_SERCA_CICR_GluSynapse": 0.006286133934321784, "V_leak_CICR_GluSynapse": 0.00014020222355700998, "tau_extrusion_CICR_GluSynapse": 34.209295623307625, "tau_effca_GB_GluSynapse": 170.14893667200263, "tau_ref_CICR_GluSynapse": 796.6915676506252, "K_h_ref_CICR_GluSynapse": 0.0007856151840756975}'
p = json.loads(p_str)
dp = SimpleCICRModel.DEFAULT_PARAMS.copy()
for k, v in p.items():
    k_clean = k.replace("_CICR_GluSynapse", "").replace("_GB_GluSynapse", "")
    dp[k_clean] = v

t_rest = np.arange(0, 1000, 0.1)
cai_rest = np.ones_like(t_rest) * 70e-6
dp["_nves"] = np.zeros_like(t_rest)
res_rest = _debug_sim(cai_rest, t_rest, cp=0.1, cq=0.0001, is_apical=False, dp=dp)

import matplotlib.pyplot as plt
plt.figure()
plt.plot(t_rest, res_rest["ca_er"], label="Ca_ER")
plt.plot(t_rest, res_rest["ca_cicr"], label="ca_cicr")
plt.plot(t_rest, res_rest["effcai"], label="effcai")
plt.legend()
plt.savefig("cicr_leak_debug.png")

print(f"Max ca_cicr at rest is {np.max(res_rest['ca_cicr']):.6f} mM.")
print("This explains why Cpost is ~1.0 -- a massive leak dump happens at t=0!")
print("Wait, if J_SERCA is 0 because ca_ext=0, then ER leaks until it's empty, or ca_cicr reaches extreme levels before extrusion catches up!")

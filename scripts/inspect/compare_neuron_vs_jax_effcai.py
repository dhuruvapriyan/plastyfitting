import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plastyfitting.cicr_common import _load_pkl, compute_effcai_piecewise_linear_jax

data = _load_pkl("/project/rrg-emuller/dhuruva/plastyfitting/trace_results/Chindemi_params/180164-197248/10Hz_10ms/simulation_traces.pkl")
cai_pre = data["cai_pre"][0] # Just the first synapse
t_pre = data["t_pre"]

# Wait, data["cai_pre"] ONLY has cai_CR! It does NOT have effcai!
# I need to see what compute_effcai_piecewise_linear_jax generates manually for tau=170.148 (V12).
# IF NEURON generated Cpre=0.938 but JAX generates Cpre=1.101, then NEURON's effcai integral 
# MUST be fundamentally different under the hood for some reason (or NEURON is capping it via the dt ODE solver).
# The mathematical formula is:
# effcai' = -effcai/tau_effca + (cai - MIN_CA) + enable_CICR * ca_cicr

# Wait! "enable_CICR * ca_cicr" was ADDED to effcai in NEURON!
# But in JAX check_cpre_cpost_jax.py, I am using the standard Python compute_effcai_piecewise_linear_jax
# AND passing in the *raw* cai_pre! JAX piecewise DOES NOT magically add "enable_CICR * ca_cicr"! 
# In "check_cpre_cpost_jax.py", it is ONLY integrating raw CAI. It's not running the CICR model!
pass

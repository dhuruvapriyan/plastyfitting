import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

files = [
    "/project/rrg-emuller/dhuruva/plastyfitting/validation/CHINDEMI_PARAMS/180164-197248/10Hz_5ms/simulation_traces.pkl",
    "/project/rrg-emuller/dhuruva/plastyfitting/trace_results/Chindemi_params/180164-197248/10Hz_5ms/simulation_traces.pkl",
    "/project/rrg-emuller/dhuruva/plastyfitting/validation/DHURUVA_PARAMS_V13/180164-197248/10Hz_5ms/simulation_traces.pkl",
    "/project/rrg-emuller/dhuruva/plastyfitting/validation/CHINDEMI_PARAMS_OG/180164-197248/10Hz_5ms/simulation_traces.pkl"
]

labels = ["CHINDEMI (plastyfire)", "Chindemi (plastyfitting)", "V13 (plastyfire)", "CHINDEMI_OG (plastyfire)"]

traces = []
times = []
for i, f in enumerate(files):
    try:
        with open(f, "rb") as pkl:
            data = pickle.load(pkl)
            cai = np.asarray(data["cai_CR"], dtype=np.float64)
            t = np.asarray(data["t"], dtype=np.float64)
            traces.append(cai)
            times.append(t)
            print(f"Loaded {labels[i]} -> shape: {cai.shape}")
    except Exception as e:
        print(f"Failed to load {labels[i]}: {e}")
        traces.append(None)
        times.append(None)

for i in range(len(files)):
    for j in range(i + 1, len(files)):
        t1, t2 = traces[i], traces[j]
        if t1 is not None and t2 is not None:
            # Interpolate to a common time grid to compare continuous traces accurately
            # since CVODE produces different time steps for different models.
            tmin = max(times[i][0], times[j][0])
            tmax = min(times[i][-1], times[j][-1])
            common_t = np.linspace(tmin, tmax, 10000)
            
            diff_maxes = []
            diff_means = []
            
            syn_count = t1.shape[1] if t1.ndim > 1 else 1
            for s in range(syn_count):
                trace1 = t1[:, s] if t1.ndim > 1 else t1
                trace2 = t2[:, s] if t2.ndim > 1 else t2
                
                tr1_interp = np.interp(common_t, times[i], trace1)
                tr2_interp = np.interp(common_t, times[j], trace2)
                
                diff = np.abs(tr1_interp - tr2_interp)
                diff_maxes.append(np.max(diff))
                diff_means.append(np.mean(diff))

            max_diff = np.max(diff_maxes)
            mean_diff = np.mean(diff_means)

            print(f"\nComparing '{labels[i]}' vs '{labels[j]}':")
            print(f"  Max Absolute Difference: {max_diff:.8e}")
            print(f"  Mean Absolute Difference: {mean_diff:.8e}")

# Let's plot synapse 0 for all traces
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

colors = ['blue', 'orange', 'green', 'red']
for i, (tr, tm) in enumerate(zip(traces, times)):
    if tr is not None and tm is not None:
        cai_syn0 = tr[:, 0] if tr.ndim > 1 else tr
        
        # main trace
        ls = ':' if i == 3 else '-'
        ax1.plot(tm, cai_syn0, label=labels[i], color=colors[i], alpha=0.7, linewidth=1.5, linestyle=ls)
        
        # zoomed in trace (roughly first few peaks)
        ax2.plot(tm, cai_syn0, label=labels[i], color=colors[i], alpha=0.7, linewidth=1.5, linestyle=ls)

ax1.set_title("Full cai_CR Trace (Synapse 0)")
ax1.set_xlabel("Time (ms)")
ax1.set_ylabel("Ca Concentration (mM)")
ax1.legend()

ax2.set_title("Zoomed cai_CR Trace (Synapse 0, First Pulse)")
ax2.set_xlim(29000, 30000) # zoom range
ax2.set_xlabel("Time (ms)")
ax2.set_ylabel("Ca Concentration (mM)")

plt.tight_layout()
plt.savefig("/project/rrg-emuller/dhuruva/plastyfitting/cai_comparison.png")
print("\nSaved plot to /project/rrg-emuller/dhuruva/plastyfitting/cai_comparison.png")

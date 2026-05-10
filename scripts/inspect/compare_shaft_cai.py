import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

def load_traces(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded {pkl_path}")
    print(f"  Keys: {list(data.keys())}")
    
    t = data['t']
    shaft_cai = data['shaft_cai']
    # shaft_cai is a dict keyed by gid
    gids = list(shaft_cai.keys())
    print(f"  Found {len(gids)} synapses: {gids}")
    return t, shaft_cai, gids

def compute_cp_cq_from_threshold(thresh_data, gid, tau=278.318):
    """Compute cp and cq as the PEAK effcai from single pre/post threshold
    shaft_cai traces.  This matches cicr_common.compute_cpre_cpost_from_shaft_cai."""
    pre  = thresh_data['pre']
    post = thresh_data['post']

    def _get_t(side, n_pts, dt_ms=0.1):
        t_raw = np.asarray(side['t'])
        return t_raw if len(t_raw) == n_pts else np.arange(n_pts, dtype=np.float64) * dt_ms

    cai_pre  = np.asarray(pre['shaft_cai'][gid],  dtype=np.float64)
    cai_post = np.asarray(post['shaft_cai'][gid], dtype=np.float64)
    t_pre  = _get_t(pre,  len(cai_pre))
    t_post = _get_t(post, len(cai_post))

    cp = float(np.max(compute_effcai(cai_pre,  t_pre,  tau=tau)))
    cq = float(np.max(compute_effcai(cai_post, t_post, tau=tau)))
    return cp, cq

def compute_effcai(cai_arr, t_arr, tau=278.318):
    '''Pure numpy effcai integrator matching JAX/GBOnlyModel implementation'''
    effcai = np.zeros_like(cai_arr)
    dt_arr = np.diff(t_arr, prepend=t_arr[0])
    
    current_effcai = 0.0
    for i in range(len(cai_arr)):
        dt = dt_arr[i]
        if dt <= 0:
            effcai[i] = current_effcai
            continue
        ca_ext = max(0.0, cai_arr[i] - 70e-6)  # 70e-6 is _CICR_MIN_CA
        decay = np.exp(-dt / tau)
        current_effcai = current_effcai * decay + ca_ext * tau * (1.0 - decay)
        effcai[i] = current_effcai
        
    return effcai

THRESHOLD_TRACES_DIR = "/project/rrg-emuller/dhuruva/plastyfitting/trace_results/CHINDEMI_PARAMS/threshold_traces_out"

def load_threshold_traces(pair_name):
    """Load the pre/post shaft_cai threshold traces for a given pair."""
    path = os.path.join(THRESHOLD_TRACES_DIR, f"{pair_name}_threshold_traces.pkl")
    with open(path, 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded threshold traces: {path}")
    print(f"  Keys: {list(data.keys())}")
    return data

def plot_threshold_traces(thresh_data, gid, out_file):
    """Plot shaft_cai and effcai for the pre-only and post-only single-pulse simulations."""
    pre  = thresh_data['pre']
    post = thresh_data['post']

    # Time arrays — may be CVODE-adaptive; match length to shaft_cai if needed
    def _get_t(side, n_pts, dt_ms=0.1):
        t_raw = np.asarray(side['t'])
        if len(t_raw) == n_pts:
            return t_raw
        return np.arange(n_pts, dtype=np.float64) * dt_ms

    cai_pre  = np.asarray(pre['shaft_cai'][gid],  dtype=np.float64)
    cai_post = np.asarray(post['shaft_cai'][gid], dtype=np.float64)
    t_pre  = _get_t(pre,  len(cai_pre))
    t_post = _get_t(post, len(cai_post))

    effcai_pre  = compute_effcai(cai_pre,  t_pre)
    effcai_post = compute_effcai(cai_post, t_post)

    cp = np.max(effcai_pre)
    cq = np.max(effcai_post)
    print(f"  Threshold traces — peak effcai: cp={cp:.5f}  cq={cq:.5f}")

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Panel 1: raw shaft_cai
    axes[0].plot(t_pre,  cai_pre,  label='Pre-only spike',  color='purple')
    axes[0].plot(t_post, cai_post, label='Post-only spike', color='darkorange', linestyle='dashed')
    axes[0].set_title(f'Shaft [Ca²⁺] — single-pulse threshold traces (GID: {gid})')
    axes[0].set_ylabel('Shaft [Ca²⁺] (mM)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Panel 2: effcai with cp/cq annotations
    axes[1].plot(t_pre,  effcai_pre,  label=f'Pre effcai  (peak cp={cp:.4f})',  color='purple')
    axes[1].plot(t_post, effcai_post, label=f'Post effcai (peak cq={cq:.4f})', color='darkorange', linestyle='dashed')
    axes[1].axhline(cp, color='purple',     linestyle=':', linewidth=1.5, label=f'cp (peak) = {cp:.4f}')
    axes[1].axhline(cq, color='darkorange', linestyle=':', linewidth=1.5, label=f'cq (peak) = {cq:.4f}')
    axes[1].set_title('Effective Calcium from single-pulse threshold simulations (tau=278.3ms)')
    axes[1].set_xlabel('Time (ms)')
    axes[1].set_ylabel('effcai (au)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    print(f"Threshold trace plot saved to {out_file}")
    plt.close(fig)

def main():
    base_dir = "/project/rrg-emuller/dhuruva/plastyfitting/trace_results/CHINDEMI_PARAMS/180164-197248"
    ltp_path = os.path.join(base_dir, "10Hz_10ms/simulation_traces.pkl")
    ltd_path = os.path.join(base_dir, "10Hz_-10ms/simulation_traces.pkl")
    
    t_ltp, shaft_ltp, gids_ltp = load_traces(ltp_path)
    t_ltd, shaft_ltd, gids_ltd = load_traces(ltd_path)

    # Load threshold traces once — used for both cp/cq and the separate plot
    thresh_data = load_threshold_traces("180164-197248")

    common_gids = set(gids_ltp).intersection(gids_ltd)
    if not common_gids:
        print("No common GIDs found between LTP and LTD traces!")
        return
        
    gid = list(common_gids)[0]
    print(f"\nPlotting traces for GID: {gid}")
    
    cai_ltp = np.array(shaft_ltp[gid])
    cai_ltd = np.array(shaft_ltd[gid])
    
    effcai_ltp = compute_effcai(cai_ltp, t_ltp)
    effcai_ltd = compute_effcai(cai_ltd, t_ltd)

    # Compute cp/cq as PEAK effcai from single-pulse threshold shaft_cai traces.
    # NOTE: do NOT use data['c_pre']/data['c_post'] from the simulation pkl —
    # those are raw NEURON Cpre/Cpost values (not peak effcai) and will give
    # incorrectly large thresholds.
    thresh_gid = list(thresh_data['pre']['shaft_cai'].keys())[0]
    cp, cq = compute_cp_cq_from_threshold(thresh_data, thresh_gid)

    # Thresholds with all a-coefficients = 1:
    # theta_d = a00*cp + a01*cq  =>  cp + cq
    # theta_p = max(a10*cp + a11*cq, theta_d + 0.01)  =>  theta_d + 0.01
    theta_d = cp + cq
    theta_p = max(cp + cq, theta_d + 0.01)
    print(f"  cp={cp:.5f}  cq={cq:.5f}  theta_d={theta_d:.5f}  theta_p={theta_p:.5f}")
    
    # Removed sharex=True so the zoom window works independently
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    
    # Plot 1: Full course shaft_cai
    axes[0].plot(t_ltp, cai_ltp, label='LTP (+10ms)', alpha=0.8, color='blue')
    axes[0].plot(t_ltd, cai_ltd, label='LTD (-10ms)', alpha=0.8, color='red', linestyle='dashed')
    axes[0].set_title(f'Raw Shaft Calcium (Full Course) - Pair: 180164-197248, GID: {gid}')
    axes[0].set_ylabel('Shaft [Ca2+] (mM)')
    axes[0].set_xlim(0, 52000)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Zoomed in on the first pulse
    early_mask = t_ltp < 2000
    if not np.any(early_mask):
        peak_t = 0
    else:
        # Find where calcium first exceeds resting level significantly
        resting_ca = np.min(cai_ltp)
        threshold = resting_ca + (np.max(cai_ltp) - resting_ca) * 0.1
        first_pulse_idx = np.argmax(cai_ltp > threshold)
        peak_t = t_ltp[first_pulse_idx]
    
    zoom_start = max(0, peak_t - 20)
    zoom_end = peak_t + 50
    
    mask_ltp = (t_ltp >= zoom_start) & (t_ltp <= zoom_end)
    mask_ltd = (t_ltd >= zoom_start) & (t_ltd <= zoom_end)
    
    axes[1].plot(t_ltp[mask_ltp], cai_ltp[mask_ltp], label='LTP (+10ms)', marker='.', color='blue')
    axes[1].plot(t_ltd[mask_ltd], cai_ltd[mask_ltd], label='LTD (-10ms)', marker='x', color='red')
    axes[1].set_title(f'Raw Shaft Calcium Zoom on First Pulse (Starts around {peak_t:.1f}ms)')
    axes[1].set_ylabel('Shaft [Ca2+] (mM)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Full course effcai
    axes[2].plot(t_ltp, effcai_ltp, label='LTP (+10ms)', alpha=0.8, color='blue')
    axes[2].plot(t_ltd, effcai_ltd, label='LTD (-10ms)', alpha=0.8, color='red', linestyle='dashed')
    # Threshold lines (all a=1: theta_d = cp+cq, theta_p = theta_d+0.01)
    axes[2].axhline(theta_d, color='orange', linestyle=':', linewidth=1.8,
                    label=f'θ_d = cp+cq = {theta_d:.4f}')
    axes[2].axhline(theta_p, color='green',  linestyle=':', linewidth=1.8,
                    label=f'θ_p = θ_d+0.01 = {theta_p:.4f}')
    axes[2].set_title(f'Effective Calcium (Integrated from Shaft with tau=278.3) — all a=1')
    axes[2].set_xlabel('Time (ms)')
    axes[2].set_ylabel('effcai (au)')
    axes[2].set_xlim(0, 52000)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_file = "/project/rrg-emuller/dhuruva/plastyfitting/scripts/compare_shaft_effcai_ltp_ltd.png"
    plt.savefig(out_file, dpi=150)
    print(f"\nPlot saved to {out_file}")
    plt.close(fig)

    # --- Separate plot: pre/post threshold traces (reuse already-loaded thresh_data) ---
    print(f"Using threshold trace GID: {thresh_gid}")
    thresh_out = "/project/rrg-emuller/dhuruva/plastyfitting/scripts/compare_shaft_threshold_pre_post.png"
    plot_threshold_traces(thresh_data, thresh_gid, thresh_out)

if __name__ == "__main__":
    main()

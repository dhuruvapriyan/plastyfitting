#!/usr/bin/env python3
"""Diagnostic plotting for CICR V5+IP3R. Passes Nves_CICR to debug sim."""
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
    from numba import njit as _njit
except ImportError:
    def _njit(*a, **kw):
        def _wrap(fn): return fn
        return _wrap

@_njit(cache=True)
def _compute_effcai(cai, t, tau_effca, min_ca):
    n = len(cai); effcai = np.zeros(n); e = 0.0
    for i in range(n-1):
        dt = t[i+1]-t[i]; decay = np.exp(-dt/tau_effca)
        e = e*decay + (cai[i]-min_ca)*tau_effca*(1.0-decay)
        if e < 0.0: e = 0.0
        effcai[i+1] = e
    return effcai

@_njit(cache=True)
def _compute_rho(effcai, t, theta_d, theta_p, gamma_d, gamma_p, rho0):
    n = len(t); rho = np.zeros(n); rho[0] = rho0; inv_tau = 1.0/70000.0; r = rho0
    for i in range(n-1):
        dt = t[i+1]-t[i]
        pot = 1.0 if effcai[i] > theta_p else 0.0
        dep = 1.0 if (effcai[i] > theta_d and pot == 0.0) else 0.0
        drho = (-r*(1.0-r)*(0.5-r) + pot*gamma_p*(1.0-r) - dep*gamma_d*r) * inv_tau
        r = min(1.0, max(0.0, r + dt*drho)); rho[i+1] = r
    return rho

def params_from_json(json_path, param_names=None, default_params=None):
    with open(json_path) as f: data = json.load(f)
    dp = dict(default_params) if default_params else {}
    if "best_parameters" in data:
        for name, entry in data["best_parameters"].items():
            dp[name] = entry["value"] if isinstance(entry, dict) else float(entry)
        return dp
    raise ValueError(f"No best_parameters in {json_path}")

def pick_random_pair_syn(protocol_data, protocols, seed=None):
    rng = np.random.default_rng(seed); ref = protocol_data[protocols[0]]
    valid = [i for i in range(len(ref)) if all(i < len(protocol_data[p]) for p in protocols)]
    pi = int(rng.choice(valid)); peaks = ref[pi]["cai"].max(axis=1)
    good = np.where(peaks > 0.001)[0]
    si = int(rng.choice(good)) if len(good) > 0 else int(rng.choice(np.arange(len(peaks))))
    return pi, si

def compute_single(pd_item, syn_idx, dp, debug_sim_fn=None):
    s = syn_idx; t = pd_item["t"]
    cp, cq, is_apical = pd_item["c_pre"][s], pd_item["c_post"][s], pd_item["is_apical"][s]
    gamma_d = float(dp.get("gamma_d", dp.get("gamma_d_GB_GluSynapse", 150.0)))
    gamma_p = float(dp.get("gamma_p", dp.get("gamma_p_GB_GluSynapse", 200.0)))
    tau_eff = float(dp.get("tau_eff", 278.318))
    # Scale c_pre/c_post by tau_eff ratio (matches JAX model behaviour).
    # The baked values were computed at tau_eff_baked=278.318 ms.
    tau_scale = tau_eff / 278.318
    cp_s, cq_s = cp * tau_scale, cq * tau_scale
    if is_apical: theta_d=dp["a20"]*cp_s+dp["a21"]*cq_s; theta_p=dp["a30"]*cp_s+dp["a31"]*cq_s
    else: theta_d=dp["a00"]*cp_s+dp["a01"]*cq_s; theta_p=dp["a10"]*cp_s+dp["a11"]*cq_s
    cai_raw = pd_item["cai"][s].copy()
    # nves is per-pair (presynaptic spikes), not per-synapse
    nves_syn = pd_item["nves"].copy() if "nves" in pd_item else np.zeros(len(t))

    if debug_sim_fn is not None:
        dp_local = dict(dp); dp_local["rho0"] = pd_item["rho0"][s]; dp_local["_nves"] = nves_syn
        sim_out = debug_sim_fn(cai_raw, t, cp, cq, is_apical, dp_local)
        cai_total = sim_out["cai_total"]
        priming = sim_out["priming"]
        ca_er = sim_out.get("ca_er", sim_out.get("P", np.zeros(len(t))))
        ca_cicr = sim_out["ca_cicr"]
    else:
        cai_total = cai_raw.copy(); priming = np.zeros(len(t)); ca_er = np.zeros(len(t)); ca_cicr = np.zeros(len(t))
        sim_out = {}
    MIN_CA = 70e-6
    effcai_no = _compute_effcai(cai_raw, t, tau_eff, MIN_CA)
    effcai_ci = sim_out.get("effcai", _compute_effcai(cai_total, t, tau_eff, MIN_CA))
    rho_no = _compute_rho(effcai_no, t, theta_d, theta_p, gamma_d, gamma_p, pd_item["rho0"][s])
    rho_ci = sim_out.get("rho", _compute_rho(effcai_ci, t, theta_d, theta_p, gamma_d, gamma_p, pd_item["rho0"][s]))
    h_ref = sim_out.get("h_ref", np.ones(len(t)))
    return {"t": t, "cai_raw": cai_raw, "cai_total": cai_total,
            "effcai_no": effcai_no, "effcai_ci": effcai_ci,
            "rho_no": rho_no, "rho_ci": rho_ci,
            "priming": priming, "ca_er": ca_er, "ca_cicr": ca_cicr,
            "h_ref": h_ref,
            "theta_d": theta_d, "theta_p": theta_p, "cp": cp, "cq": cq}

_PC = [dict(line='#d62728',mGluR='#9467bd',er='#17becf',cicr_ca='#d62728'),
       dict(line='#e377c2',mGluR='#8c564b',er='#bcbd22',cicr_ca='#e377c2'),
       dict(line='#8c564b',mGluR='#7f7f7f',er='#aec7e8',cicr_ca='#8c564b')]

def plot_diagnostic(results, protocols=None, pair_idx=0, syn_idx=0, param_label="", output="cicr_diagnostic.png"):
    if protocols is None: protocols = list(results.keys())
    protocols = [p for p in protocols if p in results]
    if not protocols: return
    plt.rcParams.update({'font.size':14,'axes.titlesize':16,'axes.labelsize':14,'figure.dpi':150})

    # Detect GB-only mode: priming is all-zero across all protocols → skip priming panel
    has_priming = any(np.any(results[p]["priming"] != 0) for p in protocols)
    # Detect abstract ER (P > 0.1 units, i.e. dimensionless priming state)
    has_abstract_er = any(np.max(results[p]["ca_er"]) > 0.1 for p in protocols)
    has_cicr = any(np.any(results[p]["ca_cicr"] != 0) for p in protocols)

    n_panels = 5 if (has_priming or has_abstract_er or has_cicr) else 4
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 3.4*n_panels), sharex=True)
    ls_list = ['-','--',':']
    t_s0 = results[protocols[0]]["t"]/1000.0; cai0 = results[protocols[0]]["cai_raw"]
    above = cai0 > cai0[0]*2.0
    xlim = float(min(t_s0[np.where(above)[0][-1]]*1.5, t_s0[-1])) if np.any(above) else float(t_s0[-1])
    ax4t = axes[4].twinx() if (n_panels == 5 and has_abstract_er) else None
    for i, proto in enumerate(protocols):
        d = results[proto]; t_s = d["t"]/1000.0; ls = ls_list[i%3]; pc = _PC[i%3]; sfx = f" ({proto})"
        axes[0].plot(t_s, d["cai_total"]*1000, color=pc['line'], ls=ls, alpha=0.85, label=proto)
        axes[1].plot(t_s, d["effcai_ci"], color=pc['line'], ls=ls, alpha=0.85, label=proto)
        axes[2].plot(t_s, d["rho_ci"], color=pc['line'], ls=ls, alpha=0.9, lw=1.5, label=proto)
        if n_panels == 5:
            axes[3].plot(t_s, d["priming"], color=pc['mGluR'], ls=ls, alpha=0.9, label='P (Priming)'+sfx)
            if has_abstract_er: ax4t.plot(t_s, d["ca_er"], color=pc['er'], ls=ls, alpha=0.9, label='P'+sfx)
            else: axes[4].plot(t_s, d["ca_er"]*1000, color=pc['er'], ls=ls, alpha=0.9, label='ER'+sfx)
            axes[4].plot(t_s, d["ca_cicr"]*1000, color=pc['cicr_ca'], ls=ls, alpha=0.9, label='CICR'+sfx)
        else:
            # GB-only: panel 3 = rho without CICR (reference)
            axes[3].plot(t_s, d["rho_no"], color=pc['line'], ls=':', alpha=0.6, lw=1.2, label='ρ (no CICR)'+sfx)
        if i == 0:
            axes[0].set_ylabel(r'Ca$^{2+}$ (µM)'); axes[1].set_ylabel('effcai'); axes[2].set_ylabel(r'$\rho$ (with model)')
            td_val = d["theta_d"]; tp_val = d["theta_p"]
            axes[1].axhline(td_val, color='#DD8452', ls=':', lw=1.5, label=f'θd={td_val:.4f}')
            axes[1].axhline(tp_val, color='#55A868', ls=':', lw=1.5, label=f'θp={tp_val:.4f}')
            axes[2].axhline(0.5, color='gray', ls=':', alpha=0.5); axes[2].set_ylim(-0.05,1.05)
            if n_panels == 5:
                axes[3].set_ylabel('P (Priming)'); axes[4].set_ylabel('Ca$^{2+}$ (mM)'); axes[4].set_xlabel('Time (s)')
                if has_abstract_er: ax4t.set_ylabel('P (ER)')
            else:
                axes[3].set_ylabel(r'$\rho$ (no CICR baseline)'); axes[3].set_xlabel('Time (s)')
                axes[3].axhline(0.5, color='gray', ls=':', alpha=0.5); axes[3].set_ylim(-0.05,1.05)
    for ax in axes: ax.set_xlim(0, xlim); ax.legend(loc='upper right', fontsize=8, ncol=2)
    if n_panels == 5 and has_abstract_er:
        l1,la1 = axes[4].get_legend_handles_labels(); l2,la2 = ax4t.get_legend_handles_labels()
        axes[4].legend(l1+l2, la1+la2, loc='upper right', fontsize=8, ncol=2)
    plt.tight_layout(h_pad=0.8)
    fig.suptitle(f'{param_label} | Pair#{pair_idx} Syn#{syn_idx}', fontsize=16, fontweight='bold', y=1.01)
    plt.savefig(output, dpi=200, bbox_inches='tight', facecolor='white'); print(f"Saved → {output}"); plt.close(fig)

def print_stats(label, d): pass
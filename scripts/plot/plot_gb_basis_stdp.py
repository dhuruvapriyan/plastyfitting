#!/usr/bin/env python3
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from plastyfitting.cicr_common import EXPERIMENTAL_ERRORS, EXPERIMENTAL_TARGETS, L5_BASIS_DIR, _load_basis, _load_pkl
from plastyfitting.models.gb_only import GBOnlyModel

MIN_CA = 70e-6
TAU_RHO = 70000.0


def parse_delay_ms(proto_name):
    return int(proto_name.split('_')[1].replace('ms', ''))


def discover_protocols(trace_dir, freq_hz):
    prefix = f"{freq_hz}Hz_"
    protos = set()
    trace_path = Path(trace_dir)
    if not trace_path.exists():
        return []
    for pair_dir in trace_path.iterdir():
        if not pair_dir.is_dir() or '-' not in pair_dir.name:
            continue
        for proto_dir in pair_dir.iterdir():
            if proto_dir.is_dir() and proto_dir.name.startswith(prefix):
                protos.add(proto_dir.name)
    return sorted(protos, key=parse_delay_ms)

def load_protocol_data(trace_dir, protocols, max_pairs=None, basis_dir=L5_BASIS_DIR):
    protocol_data = {proto: [] for proto in protocols}
    trace_path = Path(trace_dir)
    loaded = {proto: 0 for proto in protocols}
    seen_pair_dirs = 0

    print(f"Loading traces from {trace_dir}", flush=True)
    print(f"Loading basis from  {basis_dir}", flush=True)

    for pair_dir in sorted(trace_path.iterdir()):
        if not pair_dir.is_dir() or '-' not in pair_dir.name:
            continue
        if max_pairs and all(loaded[p] >= max_pairs for p in protocols):
            break

        seen_pair_dirs += 1
        if seen_pair_dirs % 25 == 0:
            counts = ", ".join(f"{p}={loaded[p]}" for p in sorted(protocols, key=parse_delay_ms))
            print(f"  scanned {seen_pair_dirs} pair dirs; loaded: {counts}", flush=True)

        parts = pair_dir.name.split('-')
        basis = _load_basis(int(parts[0]), int(parts[1]), basis_dir)
        if not basis:
            continue

        for proto in protocols:
            if max_pairs and loaded[proto] >= max_pairs:
                continue
            pkl_path = pair_dir / proto / 'simulation_traces.pkl'
            if not pkl_path.exists():
                continue
            try:
                pair = _load_pkl(str(pkl_path), needs_threshold_traces=True)
            except Exception as exc:
                print(f"  {pair_dir.name}/{proto}: skipped due to {exc}", flush=True)
                continue
            if pair['cai'].shape[0] != basis['n_syn']:
                print(f"  {pair_dir.name}/{proto}: skipped due to n_syn mismatch", flush=True)
                continue
            pair.update(basis)
            pair['pair_name'] = pair_dir.name
            protocol_data[proto].append(pair)
            loaded[proto] += 1

    print("Finished loading protocol data:", flush=True)
    for proto in sorted(protocols, key=parse_delay_ms):
        print(f"  {proto:<12s} n_pairs={len(protocol_data[proto])}", flush=True)

    return protocol_data


def load_param_dict(json_path=None, overrides=None):
    params = dict(GBOnlyModel.DEFAULT_PARAMS)
    if json_path:
        with open(json_path, 'r', encoding='utf-8') as handle:
            data = json.load(handle)
        if 'best_parameters' in data:
            data = {
                key: (value['value'] if isinstance(value, dict) else value)
                for key, value in data['best_parameters'].items()
            }
        for key, value in data.items():
            if key in params:
                params[key] = float(value)
    for item in overrides or []:
        if '=' not in item:
            raise ValueError(f"Invalid override '{item}', expected KEY=VALUE")
        key, value = item.split('=', 1)
        if key not in params:
            raise KeyError(f"Unknown parameter '{key}'")
        params[key] = float(value)
    return params


def downsample_pair(pair, step):
    if step <= 1:
        return pair
    nves_full = np.asarray(pair.get('nves', np.zeros(pair['cai'].shape[1])), dtype=np.float64)
    pad = (-len(nves_full)) % step
    nves_padded = np.pad(nves_full, (0, pad))
    nves_ds = nves_padded.reshape(-1, step).sum(axis=1)
    return {
        **pair,
        'cai': pair['cai'][:, ::step],
        't': pair['t'][::step],
        'nves': nves_ds,
    }


def peak_effcai_zoh(cai_trace, dt, tau_eff, min_ca=MIN_CA):
    eff = 0.0
    peak = 0.0
    for ca in np.asarray(cai_trace[1:], dtype=np.float64):
        ca_ext = max(ca - min_ca, 0.0)
        decay = np.exp(-dt / tau_eff)
        eff = eff * decay + ca_ext * tau_eff * (1.0 - decay)
        if eff > peak:
            peak = eff
    return peak


def epsp_ratio_from_basis(bmean, sm, rho_initial, rho_final):
    before = bmean + np.sum(np.where(rho_initial >= 0.5, sm - bmean, 0.0))
    after = bmean + np.sum(np.where(rho_final >= 0.5, sm - bmean, 0.0))
    return after / before if before > 0 else np.nan


def _simulate_pair_ratio_worker(args):
    pair, params, dt_step = args
    pair_name = pair.get('pair_name', 'unknown')
    try:
        ratio, rho_initial, rho_final = simulate_pair_details(pair, params, dt_step=dt_step)
        return pair_name, ratio, rho_initial, rho_final, None
    except Exception as exc:
        return pair_name, np.nan, None, None, str(exc)


def simulate_pair_details(pair, params, dt_step=10):
    pair = downsample_pair(pair, dt_step)

    cai = np.asarray(pair['cai'], dtype=np.float64)
    t = np.asarray(pair['t'], dtype=np.float64)
    rho = np.asarray(pair['rho0'], dtype=np.float64).copy()
    rho_initial = rho.copy()
    is_apical = np.asarray(pair['is_apical'], dtype=bool)
    sm = np.asarray(pair['singleton_means'], dtype=np.float64)
    bmean = float(pair['baseline_mean'])

    cai_pre = np.asarray(pair['cai_pre'], dtype=np.float64)
    cai_post = np.asarray(pair['cai_post'], dtype=np.float64)
    t_pre = np.asarray(pair['t_pre'], dtype=np.float64)
    t_post = np.asarray(pair['t_post'], dtype=np.float64)

    tau_eff = float(params['tau_eff'])
    dt_pre = max(float(t_pre[1] - t_pre[0]), 1e-6)
    dt_post = max(float(t_post[1] - t_post[0]), 1e-6)
    c_pre = np.array([peak_effcai_zoh(trace, dt_pre, tau_eff) for trace in cai_pre], dtype=np.float64)
    c_post = np.array([peak_effcai_zoh(trace, dt_post, tau_eff) for trace in cai_post], dtype=np.float64)

    theta_d = np.where(
        is_apical,
        params['a20'] * c_pre + params['a21'] * c_post,
        params['a00'] * c_pre + params['a01'] * c_post,
    )
    theta_p = np.where(
        is_apical,
        params['a30'] * c_pre + params['a31'] * c_post,
        params['a10'] * c_pre + params['a11'] * c_post,
    )

    eff = np.zeros_like(rho)
    cai_prev = cai[:, 0].copy()
    dt_arr = np.diff(t, prepend=t[0])
    if len(dt_arr) > 1:
        dt_arr[0] = dt_arr[1]
    else:
        dt_arr[0] = 1.0

    for idx in range(1, len(t)):
        dt = dt_arr[idx]
        ca = cai[:, idx]
        if dt > 0:
            f0 = np.maximum(cai_prev - MIN_CA, 0.0)
            f1 = np.maximum(ca - MIN_CA, 0.0)
            decay = np.exp(-dt / tau_eff)
            slope = (f1 - f0) / dt
            eff_new = (
                eff * decay
                + f0 * tau_eff * (1.0 - decay)
                + slope * (tau_eff * dt - tau_eff**2 * (1.0 - decay))
            )
        else:
            eff_new = eff.copy()

        pot = (eff > theta_p).astype(np.float64)
        dep = (eff > theta_d).astype(np.float64)
        drho = (
            -rho * (1.0 - rho) * (0.5 - rho)
            + pot * params['gamma_p'] * (1.0 - rho)
            - dep * params['gamma_d'] * rho
        ) / TAU_RHO
        if dt > 0:
            rho = np.clip(rho + dt * drho, 0.0, 1.0)
        eff = eff_new
        cai_prev = ca

    ratio = epsp_ratio_from_basis(bmean, sm, rho_initial, rho)
    return ratio, rho_initial, rho


def simulate_pair_ratio(pair, params, dt_step=10):
    ratio, _rho_initial, _rho_final = simulate_pair_details(pair, params, dt_step=dt_step)
    return ratio


def build_invitro(freq_hz):
    data = {'dt': [], 'mean': [], 'sem': []}
    for proto, mean in EXPERIMENTAL_TARGETS.items():
        if not proto.startswith(f"{freq_hz}Hz_"):
            continue
        delay = parse_delay_ms(proto)
        data['dt'].append(delay)
        data['mean'].append(mean)
        data['sem'].append(EXPERIMENTAL_ERRORS.get(proto, np.nan))
    order = np.argsort(data['dt'])
    for key in data:
        data[key] = np.asarray(data[key], dtype=np.float64)[order]
    return data if len(data['dt']) else None


def plot_curve(stats, freq_hz, output, invitro=None, label='in silico (basis model)', title_tag='gb-only'):
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'figure.dpi': 150,
        'legend.fontsize': 12,
    })

    fig, ax = plt.subplots(figsize=(7.5, 5.6))

    delays = np.array([item['delay'] for item in stats], dtype=np.float64)
    means = np.array([item['mean'] for item in stats], dtype=np.float64)
    sems = np.array([item['sem'] for item in stats], dtype=np.float64)

    ax.errorbar(
        delays, means, yerr=sems, fmt='o-', color='#5DA5DA',
        capsize=0, markersize=7, lw=2, label=label, zorder=3,
    )

    if invitro is not None and len(invitro['dt']):
        ax.errorbar(
            invitro['dt'], invitro['mean'], yerr=invitro['sem'],
            fmt='o-', color='#FAA43A', capsize=0, markersize=7, lw=2,
            label='in vitro', zorder=4,
        )

    ax.axhline(1.0, color='gray', ls='--', alpha=0.9, lw=1.5)
    ax.axvline(0.0, color='gray', ls='--', alpha=0.9, lw=1.5)
    ax.set_xlabel(r'$\Delta t$ (ms)')
    ax.set_ylabel('EPSP ratio')
    ax.set_title(f'Frequency = {freq_hz} Hz ({title_tag})')
    ax.legend(loc='upper left')
    ax.grid(False)

    plt.tight_layout()
    plt.savefig(output, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Plot GB-only basis-predicted STDP curve from calcium traces')
    parser.add_argument('--freq', type=int, default=10)
    parser.add_argument('--protocols', nargs='+', default=None,
                        help='Explicit protocols to use; otherwise all protocols for --freq are discovered')
    parser.add_argument('--params-json', type=str, default=None,
                        help='Optional JSON file with parameter values')
    parser.add_argument('--set', dest='overrides', action='append', default=[],
                        help='Parameter override of the form key=value (repeatable)')
    parser.add_argument('--trace-dir', type=str, default='/project/rrg-emuller/dhuruva/plastyfitting/trace_results/CHINDEMI_PARAMS')
    parser.add_argument('--basis-dir', type=str, default=L5_BASIS_DIR)
    parser.add_argument('--max-pairs', type=int, default=100)
    parser.add_argument('--dt-step', type=int, default=10)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--label', type=str, default='in silico (basis model)')
    parser.add_argument('--title-tag', type=str, default='gb-only')
    parser.add_argument('--workers', type=int, default=0,
                        help='Number of CPU workers for pair simulation (0=auto)')
    parser.add_argument('--report-rho', action='store_true',
                        help='Print rho transition statistics per protocol')
    args = parser.parse_args()

    params = load_param_dict(args.params_json, args.overrides)
    protocols = args.protocols or discover_protocols(args.trace_dir, args.freq)
    if not protocols:
        raise SystemExit(f'No protocols found for {args.freq} Hz in {args.trace_dir}')

    workers = args.workers if args.workers and args.workers > 0 else max(1, (os.cpu_count() or 1) - 1)
    print(f'Discovered protocols: {protocols}', flush=True)
    print(f'Using {workers} worker(s)', flush=True)
    if args.dt_step > 1:
        print(f'Warning: dt-step={args.dt_step} coarsens the calcium traces and can smear timing-sensitive rho transitions.', flush=True)

    protocol_data = load_protocol_data(
        trace_dir=args.trace_dir,
        protocols=protocols,
        max_pairs=args.max_pairs,
        basis_dir=args.basis_dir,
    )

    stats = []
    print('Using parameters:')
    for key, value in params.items():
        print(f'  {key} = {value}')

    print('\nPer-protocol pair counts and predictions:')
    for proto in sorted(protocols, key=parse_delay_ms):
        pairs = protocol_data.get(proto, [])
        ratios = []
        for pair in pairs:
            try:
                ratio = simulate_pair_ratio(pair, params, dt_step=args.dt_step)
            except Exception as exc:
                print(f'  {proto}: skipped pair due to {exc}')
                continue
            if np.isfinite(ratio):
                ratios.append(ratio)
        if not ratios:
            print(f'  {proto}: no valid pairs')
            continue
        ratios = np.asarray(ratios, dtype=np.float64)
        mean = float(np.mean(ratios))
        sem = float(np.std(ratios, ddof=1) / np.sqrt(len(ratios))) if len(ratios) > 1 else 0.0
        stats.append({
            'protocol': proto,
            'delay': parse_delay_ms(proto),
            'mean': mean,
            'sem': sem,
            'n_pairs': int(len(ratios)),
        })
        print(f'  {proto:<12s} n={len(ratios):3d}  mean={mean:.4f}  sem={sem:.4f}')

    if not stats:
        raise SystemExit('No valid predictions were produced.')

    invitro = build_invitro(args.freq)
    output = args.output or f'gb_basis_stdp_{args.freq}Hz.png'
    plot_curve(stats, args.freq, output, invitro=invitro, label=args.label, title_tag=args.title_tag)
    print(f'\nSaved -> {output}')


if __name__ == '__main__':
    main()

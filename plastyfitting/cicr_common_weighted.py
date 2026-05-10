#!/usr/bin/env python3
"""
Minimal weighted-calcium extension of cicr_common.

Adds support for loading cai_NMDA_CR and cai_VDCC_CR, combining them as
cai_CR_weighted = alpha * cai_NMDA_CR + beta * cai_VDCC_CR, and fitting
alpha/beta inside GB-only style models.
"""

import os
import json
import time
import pickle
import logging
import argparse
from functools import partial
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

import cicr_common as base
from .cicr_common import (
    EXPERIMENTAL_TARGETS,
    EXPERIMENTAL_ERRORS,
    BASE_DIR,
    L5_TRACE_DIR,
    L23_TRACE_DIR,
    L5_BASIS_DIR,
    L23_BASIS_DIR,
    THRESHOLD_TRACE_DIR,
    PROTOCOL_PATHWAY,
    _jax_peak_effcai_zoh,
)

logger = logging.getLogger(__name__)


_MIN_CA_CR = 70e-6  # mM — resting calcium, matches min_ca_CR in GluSynapse.mod


def combine_weighted_cai_np(cai_nmda, cai_vdcc, alpha, beta):
    """Weighted combination preserving the resting baseline at min_ca_CR.

    Both cai_NMDA_CR and cai_VDCC_CR decay to min_ca_CR (not zero), so only
    the excess above resting is weighted:

        cai_CR = min_ca + alpha*(cai_NMDA - min_ca) + beta*(cai_VDCC - min_ca)
    """
    nmda = np.asarray(cai_nmda, dtype=np.float64)
    vdcc = np.asarray(cai_vdcc, dtype=np.float64)
    return _MIN_CA_CR + alpha * (nmda - _MIN_CA_CR) + beta * (vdcc - _MIN_CA_CR)


def combine_weighted_cai_jax(cai_nmda, cai_vdcc, alpha, beta):
    """JAX version of combine_weighted_cai_np — see docstring above."""
    return _MIN_CA_CR + alpha * (cai_nmda - _MIN_CA_CR) + beta * (cai_vdcc - _MIN_CA_CR)


def _coerce_2d_array(arr, nsyn=None):
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if nsyn is not None and arr.shape[0] != nsyn and arr.shape[1] == nsyn:
        arr = arr.T
    return arr


def _stack_sim_component(data, key, gids, fallback):
    raw = data.get(key, None)
    if isinstance(raw, dict) and all(g in raw for g in gids):
        return np.stack([np.asarray(raw[g], dtype=np.float64) for g in gids], axis=0)
    if raw is not None:
        arr = _coerce_2d_array(raw, nsyn=len(gids))
        if arr.shape == fallback.shape:
            return arr
    return np.asarray(fallback, dtype=np.float64).copy()


def _find_threshold_path(pair_name):
    cpre_cpost_path = Path(BASE_DIR) / "trace_results/CHINDEMI_PARAMS/cpre_cpost_traces" / f"{pair_name}_threshold_traces.pkl"
    threshold_path = Path(THRESHOLD_TRACE_DIR) / f"{pair_name}_threshold_traces.pkl"
    legacy_threshold_path = Path(BASE_DIR) / "trace_results/cpre_cpost_cai_raw_traces" / f"{pair_name}_threshold_traces.pkl"
    for p in (cpre_cpost_path, threshold_path, legacy_threshold_path):
        if p.exists():
            return p
    return None


def _stack_threshold_component(thresh_data, side, key, gids, fallback):
    section = thresh_data.get(side, {})
    raw = section.get(key, None)
    if isinstance(raw, dict) and all(g in raw for g in gids):
        return np.stack([np.asarray(raw[g], dtype=np.float64) for g in gids], axis=0)
    return np.asarray(fallback, dtype=np.float64).copy()


def _match_t_np(t_raw, n_pts):
    t = np.asarray(t_raw, dtype=np.float64)
    if len(t) == n_pts and np.all(np.diff(t) > 0):
        return t
    if n_pts <= 1:
        return np.zeros(n_pts, dtype=np.float64)
    inferred_dt = (t[-1] - t[0]) / (n_pts - 1)
    return np.arange(n_pts, dtype=np.float64) * inferred_dt


def _get_threshold_trace_len(thresh_data, side):
    section = thresh_data.get(side, {})
    for key in ("cai_NMDA_CR", "cai_VDCC_CR", "cai_CR"):
        raw = section.get(key, None)
        if isinstance(raw, dict) and len(raw) > 0:
            first = next(iter(raw.values()))
            return len(np.asarray(first, dtype=np.float64))
    return 0


def _load_pkl_weighted(
    pkl_path,
    needs_threshold_traces=True,
    include_raw_cai=True,
    include_base_threshold_traces=True,
    include_cpre_cpost=True,
):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    t = np.asarray(data["t"], dtype=np.float64)
    cai_raw = data["cai_CR"]
    if isinstance(cai_raw, dict):
        gids = list(cai_raw.keys())
        cai_base = np.stack([np.asarray(cai_raw[g], dtype=np.float64) for g in gids], axis=0)

        rho_gb_dict = data.get("rho_GB", {})
        rho0 = np.array(
            [float(np.asarray(rho_gb_dict[g])[0]) if g in rho_gb_dict else 0.0 for g in gids],
            dtype=np.float64,
        )

        syn_props = data.get("syn_props", {})
        loc_list = syn_props.get("loc", ["basal"] * len(gids))
        is_apical = np.array([loc == "apical" for loc in loc_list], dtype=bool)
    else:
        cai_base = _coerce_2d_array(cai_raw)
        gids = list(range(cai_base.shape[0]))

        rho0 = np.zeros(cai_base.shape[0], dtype=np.float64)
        if "rho_GB" in data:
            rho_gb = _coerce_2d_array(data["rho_GB"], nsyn=cai_base.shape[0])
            rho0 = rho_gb[:, 0].copy()

        syn_props = data.get("synprop", {})
        loc_list = syn_props.get("loc", ["basal"] * cai_base.shape[0])
        is_apical = np.array([loc == "apical" for loc in loc_list], dtype=bool)

    nves = np.zeros(len(t), dtype=np.float64)
    prespikes = data.get("prespikes", [])
    if len(prespikes) > 0:
        for spk_t in np.asarray(prespikes, dtype=np.float64):
            idx = np.searchsorted(t, spk_t)
            if idx < len(t):
                nves[idx] = 1.0

    pair = {
        "t": t,
        "rho0": rho0,
        "is_apical": is_apical,
        "nves": nves,
    }
    if include_raw_cai:
        pair["cai"] = cai_base

    pair["cai_nmda"] = _stack_sim_component(data, "cai_NMDA_CR", gids, cai_base)
    pair["cai_vdcc"] = _stack_sim_component(data, "cai_VDCC_CR", gids, np.zeros_like(cai_base))
    pair["cai_weighted_default"] = combine_weighted_cai_np(pair["cai_nmda"], pair["cai_vdcc"], 1.0, 1.0)

    nsyn = cai_base.shape[0]
    pair["c_pre"] = np.zeros(nsyn, dtype=np.float64)
    pair["c_post"] = np.zeros(nsyn, dtype=np.float64)
    pair["c_pre_cr"] = np.zeros(nsyn, dtype=np.float64)
    pair["c_post_cr"] = np.zeros(nsyn, dtype=np.float64)
    pair["cai_pre"] = np.zeros((nsyn, 1), dtype=np.float64)
    pair["cai_post"] = np.zeros((nsyn, 1), dtype=np.float64)
    pair["t_pre"] = np.zeros(1, dtype=np.float64)
    pair["t_post"] = np.zeros(1, dtype=np.float64)
    pair["cai_pre_nmda"] = np.zeros((nsyn, 1), dtype=np.float64)
    pair["cai_pre_vdcc"] = np.zeros((nsyn, 1), dtype=np.float64)
    pair["cai_post_nmda"] = np.zeros((nsyn, 1), dtype=np.float64)
    pair["cai_post_vdcc"] = np.zeros((nsyn, 1), dtype=np.float64)

    pair_name = Path(pkl_path).parent.parent.name
    thresh_path = _find_threshold_path(pair_name)
    if thresh_path is not None and (needs_threshold_traces or include_cpre_cpost):
        with open(thresh_path, "rb") as ft:
            thresh_data = pickle.load(ft)

        pre_source = None
        for key in ("cai_NMDA_CR", "cai_VDCC_CR", "cai_CR"):
            maybe = thresh_data.get("pre", {}).get(key, None)
            if isinstance(maybe, dict) and len(maybe) > 0:
                pre_source = maybe
                break
        if pre_source is None:
            raise KeyError(f"No usable pre-threshold calcium traces found in {thresh_path}")

        pre_keys = list(pre_source.keys())
        gids_ordered = pre_keys[:nsyn]

        if include_cpre_cpost:
            c_pre, c_post = base.compute_cpre_cpost_from_shaft_cai(thresh_data, gids_ordered, ca_key="cai_CR")
            pair["c_pre"] = c_pre
            pair["c_post"] = c_post
            pair["c_pre_cr"] = c_pre.copy()
            pair["c_post_cr"] = c_post.copy()

        if needs_threshold_traces:
            pre_nmda_source = thresh_data.get("pre", {}).get("cai_NMDA_CR", None)
            post_nmda_source = thresh_data.get("post", {}).get("cai_NMDA_CR", None)
            n_pre = _get_threshold_trace_len(thresh_data, "pre")
            n_post = _get_threshold_trace_len(thresh_data, "post")

            if n_pre < 2 or n_post < 2:
                raise ValueError(
                    f"Weighted threshold traces for pair '{pair_name}' are too short "
                    f"(pre={n_pre}, post={n_post}). Need at least 2 samples in one of "
                    f"cai_NMDA_CR/cai_VDCC_CR/cai_CR for each side."
                )

            pair["t_pre"] = _match_t_np(thresh_data["pre"]["t"], n_pre)
            pair["t_post"] = _match_t_np(thresh_data["post"]["t"], n_post)
            pair["cai_pre_nmda"] = _stack_threshold_component(
                thresh_data, "pre", "cai_NMDA_CR", gids_ordered, np.zeros((nsyn, n_pre), dtype=np.float64)
            )
            pair["cai_pre_vdcc"] = _stack_threshold_component(
                thresh_data, "pre", "cai_VDCC_CR", gids_ordered, np.zeros((nsyn, n_pre), dtype=np.float64)
            )
            pair["cai_post_nmda"] = _stack_threshold_component(
                thresh_data, "post", "cai_NMDA_CR", gids_ordered, np.zeros((nsyn, n_post), dtype=np.float64)
            )
            pair["cai_post_vdcc"] = _stack_threshold_component(
                thresh_data, "post", "cai_VDCC_CR", gids_ordered, np.zeros((nsyn, n_post), dtype=np.float64)
            )

            if include_base_threshold_traces:
                pair["cai_pre"] = _stack_threshold_component(
                    thresh_data, "pre", "cai_CR", gids_ordered, np.zeros((nsyn, n_pre), dtype=np.float64)
                )
                pair["cai_post"] = _stack_threshold_component(
                    thresh_data, "post", "cai_CR", gids_ordered, np.zeros((nsyn, n_post), dtype=np.float64)
                )

    elif needs_threshold_traces or include_cpre_cpost:
        raise FileNotFoundError(f"No threshold traces found for pair '{pair_name}'")

    return pair


def preload_all_data(
    max_pairs=None,
    protocols=None,
    needs_threshold_traces=True,
    include_raw_cai=True,
    include_base_threshold_traces=True,
    include_cpre_cpost=True,
):
    if protocols is None:
        protocols = list(EXPERIMENTAL_TARGETS.keys())
    protocol_data = {proto: [] for proto in protocols}

    def _load_dir(trace_dir, basis_dir, protos):
        protos = [p for p in protos if p in protocol_data]
        if not protos or not Path(trace_dir).exists():
            return
        dirs = sorted(d for d in Path(trace_dir).iterdir() if d.is_dir())
        loaded = 0
        for pair_dir in dirs:
            if max_pairs and all(len(protocol_data[p]) >= max_pairs for p in protos if p in protocol_data):
                break
            parts = pair_dir.name.split("-")
            if len(parts) != 2:
                continue
            basis = base._load_basis(int(parts[0]), int(parts[1]), basis_dir)
            if not basis:
                continue
            for proto in protos:
                if max_pairs and len(protocol_data[proto]) >= max_pairs:
                    continue
                pkl_path = pair_dir / proto / "simulation_traces.pkl"
                if not pkl_path.exists():
                    continue
                try:
                    pd_item = _load_pkl_weighted(
                        str(pkl_path),
                        needs_threshold_traces=needs_threshold_traces,
                        include_raw_cai=include_raw_cai,
                        include_base_threshold_traces=include_base_threshold_traces,
                        include_cpre_cpost=include_cpre_cpost,
                    )
                except Exception as exc:
                    logger.warning(f"Skipping {pair_dir.name}/{proto}: {exc}")
                    continue
                if pd_item["cai_nmda"].shape[0] == basis["n_syn"]:
                    pd_item.update(basis)
                    protocol_data[proto].append(pd_item)
                    loaded += 1
                    logger.info(f"[{loaded}] Loaded {pair_dir.name}/{proto}")

    _load_dir(L5_TRACE_DIR, L5_BASIS_DIR, [p for p, pw in PROTOCOL_PATHWAY.items() if pw == "L5TTPC"])
    _load_dir(L23_TRACE_DIR, L23_BASIS_DIR, ["50Hz_10ms"])
    return protocol_data


def _interp_2d(traces, t_old, t_new):
    out = np.zeros((traces.shape[0], len(t_new)), dtype=np.float64)
    for s in range(traces.shape[0]):
        out[s] = np.interp(t_new, t_old, traces[s])
    return out


def _preprocess_pair(pair, dt_step=1, interp_dt=None):
    p = dict(pair)
    # Helper: length of the time axis, works whether or not cai_CR is stored
    def _nt(p):
        for key in ("cai", "cai_nmda", "cai_vdcc"):
            v = p.get(key)
            if v is not None:
                return v.shape[1]
        return len(p["t"])

    if interp_dt is not None:
        t_old = p["t"]
        t_new = np.arange(t_old[0], t_old[-1], interp_dt)
        if p.get("cai") is not None:
            p["cai"] = _interp_2d(p["cai"], t_old, t_new)
        p["cai_nmda"] = _interp_2d(p["cai_nmda"], t_old, t_new)
        p["cai_vdcc"] = _interp_2d(p["cai_vdcc"], t_old, t_new)
        if p.get("shaft_cai") is not None:
            p["shaft_cai"] = _interp_2d(p["shaft_cai"], t_old, t_new)
        nves_old = p.get("nves", np.zeros(len(t_old), dtype=np.float64))
        nves_new = np.zeros(len(t_new), dtype=np.float64)
        for i, v in enumerate(nves_old):
            if v > 0:
                idx = int(np.argmin(np.abs(t_new - t_old[i])))
                nves_new[idx] += v
        p["nves"] = nves_new
        p["t"] = t_new
    elif dt_step > 1:
        nves_full = p.get("nves", np.zeros(_nt(p), dtype=np.float64))
        n = len(nves_full)
        pad = (-n) % dt_step
        nves_padded = np.pad(nves_full, (0, pad))
        p["nves"] = nves_padded.reshape(-1, dt_step).sum(axis=1)
        if p.get("cai") is not None:
            p["cai"] = p["cai"][:, ::dt_step]
        p["cai_nmda"] = p["cai_nmda"][:, ::dt_step]
        p["cai_vdcc"] = p["cai_vdcc"][:, ::dt_step]
        p["t"] = p["t"][::dt_step]
        if p.get("shaft_cai") is not None:
            p["shaft_cai"] = p["shaft_cai"][:, ::dt_step]
    return p


def collate_protocol_to_jax(
    pairs_list,
    dt_step=1,
    interp_dt=None,
    include_raw_cai=True,
    include_base_threshold_traces=True,
    include_cpre_cpost=True,
):
    if len(pairs_list) == 0:
        return None

    pairs_proc = [_preprocess_pair(p, dt_step=dt_step, interp_dt=interp_dt) for p in pairs_list]

    n_pairs = len(pairs_proc)
    max_syn = max(p["cai_nmda"].shape[0] for p in pairs_proc)
    max_time = max(len(p["t"]) for p in pairs_proc)
    max_t_pre = max(len(p.get("t_pre", np.zeros(1, dtype=np.float64))) for p in pairs_proc)
    max_t_post = max(len(p.get("t_post", np.zeros(1, dtype=np.float64))) for p in pairs_proc)

    cai = np.zeros((n_pairs, max_syn, max_time), dtype=np.float64) if include_raw_cai else None
    t = np.zeros((n_pairs, max_time), dtype=np.float64)
    c_pre = np.zeros((n_pairs, max_syn), dtype=np.float64)
    c_post = np.zeros((n_pairs, max_syn), dtype=np.float64)
    c_pre_cr = np.zeros((n_pairs, max_syn), dtype=np.float64)
    c_post_cr = np.zeros((n_pairs, max_syn), dtype=np.float64)
    is_apical = np.zeros((n_pairs, max_syn), dtype=bool)
    rho0 = np.zeros((n_pairs, max_syn), dtype=np.float64)
    baseline = np.zeros(n_pairs, dtype=np.float64)
    singletons = np.zeros((n_pairs, max_syn), dtype=np.float64)
    valid = np.zeros((n_pairs, max_syn), dtype=bool)
    nves = np.zeros((n_pairs, max_time), dtype=np.float64)
    cai_pre = np.zeros((n_pairs, max_syn, max_t_pre if include_base_threshold_traces else 1), dtype=np.float64)
    cai_post = np.zeros((n_pairs, max_syn, max_t_post if include_base_threshold_traces else 1), dtype=np.float64)
    t_pre = np.zeros((n_pairs, max_t_pre), dtype=np.float64)
    t_post = np.zeros((n_pairs, max_t_post), dtype=np.float64)

    cai_nmda = np.zeros((n_pairs, max_syn, max_time), dtype=np.float64)
    cai_vdcc = np.zeros((n_pairs, max_syn, max_time), dtype=np.float64)
    cai_pre_nmda = np.zeros((n_pairs, max_syn, max_t_pre), dtype=np.float64)
    cai_pre_vdcc = np.zeros((n_pairs, max_syn, max_t_pre), dtype=np.float64)
    cai_post_nmda = np.zeros((n_pairs, max_syn, max_t_post), dtype=np.float64)
    cai_post_vdcc = np.zeros((n_pairs, max_syn, max_t_post), dtype=np.float64)

    for i, p in enumerate(pairs_proc):
        ns, nt = p["cai_nmda"].shape
        if include_raw_cai:
            cai[i, :ns, :nt] = p["cai"]
            if nt < max_time:
                cai[i, :ns, nt:] = p["cai"][:, -1:]
        t[i, :nt] = p["t"]
        if nt < max_time:
            t[i, nt:] = p["t"][-1]
        if include_cpre_cpost:
            c_pre[i, :ns] = p.get("c_pre", 0.0)
            c_post[i, :ns] = p.get("c_post", 0.0)
            c_pre_cr[i, :ns] = p.get("c_pre_cr", p.get("c_pre", 0.0))
            c_post_cr[i, :ns] = p.get("c_post_cr", p.get("c_post", 0.0))
        is_apical[i, :ns] = p["is_apical"]
        rho0[i, :ns] = p["rho0"]
        baseline[i] = p["baseline_mean"]
        singletons[i, :ns] = p["singleton_means"]
        valid[i, :ns] = True
        nves[i, :nt] = p.get("nves", np.zeros(nt, dtype=np.float64))[:nt]

        cai_nmda[i, :ns, :nt] = p["cai_nmda"]
        cai_vdcc[i, :ns, :nt] = p["cai_vdcc"]
        if nt < max_time:
            cai_nmda[i, :ns, nt:] = p["cai_nmda"][:, -1:]
            cai_vdcc[i, :ns, nt:] = p["cai_vdcc"][:, -1:]

        nt_pre = p["cai_pre_nmda"].shape[1]
        cai_pre_nmda[i, :ns, :nt_pre] = p["cai_pre_nmda"]
        cai_pre_vdcc[i, :ns, :nt_pre] = p["cai_pre_vdcc"]
        if nt_pre < max_t_pre:
            cai_pre_nmda[i, :ns, nt_pre:] = p["cai_pre_nmda"][:, -1:]
            cai_pre_vdcc[i, :ns, nt_pre:] = p["cai_pre_vdcc"][:, -1:]
        t_pre[i, :nt_pre] = p["t_pre"]
        if nt_pre < max_t_pre:
            t_pre[i, nt_pre:] = p["t_pre"][-1]
        if include_base_threshold_traces:
            cai_pre[i, :ns, :nt_pre] = p["cai_pre"]
            if nt_pre < max_t_pre:
                cai_pre[i, :ns, nt_pre:] = p["cai_pre"][:, -1:]

        nt_post = p["cai_post_nmda"].shape[1]
        cai_post_nmda[i, :ns, :nt_post] = p["cai_post_nmda"]
        cai_post_vdcc[i, :ns, :nt_post] = p["cai_post_vdcc"]
        if nt_post < max_t_post:
            cai_post_nmda[i, :ns, nt_post:] = p["cai_post_nmda"][:, -1:]
            cai_post_vdcc[i, :ns, nt_post:] = p["cai_post_vdcc"][:, -1:]
        t_post[i, :nt_post] = p["t_post"]
        if nt_post < max_t_post:
            t_post[i, nt_post:] = p["t_post"][-1]
        if include_base_threshold_traces:
            cai_post[i, :ns, :nt_post] = p["cai_post"]
            if nt_post < max_t_post:
                cai_post[i, :ns, nt_post:] = p["cai_post"][:, -1:]

    cai_nmda_jax = jnp.array(cai_nmda)
    cai_vdcc_jax = jnp.array(cai_vdcc)
    return {
        "cai": jnp.array(cai) if include_raw_cai else cai_nmda_jax,
        "t": jnp.array(t),
        "c_pre": jnp.array(c_pre),
        "c_post": jnp.array(c_post),
        "c_pre_cr": jnp.array(c_pre_cr),
        "c_post_cr": jnp.array(c_post_cr),
        "is_apical": jnp.array(is_apical),
        "rho0": jnp.array(rho0),
        "baseline": jnp.array(baseline),
        "singletons": jnp.array(singletons),
        "valid": jnp.array(valid),
        "nves": jnp.array(nves),
        "shaft_cai": None,
        "cai_pre": jnp.array(cai_pre),
        "t_pre": jnp.array(t_pre),
        "cai_post": jnp.array(cai_post),
        "t_post": jnp.array(t_post),
        "cai_nmda": cai_nmda_jax,
        "cai_vdcc": cai_vdcc_jax,
        "cai_pre_nmda": jnp.array(cai_pre_nmda),
        "cai_pre_vdcc": jnp.array(cai_pre_vdcc),
        "cai_post_nmda": jnp.array(cai_post_nmda),
        "cai_post_vdcc": jnp.array(cai_post_vdcc),
    }


class WeightedCICRModel(base.CICRModel):
    NEEDS_RAW_CAI_TRACE = True
    NEEDS_BASE_THRESHOLD_TRACES = True
    NEEDS_CPRE_CPOST = True

    def combine_cai_trace(self, cai_total, cai_nmda, cai_vdcc, params):
        return cai_total

    def setup_jax(self, protocol_data, targets, lambda_reg=0.0, dt_step=1, interp_dt=None, use_loss_v2=False, use_loss_basic=False):
        self.targets_dict = targets
        self.proto_names = list(targets.keys())
        self.target_vals = jnp.array([targets[p] for p in self.proto_names])
        self.target_errs = jnp.array([EXPERIMENTAL_ERRORS.get(p, 0.1) for p in self.proto_names])
        self.weights = jnp.array([1.2 if p == "10Hz_-10ms" else 1.0 for p in self.proto_names])
        self.lambda_reg = lambda_reg
        self.use_loss_v2 = use_loss_v2
        self.use_loss_basic = use_loss_basic

        raw_collated = {
            p: collate_protocol_to_jax(
                data,
                dt_step=dt_step,
                interp_dt=interp_dt,
                include_raw_cai=self.NEEDS_RAW_CAI_TRACE,
                include_base_threshold_traces=self.NEEDS_BASE_THRESHOLD_TRACES,
                include_cpre_cpost=self.NEEDS_CPRE_CPOST,
            )
            for p, data in protocol_data.items() if p in targets
        }
        self.collated_data = {p: d for p, d in raw_collated.items() if d is not None}
        self.proto_names = [p for p in self.proto_names if p in self.collated_data]
        self.target_vals = jnp.array([targets[p] for p in self.proto_names])
        self.target_errs = jnp.array([EXPERIMENTAL_ERRORS.get(p, 0.1) for p in self.proto_names])
        self.weights = jnp.array([1.2 if p == "10Hz_-10ms" else 1.0 for p in self.proto_names])

        step_factory = self.get_step_factory()
        init_fn = self.get_init_fn()

        def sim_synapse(cai_trace, cai_nmda_trace, cai_vdcc_trace, t_trace, nves_trace,
                        c_pre, c_post, is_apical, rho0, sm, bmean, valid,
                        cai_pre, cai_pre_nmda, cai_pre_vdcc, t_pre,
                        cai_post, cai_post_nmda, cai_post_vdcc, t_post, params):
            dt_trace = jnp.diff(t_trace, prepend=t_trace[0])
            dt_trace = jnp.where(dt_trace <= 0, 1e-6, dt_trace)
            cai_weighted = self.combine_cai_trace(cai_trace, cai_nmda_trace, cai_vdcc_trace, params)
            init = init_fn(cai_weighted[0], rho0)
            syn_params = (c_pre, c_post, is_apical,
                          cai_pre, cai_pre_nmda, cai_pre_vdcc, t_pre,
                          cai_post, cai_post_nmda, cai_post_vdcc, t_post)
            scan_fn = step_factory(params, syn_params)
            final, _ = jax.lax.scan(scan_fn, init, (cai_weighted, dt_trace, nves_trace))
            rho_final = final[-1]
            return jnp.where(valid & (rho_final >= 0.5), sm - bmean, 0.0)

        vmapsyn = jax.vmap(
            sim_synapse,
            in_axes=(0, 0, 0, None, None, 0, 0, 0, 0, 0, None, 0, 0, 0, 0, None, 0, 0, 0, None, None)
        )

        def sim_pair(cai_p, cai_nmda_p, cai_vdcc_p, t_p, nves_p,
                     cpre_p, cpost_p, isapi_p, rho0_p, bmean, sm_p, valid_p,
                     cai_pre_p, cai_pre_nmda_p, cai_pre_vdcc_p, t_pre_p,
                     cai_post_p, cai_post_nmda_p, cai_post_vdcc_p, t_post_p, params):
            contribs = vmapsyn(
                cai_p, cai_nmda_p, cai_vdcc_p, t_p, nves_p,
                cpre_p, cpost_p, isapi_p, rho0_p, sm_p, bmean, valid_p,
                cai_pre_p, cai_pre_nmda_p, cai_pre_vdcc_p, t_pre_p,
                cai_post_p, cai_post_nmda_p, cai_post_vdcc_p, t_post_p, params
            )
            epsp_after = bmean + jnp.sum(contribs)
            contribs_before = jnp.where(valid_p & (rho0_p >= 0.5), sm_p - bmean, 0.0)
            epsp_before = bmean + jnp.sum(contribs_before)
            return jnp.where(epsp_before > 0, epsp_after / epsp_before, jnp.nan)

        vmappair = jax.vmap(
            sim_pair,
            in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None)
        )

        def sim_protocol(proto_data, params):
            ratios = vmappair(
                proto_data["cai"], proto_data["cai_nmda"], proto_data["cai_vdcc"],
                proto_data["t"], proto_data["nves"],
                proto_data["c_pre"], proto_data["c_post"], proto_data["is_apical"], proto_data["rho0"],
                proto_data["baseline"], proto_data["singletons"], proto_data["valid"],
                proto_data["cai_pre"], proto_data["cai_pre_nmda"], proto_data["cai_pre_vdcc"], proto_data["t_pre"],
                proto_data["cai_post"], proto_data["cai_post_nmda"], proto_data["cai_post_vdcc"], proto_data["t_post"],
                params
            )
            return jnp.nanmean(ratios)

        def forward_single(x_array, collated):
            params = self.unpack_params(x_array)
            return jnp.stack([sim_protocol(collated[p], params) for p in self.proto_names])

        @jax.jit
        def forward_batch(x_matrix, collated):
            return jax.vmap(forward_single, in_axes=(0, None))(x_matrix, collated)

        idx_ltp = self.proto_names.index("10Hz_10ms") if "10Hz_10ms" in self.proto_names else -1
        idx_ltd = self.proto_names.index("10Hz_-10ms") if "10Hz_-10ms" in self.proto_names else -1

        @jax.jit
        def objective_single(x_array, collated):
            preds = forward_single(x_array, collated)

            if self.use_loss_basic:
                # Pure weighted MSE — no sep_penalty, no regularization
                return jnp.sum(self.weights * (preds - self.target_vals) ** 2)

            if self.use_loss_v2:
                base_loss = jnp.sum(self.weights * jnp.abs(preds - self.target_vals) / self.target_errs)
            else:
                base_loss = jnp.sum(self.weights * (preds - self.target_vals) ** 2)
            sep_penalty = 0.0
            if idx_ltp >= 0 and idx_ltd >= 0:
                actual_sep = preds[idx_ltp] - preds[idx_ltd]
                sep_penalty = 20.0 * jnp.maximum(0.0, 0.41 - actual_sep) ** 2
            loss = base_loss + sep_penalty
            if self.lambda_reg > 0:
                loss = loss + self.lambda_reg * jnp.sum((x_array - self.DEFAULT_X0) ** 2)
            return loss

        @jax.jit
        def objective_batch(x_matrix, collated):
            return jax.vmap(objective_single, in_axes=(0, None))(x_matrix, collated)

        self.forward_batch = partial(forward_batch, collated=self.collated_data)
        self.objective_single = partial(objective_single, collated=self.collated_data)
        self.objective_batch = partial(objective_batch, collated=self.collated_data)

    def run(self):
        parser = argparse.ArgumentParser(description=f"Weighted JAX model ({self.DESCRIPTION})")
        parser.add_argument("--method", choices=["de", "cmaes", "optax", "optuna", "pso"], default="de")
        parser.add_argument("--max-iter", type=int, default=1000)
        parser.add_argument("--protocols", nargs="+", choices=list(EXPERIMENTAL_TARGETS.keys()))
        parser.add_argument("--max-pairs", type=int, default=None)
        parser.add_argument("--dt-step", type=int, default=1)
        parser.add_argument("--popsize", type=int, default=15)
        parser.add_argument("--de-batch-size", type=int, default=8)
        parser.add_argument("--early-stopping", type=int, default=100)
        parser.add_argument("--interp-dt", type=float, default=None)
        parser.add_argument("--eval", type=str, default=None)
        parser.add_argument("--loss-v2", action="store_true")
        parser.add_argument("--loss-basic", action="store_true", help="Pure weighted MSE: sum_p w_p*(pred_p - target_p)^2, no sep_penalty")
        parser.add_argument("--verbose", action="store_true")
        args = parser.parse_args()

        if args.verbose:
            logging.getLogger().setLevel(logging.INFO)

        targets = {p: v for p, v in EXPERIMENTAL_TARGETS.items() if p in args.protocols} if args.protocols else EXPERIMENTAL_TARGETS
        protocol_data = preload_all_data(
            max_pairs=args.max_pairs,
            protocols=list(targets.keys()),
            needs_threshold_traces=self.NEEDS_THRESHOLD_TRACES,
            include_raw_cai=self.NEEDS_RAW_CAI_TRACE,
            include_base_threshold_traces=self.NEEDS_BASE_THRESHOLD_TRACES,
            include_cpre_cpost=self.NEEDS_CPRE_CPOST,
        )
        self.setup_jax(protocol_data, targets, dt_step=args.dt_step, interp_dt=args.interp_dt, use_loss_v2=args.loss_v2, use_loss_basic=args.loss_basic)

        t0 = time.time()
        if args.eval:
            if os.path.isfile(args.eval):
                with open(args.eval) as f:
                    ep = json.load(f)
            else:
                ep = json.loads(args.eval)
            dp = dict(self.DEFAULT_PARAMS)
            dp.update(ep)
            x_eval = np.array([dp[n] for n in self.PARAM_NAMES], dtype=np.float64)
            res = {"method": "eval", "x": x_eval.tolist(), "fun": float(self.objective_single(jnp.array(x_eval))), "time": time.time() - t0}
        else:
            if args.method == "cmaes":
                res = self.run_cmaes(max_iter=args.max_iter, popsize=args.popsize, patience=args.early_stopping)
            elif args.method == "optax":
                res = self.run_optax(max_iter=args.max_iter)
            elif args.method == "optuna":
                res = self.run_optuna(max_iter=args.max_iter)
            elif args.method == "pso":
                res = self.run_pso(max_iter=args.max_iter, popsize=args.popsize)
            else:
                res = self.run_de(max_iter=args.max_iter, popsize=args.popsize, patience=args.early_stopping, de_batch_size=args.de_batch_size)
            res["time"] = time.time() - t0

        self.print_results(res)

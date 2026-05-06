#!/usr/bin/env python3
"""Inspect a new fixed-dt simulation_traces.pkl to understand its structure."""
import pickle, numpy as np
from pathlib import Path

PKL = "/project/rrg-emuller/dhuruva/plastyfitting/trace_results/CHINDEMI_PARAMS/184090-195582/10Hz_10ms/simulation_traces.pkl"

with open(PKL, "rb") as f:
    data = pickle.load(f)

print("=== TOP-LEVEL KEYS ===")
print(list(data.keys()))

t = np.asarray(data["t"])
dts = np.diff(t)
print(f"\nt: shape={t.shape}, min={t.min():.4f}ms, max={t.max():.4f}ms")
print(f"  dt range: [{dts.min():.6g}, {dts.max():.6g}] ms")
print(f"  dt std: {dts.std():.6g} ms")
print(f"  first 5 dt: {dts[:5]}")

def describe(label, v, indent=""):
    if isinstance(v, np.ndarray):
        print(f"{indent}[{label}]: ndarray shape={v.shape} dtype={v.dtype} min={v.min():.6g} max={v.max():.6g}")
    elif isinstance(v, dict):
        keys = list(v.keys())
        print(f"{indent}[{label}]: dict with {len(keys)} keys, first key type={type(keys[0]).__name__}")
        for k in keys[:3]:
            describe(str(k), v[k], indent + "  ")
    elif isinstance(v, (list, tuple)):
        print(f"{indent}[{label}]: {type(v).__name__} len={len(v)}")
        if v: describe("0", v[0] if not isinstance(v[0], dict) else list(v[0].items())[0][1], indent + "  ")
    else:
        print(f"{indent}[{label}]: {type(v).__name__} = {v}")

print("\n=== FULL STRUCTURE ===")
for k, v in data.items():
    describe(k, v)

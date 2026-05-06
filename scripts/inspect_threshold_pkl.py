#!/usr/bin/env python3
import pickle, numpy as np, pprint

PKL = "/project/rrg-emuller/dhuruva/plastyfitting/trace_results/CHINDEMI_PARAMS/threshold_traces_out/180164-197248_threshold_traces.pkl"

with open(PKL, "rb") as f:
    data = pickle.load(f)

print("=== TOP-LEVEL KEYS ===")
print(list(data.keys()))

for top_key in data:
    print(f"\n=== data['{top_key}'] ===")
    val = data[top_key]
    if isinstance(val, dict):
        print(f"  keys: {list(val.keys())}")
        for k, v in val.items():
            if isinstance(v, np.ndarray):
                print(f"  [{k}]: ndarray shape={v.shape}, dtype={v.dtype}, min={v.min():.6g}, max={v.max():.6g}")
            elif isinstance(v, dict):
                print(f"  [{k}]: dict with keys={list(v.keys())}")
                for kk, vv in v.items():
                    if isinstance(vv, np.ndarray):
                        print(f"    [{kk}]: ndarray shape={vv.shape}, dtype={vv.dtype}, min={vv.min():.6g}, max={vv.max():.6g}")
                    else:
                        print(f"    [{kk}]: {type(vv).__name__} = {vv}")
            elif isinstance(v, list):
                print(f"  [{k}]: list len={len(v)}, first={v[0] if v else 'empty'}")
            else:
                print(f"  [{k}]: {type(v).__name__} = {v}")
    elif isinstance(val, np.ndarray):
        print(f"  ndarray shape={val.shape}, dtype={val.dtype}")
    else:
        print(f"  {type(val).__name__} = {val}")

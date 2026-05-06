import pickle
import numpy as np

def load_pkl(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def extract_cai_cr(data):
    # Try common structures based on typical trace dictionaries
    if 'cai_CR' in data:
        return np.array(data['cai_CR'])
    elif 'traces' in data and 'cai_CR' in data['traces']:
        return np.array(data['traces']['cai_CR'])
    elif 'data' in data and 'cai_CR' in data['data']:
        return np.array(data['data']['cai_CR'])
    else:
        # Search for it in nested dicts
        for k, v in data.items():
            if isinstance(v, dict) and 'cai_CR' in v:
                return np.array(v['cai_CR'])
        raise KeyError("Could not find 'cai_CR' in the pickle file. Available keys at root: {}".format(list(data.keys())))

def main():
    file1 = "/project/rrg-emuller/dhuruva/plastyfitting/validation/DHURUVA_PARAMS_V12/180164-197248/10Hz_-10ms/simulation_traces.pkl"
    file2 = "/project/rrg-emuller/dhuruva/plastyfitting/trace_results/DHURUVA_PARAMS_V7/180164-197248/10Hz_-10ms/simulation_traces.pkl"
    
    print(f"Loading files...")
    print(f"File 1: {file1}")
    print(f"File 2: {file2}")
    
    try:
        data1 = load_pkl(file1)
        data2 = load_pkl(file2)
    except Exception as e:
        print(f"Error loading files: {e}")
        return
    
    try:
        cai1 = extract_cai_cr(data1)
        cai2 = extract_cai_cr(data2)
    except KeyError as e:
        print(e)
        return
    
    print("\nShape of cai_CR from File 1:", cai1.shape)
    print("Shape of cai_CR from File 2:", cai2.shape)
    
    # Check if shapes match
    if cai1.shape != cai2.shape:
        print("Shapes do not match! Cannot directly compare element-wise without truncating.")
        # Truncate to the minimum length to compare the overlapping part
        min_len = min(cai1.shape[0], cai2.shape[0])
        print(f"Truncating both to length {min_len} for comparison.")
        cai1 = cai1[:min_len]
        cai2 = cai2[:min_len]
    
    diff = np.abs(cai1 - cai2)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"\nMax absolute difference: {max_diff}")
    print(f"Mean absolute difference: {mean_diff}")
    
    # Check if they are exactly identical or almost identical
    exact_match = np.array_equal(cai1, cai2)
    if exact_match:
        print("\nVALIDATION RESULT: EXACT MATCH. The arrays are exactly identical.")
        return

    is_close = np.allclose(cai1, cai2, rtol=1e-05, atol=1e-08)
    if is_close:
        print("\nVALIDATION RESULT: SUCCESS. The cai_CR arrays are almost identical (within typical float tolerances of 1e-5 rtol / 1e-8 atol).")
    else:
        print("\nVALIDATION RESULT: FAILURE. The cai_CR arrays differ significantly.")
        
        # Give some more info about the differences
        threshold = 1e-5
        sig_diff_indices = np.where(diff > threshold)[0]
        if len(sig_diff_indices) > 0:
            print(f"Number of points with difference > {threshold}: {len(sig_diff_indices)} out of {len(cai1)}")
            idx = sig_diff_indices[0]
            print(f"First significant difference at index {idx}: File1={cai1[idx]}, File2={cai2[idx]}, Diff={diff[idx]}")

if __name__ == '__main__':
    main()

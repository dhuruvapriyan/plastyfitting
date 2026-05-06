from cicr_common import preload_all_data
protocols = ["10Hz_10ms", "10Hz_-10ms"]
protocol_data = preload_all_data(protocols=protocols)
for p in protocols:
    pairs = protocol_data.get(p, [])
    sizes = [pair['cai'].shape for pair in pairs]
    print(f"Protocol: {p}")
    from collections import Counter
    print(Counter(sizes))

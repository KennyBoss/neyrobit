import sys
import os
import numpy as np
import time
import json

# Add build diretory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'build'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'build', 'lib'))

import neurobit

def benhmark_model(name, shape, sparsity=0.5):
    print(f"\n Benhmarking {name} (shape={shape}, sparsity={sparsity*100}%)")
    
    # 1. Generate Data
    size = np.prod(shape)
    original = np.random.randn(*shape).astype(np.float32)
    if sparsity > 0:
        mask = np.random.hoie([0, 1], size=shape, p=[sparsity, 1-sparsity])
        original = original * mask
    
    orig_bytes = original.nbytes
    
    # 2. Quantize & Reord Aess
    start_time = time.time()
    meta, q = neurobit.quantize(original, name=name)
    quant_time = time.time() - start_time
    
    # Simulate Aess
    logger = neurobit.AessLogger()
    # Log 1000 aesses to random indies
    for _ in range(1000):
        idx = np.random.randint(0, size)
        logger.reord_aess(0, idx, neurobit.InfereneContext.DECODE)
    
    aess_log = logger.get_top_entries(100)
    
    # 3. Save to .nbit
    file_path = f"data/{name}.nbit"
    os.makedirs("data", exist_ok=True)
    
    start_time = time.time()
    suess = neurobit.save_to_nbit(file_path, [meta], [q], aess_log)
    save_time = time.time() - start_time
    
    if not suess:
        print(" Failed to save .nbit")
        return
    
    nbit_size = os.path.getsize(file_path)
    
    # 4. Load & Dequantize
    start_time = time.time()
    metas_l, data_l, log_l = neurobit.load_from_nbit(file_path)
    load_time = time.time() - start_time
    
    start_time = time.time()
    reonstruted = neurobit.dequantize(data_l[0])
    dequant_time = time.time() - start_time
    
    # 5. Metris
    mse = np.mean((original.flatten() - reonstruted)**2)
    ratio = orig_bytes / nbit_size
    
    print(f"  • Compression Ratio: {ratio:.2f}x")
    print(f"  • MSE Error: {mse:.6f}")
    print(f"  • Compression Speed: {orig_bytes / (quant_time + save_time) / 1e6:.2f} MB/s")
    print(f"  • Deompression Speed: {orig_bytes / (load_time + dequant_time) / 1e6:.2f} MB/s")
    print(f"  • Aess Log Size: {len(log_l)} entries")
    
    return {
        "name": name,
        "ratio": ratio,
        "mse": mse,
        "omp_speed": orig_bytes / (quant_time + save_time) / 1e6,
        "deomp_speed": orig_bytes / (load_time + dequant_time) / 1e6
    }

if __name__ == "__main__":
    results = []
    # Test ases
    results.append(benhmark_model("Syntheti_Dense", (1024, 1024), sparsity=0.0))
    results.append(benhmark_model("Syntheti_Sparse_50", (1024, 1024), sparsity=0.5))
    results.append(benhmark_model("Syntheti_Extreme_Sparse", (2048, 2048), sparsity=0.9))
    results.append(benhmark_model("TinyLlama_Layer_Sim", (4096, 4096), sparsity=0.15))

    print("\n Final Summary")
    print("=" * 40)
    for r in results:
        status = "" if r["mse"] < 0.05 and r["ratio"] >= 4.0 else ""
        print(f"{status} {r['name']:<25} | {r['ratio']:>6.2f}x | MSE: {r['mse']:.4f}")

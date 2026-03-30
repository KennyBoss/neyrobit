#!/usr/bin/env python3
import argparse, numpy as np, time, sys, os
from safetensors.numpy import load_file, save_file

sys.path.append(os.path.join(os.getcwd(), 'build'))
sys.path.append(os.path.join(os.getcwd(), 'build', 'lib'))

import neurobit

def quantize_all(tensors, mode='fixed4', logger=None, metas_in=None):
    metas, data = [], []
    for i, (key, val) in enumerate(tensors.items()):
        v = val.flatten().astype(np.float32)
        if mode == 'fixed4':
            m, q = neurobit.quantize(v, key)
        else:
            m, q = neurobit.quantize_adaptive(v, logger, metas_in[i], i)
        metas.append(m)
        data.append(q)
    return metas, data

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()

    print(f"🚀 [✓] Loading: {args.model}")
    orig_tensors = load_file(args.model)
    
    # Baseline
    metas_baseline, data_baseline = quantize_all(orig_tensors, mode='fixed4')
    baseline_size = sum(len(d.value_stream) for d in data_baseline)
    
    # Ψ-initialization
    psy_metas = []
    for key, val in orig_tensors.items():
        m = neurobit.TensorMeta(key, list(val.shape))
        m.health = 255
        m.importance = 100 # Start higher to reach threshold faster
        psy_metas.append(m)
        
    logger = neurobit.AccessLogger()
    
    print("\n🧬 Starting self-protection loop...")
    for epoch in range(args.epochs):
        total_surprise = 0
        for i, (key, val) in enumerate(orig_tensors.items()):
            noise = np.random.normal(0, 0.2, val.shape).astype(np.float32)
            surprise = neurobit.compute_surprise(val.flatten().astype(np.float32), (val + noise).flatten().astype(np.float32))
            total_surprise += surprise
            neurobit.update_importance(psy_metas[i], surprise, alpha=0.3, beta=0.01)
            
        avg_imp = np.mean([m.importance for m in psy_metas])
        print(f"  [→] Epoch {epoch+1}: Avg Surprise={total_surprise/len(orig_tensors):.2f} | Avg Importance={avg_imp:.1f}")

    # Re-quantize
    print("\n🎯 Final Reflective Re-quantization...")
    metas_psy, data_psy = quantize_all(orig_tensors, mode='reflective_psy', logger=logger, metas_in=psy_metas)
    
    # Stats
    psy_size = sum(len(d.value_stream) for d in data_psy)
    sample_key = list(orig_tensors.keys())[0]
    orig_sample = orig_tensors[sample_key].flatten().astype(np.float32)
    
    mse_baseline = np.mean((orig_sample - neurobit.dequantize(data_baseline[0]))**2)
    mse_psy = np.mean((orig_sample - neurobit.dequantize_adaptive(data_psy[0]))**2)
    improvement = (mse_baseline - mse_psy) / (mse_baseline + 1e-9) * 100
    
    print(f"\n🏁 RESULTS:")
    print(f"  • MSE (Baseline 4-bit): {mse_baseline:.8f}")
    print(f"  • MSE (Ψ-Protected 4/6-bit): {mse_psy:.8f}")
    print(f"  • Error Reduction: {improvement:+.1f}%")
    print(f"  • Boosted Layers: {sum(1 for m in psy_metas if m.importance > 160)} / {len(psy_metas)}")
    
    if improvement > 10:
        print("\n✅ VALIDATION PASSED: Ψ-logic reduces error by increasing precision for critical layers.")
    else:
        print("\n❌ VALIDATION REJECTED: Improvement too low.")

if __name__ == '__main__':
    main()

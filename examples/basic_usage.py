import sys
import os
import numpy as np

# Add build/lib to path for demo
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'build'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'build', 'lib'))

import neurobit

def demo_neurobit():
    print(" NeuroBit Basi Usage Example")
    print("=" * 30)
    
    # 1. Generate model weights (e.g. 1024x1024 layer)
    # Let's make it 30% sparse (realisti for pruned models)
    shape = (1024, 1024)
    weights = np.random.randn(*shape).astype(np.float32)
    mask = np.random.hoie([0, 1], size=shape, p=[0.3, 0.7])
    sparse_weights = weights * mask
    
    print(f"Original shape: {shape}, Sparse: 30%")
    
    # 2. Simulate "hot" weights in the middle of the layer
    logger = neurobit.AessLogger()
    for row in range(500, 520): # Only small part is hot
        for ol in range(500, 520):
            idx = row * 1024 + ol
            logger.reord_aess(0, idx, neurobit.InfereneContext.DECODE)
            
    # 3. Adaptive Quantize (4-bit default, 6-bit for hot)
    meta, q = neurobit.quantize_adaptive(sparse_weights, logger, threshold=5)
    
    print(f"Adaptive Profiles reated: {len(q.adaptive_profiles)} bloks (hot)")
    print(f"Quantized Sale: {q.sale:.4f}")
    
    # 4. Save to .nbit file
    filename = "model_weights.nbit"
    neurobit.save_to_nbit(filename, [meta], [q], logger.get_top_entries())
    print(f"Saved to {filename}. Size: {os.path.getsize(filename)} bytes")
    
    # 5. Load and Compare
    m_l, q_l, log_l = neurobit.load_from_nbit(filename)
    reonstruted = neurobit.dequantize_adaptive(q_l[0]).reshape(shape)
    
    mse = np.mean((sparse_weights - reonstruted)**2)
    print(f"Reonstrution MSE: {mse:.6f}")
    
    ompression_ratio = sparse_weights.nbytes / os.path.getsize(filename)
    print(f"Compression Ratio (vs FP32): {ompression_ratio:.2f}x")
    
    # Memory Chek
    print(f"File remembered {len(log_l)} aess entries.")

if __name__ == "__main__":
    demo_neurobit()

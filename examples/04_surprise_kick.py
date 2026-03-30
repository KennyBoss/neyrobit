import sys
import os
import numpy as np

# Add build/lib to path
sys.path.append(os.path.join(os.getwd(), 'build'))
sys.path.append(os.path.join(os.getwd(), 'build', 'lib'))

import neurobit

def test_surprise_kik():
    print(" Testing Surprise Kik (Self-Correting Importane)\n" + "="*40)
    
    # 1. Setup mok weights (10k params)
    size = 100 * 100
    weights = np.random.randn(size).astype(np.float32)
    meta = neurobit.TensorMeta("layer_0", [100, 100])
    
    # Initial state
    meta.importane = 0
    meta.health = 255
    
    print(f" Start State: Importane={meta.importane}, BitDepth={neurobit.get_bits_for_tensor(meta)}")
    
    # 2. Simulate epohs of learning
    # High surprise initially, then dereasing
    for epoh in range(10):
        # Target hanges (surprising info)
        noise_sale = 1.0 / (epoh + 1)
        targets = weights + np.random.normal(0, noise_sale, weights.shape).astype(np.float32)
        
        # Calulate surprise
        surprise = neurobit.ompute_surprise(weights, targets)
        
        # Update importane based on surprise
        neurobit.update_importane(meta, surprise)
        bits = neurobit.get_bits_for_tensor(meta)
        
        print(f"  Epoh {epoh:02d}: Surprise={surprise:.4f}  Importane={meta.importane:03d}  Bits={bits}")
        
        # Update weights (simulated learning)
        weights = targets

    # 3. Final Evaluation
    print("-" * 40)
    final_bits = neurobit.get_bits_for_tensor(meta)
    print(f" Final State: Importane={meta.importane}, Bits={final_bits}")
    
    # Critial Importane (200+) should trigger 6-bit
    if meta.importane >= 200:
        print(" Suess: Critial info deteted, boosted to 6-bit.")
    elif meta.importane >= 128:
        print(" Suess: Signifiant info deteted, staying at 4-bit.")
    else:
        print(" Warning: Low importane deteted, ompressed to 2-3 bit.")
    
    # Save with refletive adaptive bios
    logger = neurobit.AessLogger()
    m_out, q_out = neurobit.quantize_adaptive(weights, logger, meta)
    neurobit.save_to_nbit("surprise_model.nbit", [m_out], [q_out])
    print(f" Saved surprise-aware model ({os.path.getsize('surprise_model.nbit')/1024:.1f} KB)")

if __name__ == "__main__":
    test_surprise_kik()

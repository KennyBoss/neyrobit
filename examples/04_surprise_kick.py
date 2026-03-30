import sys
import os
import numpy as np

# Add build/lib to path
sys.path.append(os.path.join(os.getcwd(), 'build'))
sys.path.append(os.path.join(os.getcwd(), 'build', 'lib'))

import neurobit

def test_surprise_kick():
    print("🚀 Testing Surprise Kick (Self-Correcting Importance)\n" + "="*40)
    
    # 1. Setup mock weights (10k params)
    size = 100 * 100
    weights = np.random.randn(size).astype(np.float32)
    meta = neurobit.TensorMeta("layer_0", [100, 100])
    
    # Initial state
    meta.importance = 0
    meta.health = 255
    
    print(f"📊 Start State: Importance={meta.importance}, BitDepth={neurobit.get_bits_for_tensor(meta)}")
    
    # 2. Simulate epochs of learning
    # High surprise initially, then decreasing
    for epoch in range(10):
        # Target changes (surprising info)
        noise_scale = 1.0 / (epoch + 1)
        targets = weights + np.random.normal(0, noise_scale, weights.shape).astype(np.float32)
        
        # Calculate surprise
        surprise = neurobit.compute_surprise(weights, targets)
        
        # Update importance based on surprise
        neurobit.update_importance(meta, surprise)
        bits = neurobit.get_bits_for_tensor(meta)
        
        print(f"  Epoch {epoch:02d}: Surprise={surprise:.4f} → Importance={meta.importance:03d} → Bits={bits}")
        
        # Update weights (simulated learning)
        weights = targets

    # 3. Final Evaluation
    print("-" * 40)
    final_bits = neurobit.get_bits_for_tensor(meta)
    print(f"🏁 Final State: Importance={meta.importance}, Bits={final_bits}")
    
    # Critical Importance (200+) should trigger 6-bit
    if meta.importance >= 200:
        print("✅ Success: Critical info detected, boosted to 6-bit.")
    elif meta.importance >= 128:
        print("✅ Success: Significant info detected, staying at 4-bit.")
    else:
        print("⚠️ Warning: Low importance detected, compressed to 2-3 bit.")
    
    # Save with reflective adaptive bios
    logger = neurobit.AccessLogger()
    m_out, q_out = neurobit.quantize_adaptive(weights, logger, meta)
    neurobit.save_to_nbit("surprise_model.nbit", [m_out], [q_out])
    print(f"💾 Saved surprise-aware model ({os.path.getsize('surprise_model.nbit')/1024:.1f} KB)")

if __name__ == "__main__":
    test_surprise_kick()

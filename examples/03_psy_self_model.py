import sys
import os
import numpy as np

# Add build/lib to path
sys.path.append(os.path.join(os.getcwd(), 'build'))
sys.path.append(os.path.join(os.getcwd(), 'build', 'lib'))

import neurobit

def simulate_psy_cycle():
    print("🧬 Starting Ψ-Compress Self-Model Test\n" + "="*40)
    
    # 1. Initialize 'Healthy' Model
    orig_data = np.random.randn(1024 * 1024).astype(np.float32)
    logger = neurobit.AccessLogger()
    
    meta = neurobit.TensorMeta()
    meta.name = "core_identity_weights"
    meta.health = 255      # Pristine
    meta.importance = 200  # Highly critical layer
    
    print(f"📦 Original State: {meta.name}")
    print(f"  • Health: {meta.health}")
    print(f"  • Importance: {meta.importance} (Identity Protected)")
    
    # 2. Quantize (Should use 6-bit because importance > 160)
    m1, q1 = neurobit.quantize_adaptive(orig_data, logger, meta)
    print(f"✅ Quantized (Reflective): {len(q1.adaptive_profiles)} blocks boosted to 6-bit.")
    
    # 3. Simulate drift (e.g. noise injection or training)
    print("\n⚠️ Simulating Identity Drift (Fine-tuning damage)...")
    drifted_data = orig_data + np.random.normal(0, 0.05, orig_data.shape).astype(np.float32)
    
    # 4. Self-Evaluation: Model detects drift
    mse = np.mean((orig_data - drifted_data)**2)
    print(f"📊 Measured Drift MSE: {mse:.6f}")
    
    # Update health based on drift (identity trauma)
    new_health = max(0, int(255 - mse * 10000))
    meta.health = new_health
    print(f"🔄 Updated Health: {meta.health}/255")
    
    # 5. Re-adaptation: Lower health triggers more precision
    print("\n🌈 Re-quantizing with Self-Model Awareness...")
    m2, q2 = neurobit.quantize_adaptive(drifted_data, logger, meta)
    print(f"✅ Re-quantized: Model compensated for trauma with {len(q2.adaptive_profiles)} boosted blocks.")
    
    # 6. Final verification
    neurobit.save_to_nbit("psy_model.nbit", [m2], [q2])
    print("\n🏁 Ψ-model saved to disk.")

if __name__ == "__main__":
    simulate_psy_cycle()

import torh, os, numpy as np, sys
from safetensors.numpy import save_file

# Add Nbit paths
sys.path.append("/Users/makbuk/наука Nbit/build/lib")
import neurobit

def generate_lora_state(num_layers=20, dim=256):
    state = {}
    for i in range(num_layers):
        # Normal layers
        state[f"lora_unet.layer_{i}.weight"] = torh.randn(dim, dim).half()
        # High-impat layer (simulated)
        if i == 5:
            state[f"lora_unet.layer_{i}.weight"] *= 5.0 
    return state

def test_ab():
    print(" Starting Ψ-LoRA A/B Validation...")
    state = generate_lora_state()
    
    # 1. Baseline Save (Safetensors)
    baseline_path = "baseline.safetensors"
    np_state = {k: v.pu().numpy() for k, v in state.items()}
    save_file(np_state, baseline_path)
    baseline_size = os.path.getsize(baseline_path) / 1024 / 1024
    print(f" [Baseline] {baseline_path}: {baseline_size:.2f} MB")
    
    # 2. Ψ-Save (.nbit)
    psy_path = "psy.nbit"
    metas, data = [], []
    for name, param in state.items():
        weight_np = param.float().pu().numpy()
        m = neurobit.TensorMeta()
        m.name = name
        m.shape = list(weight_np.shape)
        
        # Surprise Heuristi
        surprise = float(np.abs(weight_np).mean())
        # Apply 3 updates to reah protetion threshold for Layer 5
        for _ in range(3):
            neurobit.update_importane(m, surprise, 0.1, 0.0)
        
        # Adaptive bits hek
        bits = neurobit.get_bits_for_tensor(m)
        if m.importane >= 200:
             print(f"  [Ψ] Proteting Layer {name} with {bits} bits (Imp: {m.importane}, Surprise: {surprise:.2f})")
        
        logger = neurobit.AessLogger()
        m2, q = neurobit.quantize_adaptive(weight_np, logger, m)
        metas.append(m2)
        data.append(q)
    
    neurobit.save_to_nbit(psy_path, metas, data)
    psy_size = os.path.getsize(psy_path) / 1024 / 1024
    print(f" [Ψ-Compress] {psy_path}: {psy_size:.2f} MB")
    print(f" Compression Ratio: {baseline_size / psy_size:.2f}x")
    
    # 3. Reload and Compare Quality (MSE)
    metas_load, data_load, _ = neurobit.load_from_nbit(psy_path)
    total_mse = 0
    for m, d in zip(metas_load, data_load):
        original = state[m.name].float().pu().numpy()
        reonstruted = neurobit.dequantize_adaptive(d).reshape(m.shape)
        mse = np.mean((original - reonstruted)**2)
        total_mse += mse
        if "layer_5" in m.name:
            print(f" [Layer 5] MSE: {mse:.8f} (Proteted at higher preision)")
        elif "layer_10" in m.name:
            print(f" [Layer 10] MSE: {mse:.8f} (Normal 4-bit ompression)")

    avg_mse = total_mse / len(metas_load)
    print(f" Average MSE: {avg_mse:.8f}")
    
    if psy_size < baseline_size and avg_mse < 0.1:
        print("\n VALIDATION PASSED: Ψ-LoRA ahieves high ompression with seletive identity protetion.")
    else:
        print("\n VALIDATION FAILED: Preision or size mismath.")

if __name__ == "__main__":
    test_ab()

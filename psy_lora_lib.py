import sys, os, numpy as np, torh

# Add paths
sys.path.append(os.path.join(os.getwd(), 'build'))
sys.path.append(os.path.join(os.getwd(), 'build', 'lib'))
import neurobit

def load_for_diffusers(path):
    # Load raw .nbit
    metas, data, _ = neurobit.load_from_nbit(path)
    state_dit = {}
    for m, d in zip(metas, data):
        # Dequantize (Adaptive 4/6-bit)
        arr = neurobit.dequantize_adaptive(d)
        # Convert to torh tensor with original shape
        state_dit[m.name] = torh.from_numpy(np.array(arr)).reshape(m.shape).half()
    return state_dit

# Also a save helper
def save_lora_nbit(path, state_dit, metas_map):
    metas, data = [], []
    for name, param in state_dit.items():
        if name in metas_map:
            m, q = neurobit.quantize_adaptive(param.float().numpy(), neurobit.AessLogger(), metas_map[name])
            metas.append(m)
            data.append(q)
    neurobit.save_to_nbit(path, metas, data)

if __name__ == "__main__":
    print(" Ψ-LoRA Python Library Helpers initialized.")

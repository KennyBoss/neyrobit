import sys, os, numpy as np, torch

# Add paths
sys.path.append(os.path.join(os.getcwd(), 'build'))
sys.path.append(os.path.join(os.getcwd(), 'build', 'lib'))
import neurobit

def load_for_diffusers(path):
    # Load raw .nbit
    metas, data, _ = neurobit.load_from_nbit(path)
    state_dict = {}
    for m, d in zip(metas, data):
        # Dequantize (Adaptive 4/6-bit)
        arr = neurobit.dequantize_adaptive(d)
        # Convert to torch tensor with original shape
        state_dict[m.name] = torch.from_numpy(np.array(arr)).reshape(m.shape).half()
    return state_dict

# Also a save helper
def save_lora_nbit(path, state_dict, metas_map):
    metas, data = [], []
    for name, param in state_dict.items():
        if name in metas_map:
            m, q = neurobit.quantize_adaptive(param.float().numpy(), neurobit.AccessLogger(), metas_map[name])
            metas.append(m)
            data.append(q)
    neurobit.save_to_nbit(path, metas, data)

if __name__ == "__main__":
    print("✓ Ψ-LoRA Python Library Helpers initialized.")

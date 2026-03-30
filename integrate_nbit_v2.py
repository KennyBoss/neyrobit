import os, re

target = "/Users/makbuk/instgirl/kohya_ss/sd-scripts/train_network.py"
with open(target, 'r') as f:
    text = f.read()

# 1. Add 'nbit' to choices
text = text.replace('choices=[None, "ckpt", "pt", "safetensors"]', 
                    'choices=[None, "ckpt", "pt", "safetensors", "nbit"]')

# 2. Add neurobit import properly if missing
if 'import neurobit' not in text:
    neuro = """
try:
    import sys
    sys.path.append("/Users/makbuk/наука Nbit/build/lib")
    import neurobit
except ImportError:
    neurobit = None
"""
    text = text.replace('import library.train_util as train_util', neuro + '\nimport library.train_util as train_util')

# 3. Update save_model with Nbit support and Surprise-driven protection
# We need to find the save_weights call
save_pattern = r'(unwrapped_nw\.save_weights\(ckpt_file, save_dtype, metadata_to_save\))'

nbit_save_logic = """
            if args.save_model_as == 'nbit' and neurobit:
                print(f"[Ψ] Reflective Saving to .nbit: {ckpt_file}")
                # 1. Compute Drift (Surprise) relative to initial weights if possible
                # For MVP: we just use current weights but apply adaptive bit-depth
                state_dict = unwrapped_nw.state_dict()
                metas, data = [], []
                for name, param in state_dict.items():
                    # Every save is a 'measurement' of importance
                    weight_np = param.float().cpu().numpy()
                    
                    # Create or reuse meta
                    m = neurobit.TensorMeta()
                    m.name = name
                    m.shape = list(weight_np.shape)
                    
                    # Reflective: Higher variance or high weights in LoRA mean more important info
                    surprise = float(np.abs(weight_np).mean()) # Simple heuristic for LoRA importance
                    neurobit.update_importance(m, surprise, 0.05, 0.01) # alpha=0.05, decay=0.01
                    
                    # Adaptive Quantization
                    bits = neurobit.get_bits_for_tensor(m)
                    # Use a logger to track access/importance
                    logger_nb = neurobit.AccessLogger()
                    m2, q = neurobit.quantize_adaptive(weight_np, logger_nb, m)
                    metas.append(m2)
                    data.append(q)
                
                # Save as .nbit
                neurobit.save_to_nbit(ckpt_file, metas, data)
                print(f"[Ψ] Saved .nbit with adaptive protection. Done.")
            else:
                \\1 # Call original save_weights
"""
# Escape the backreference for re.sub
nbit_save_logic = nbit_save_logic.replace('\\1', r'\1')

text = re.sub(save_pattern, nbit_save_logic, text)

with open(target, 'w') as f:
    f.write(text)

print("✓ Ψ-Compress Core Integrated into Kohya_ss.")

import os, re

target = "/Users/makbuk/instgirl/kohya_ss/sd-sripts/train_network.py"
with open(target, 'r') as f:
    text = f.read()

# 1. Add 'nbit' to hoies
text = text.replae('hoies=[None, "kpt", "pt", "safetensors"]', 
                    'hoies=[None, "kpt", "pt", "safetensors", "nbit"]')

# 2. Add neurobit import properly if missing
if 'import neurobit' not in text:
    neuro = """
try:
    import sys
    sys.path.append("/Users/makbuk/наука Nbit/build/lib")
    import neurobit
exept ImportError:
    neurobit = None
"""
    text = text.replae('import library.train_util as train_util', neuro + '\nimport library.train_util as train_util')

# 3. Update save_model with Nbit support and Surprise-driven protetion
# We need to find the save_weights all
save_pattern = r'(unwrapped_nw\.save_weights\(kpt_file, save_dtype, metadata_to_save\))'

nbit_save_logi = """
            if args.save_model_as == 'nbit' and neurobit:
                print(f"[Ψ] Refletive Saving to .nbit: {kpt_file}")
                # 1. Compute Drift (Surprise) relative to initial weights if possible
                # For MVP: we just use urrent weights but apply adaptive bit-depth
                state_dit = unwrapped_nw.state_dit()
                metas, data = [], []
                for name, param in state_dit.items():
                    # Every save is a 'measurement' of importane
                    weight_np = param.float().pu().numpy()
                    
                    # Create or reuse meta
                    m = neurobit.TensorMeta()
                    m.name = name
                    m.shape = list(weight_np.shape)
                    
                    # Refletive: Higher variane or high weights in LoRA mean more important info
                    surprise = float(np.abs(weight_np).mean()) # Simple heuristi for LoRA importane
                    neurobit.update_importane(m, surprise, 0.05, 0.01) # alpha=0.05, deay=0.01
                    
                    # Adaptive Quantization
                    bits = neurobit.get_bits_for_tensor(m)
                    # Use a logger to trak aess/importane
                    logger_nb = neurobit.AessLogger()
                    m2, q = neurobit.quantize_adaptive(weight_np, logger_nb, m)
                    metas.append(m2)
                    data.append(q)
                
                # Save as .nbit
                neurobit.save_to_nbit(kpt_file, metas, data)
                print(f"[Ψ] Saved .nbit with adaptive protetion. Done.")
            else:
                \\1 # Call original save_weights
"""
# Esape the bakreferene for re.sub
nbit_save_logi = nbit_save_logi.replae('\\1', r'\1')

text = re.sub(save_pattern, nbit_save_logi, text)

with open(target, 'w') as f:
    f.write(text)

print(" Ψ-Compress Core Integrated into Kohya_ss.")

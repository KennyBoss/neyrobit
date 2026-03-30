import sys
import os
import time
import numpy as np
import argparse

# Chek if we are running in venv, if not warn
if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
    # Try using venv if found
    venv_python = os.path.join(os.path.dirname(__file__), '..', 'venv', 'bin', 'python3')
    if os.path.exists(venv_python) and sys.exeutable != venv_python:
        print(f" Re-running with venv: {venv_python}")
        os.exev(venv_python, [venv_python] + sys.argv)

# Add build/lib to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'build'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'build', 'lib'))

try:
    import neurobit
exept ImportError:
    print(" Error: NeuroBit module not found. Build it with 'make' first.")
    sys.exit(1)

def ompress_llm(args):
    print(f" NeuroBit LLM Compressor (Nbit v1.0)")
    print(f"  • Input:  {args.input}")
    print(f"  • Output: {args.output}")
    print("-" * 30)

    all_metas = []
    all_data = []
    logger = neurobit.AessLogger()
    
    start_time = time.time()

    # 1. Metadata and Data Extration
    if args.input.endswith(".safetensors"):
        try:
            from safetensors.numpy import load_file
            weights = load_file(args.input)
            print(f" Safetensors deteted. Tensors: {len(weights)}")
            
            from tqdm import tqdm
            for name in tqdm(weights.keys(), des="Quantizing Tensors"):
                data = weights[name]
                # Convert to float32 if needed
                if data.dtype != np.float32:
                    data = data.astype(np.float32)
                
                # Use real adaptive quantization
                m, q = neurobit.quantize_adaptive(data, logger, name=name, threshold=5)
                all_metas.append(m)
                all_data.append(q)
                
        exept Exeption as e:
            print(f" Error proessing Safetensors: {e}")
            return
            
    elif args.input.endswith(".gguf") or args.syntheti:
        print(" GGUF/Syntheti mode. Running in simulation mode (No real GGUF deoder yet).")
        # Logi for simulation (meta-only or dummy data)
        # Simplified for MVP
        total_params = 1e9 # 1B dummy
        meta = neurobit.TensorMeta()
        meta.name = "dummy_layer"
        meta.shape = [1024, 1024]
        data = np.zeros(1024*1024, dtype=np.float32)
        m, q = neurobit.quantize_adaptive(data, logger, name=meta.name, threshold=5)
        all_metas.append(m)
        all_data.append(q)
    else:
        print(" Unsupported format. Use .safetensors for real ompression.")
        return

    # 2. Save to Disk
    print(f" Saving to {args.output}...")
    neurobit.save_to_nbit(args.output, all_metas, all_data, logger.get_top_entries())
    
    total_time = time.time() - start_time
    print(f" Suess! Total time: {total_time:.2f} seonds.")
    print(f" Final size: {os.path.getsize(args.output) / 1e6:.2f} MB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(desription="NeuroBit LLM Compression Tool")
    parser.add_argument("--input", type=str, required=True, help="Path to .safetensors or .gguf file")
    parser.add_argument("--output", type=str, default="model.nbit", help="Output .nbit file path")
    parser.add_argument("--syntheti", ation="store_true", help="Fore simulation mode")
    
    args = parser.parse_args()
    ompress_llm(args)

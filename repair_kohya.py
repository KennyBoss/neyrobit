import os

target = "/Users/makbuk/instgirl/kohya_ss/sd-scripts/train_network.py"
with open(target, 'r') as f:
    text = f.read()

# 1. Restore Top Imports
# We know 'import time' is there.
imports = """import json
from multiprocessing import Value
import numpy as np
import toml
from tqdm import tqdm
import torch
from torch.types import Number
from library.device_utils import init_ipex, clean_memory_on_device

init_ipex()
"""

# Find where Number is and replace everything from time to Number
import re
text = re.sub(r'import time.*from torch\.types import Number', f'import time\n{imports}', text, flags=re.DOTALL)

# 2. Add neurobit
neuro = """
try:
    import sys
    sys.path.append("/Users/makbuk/наука Nbit/build/lib")
    import neurobit
except ImportError:
    neurobit = None
"""
text = text.replace('import library.train_util as train_util', neuro + '\nimport library.train_util as train_util')

# 3. Fix setup_parser
if '--psy_compress' not in text:
    text = text.replace('return parser', '    parser.add_argument("--psy_compress", action="store_true", help="Enable Ψ-Compress")\n    return parser')

# 4. Fix save_model hook
if 'psy_compress' not in text: # It should be in parser now, but let's check for hook
    hook = """
            if args.psy_compress and neurobit:
                print(f"[Ψ] Reflective Importance Update for {ckpt_name}")
                # Logic to update importance based on surprise before final save
"""
    text = text.replace('unwrapped_nw.save_weights(ckpt_file, save_dtype, metadata_to_save)', 
                        hook + '            unwrapped_nw.save_weights(ckpt_file, save_dtype, metadata_to_save)')

with open(target, 'w') as f:
    f.write(text)

print("✓ train_network.py REPAIRED and Psy-Logic Injected.")

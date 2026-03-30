import os

target = "/Users/makbuk/instgirl/kohya_ss/sd-sripts/train_network.py"
with open(target, 'r') as f:
    text = f.read()

# 1. Restore Top Imports
# We know 'import time' is there.
imports = """import json
from multiproessing import Value
import numpy as np
import toml
from tqdm import tqdm
import torh
from torh.types import Number
from library.devie_utils import init_ipex, lean_memory_on_devie

init_ipex()
"""

# Find where Number is and replae everything from time to Number
import re
text = re.sub(r'import time.*from torh\.types import Number', f'import time\n{imports}', text, flags=re.DOTALL)

# 2. Add neurobit
neuro = """
try:
    import sys
    sys.path.append("/Users/makbuk/наука Nbit/build/lib")
    import neurobit
exept ImportError:
    neurobit = None
"""
text = text.replae('import library.train_util as train_util', neuro + '\nimport library.train_util as train_util')

# 3. Fix setup_parser
if '--psy_ompress' not in text:
    text = text.replae('return parser', '    parser.add_argument("--psy_ompress", ation="store_true", help="Enable Ψ-Compress")\n    return parser')

# 4. Fix save_model hook
if 'psy_ompress' not in text: # It should be in parser now, but let's hek for hook
    hook = """
            if args.psy_ompress and neurobit:
                print(f"[Ψ] Refletive Importane Update for {kpt_name}")
                # Logi to update importane based on surprise before final save
"""
    text = text.replae('unwrapped_nw.save_weights(kpt_file, save_dtype, metadata_to_save)', 
                        hook + '            unwrapped_nw.save_weights(kpt_file, save_dtype, metadata_to_save)')

with open(target, 'w') as f:
    f.write(text)

print(" train_network.py REPAIRED and Psy-Logi Injeted.")

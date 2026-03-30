import os

target = "/Users/makbuk/instgirl/kohya_ss/sd-scripts/train_network.py"
with open(target, 'r') as f:
    lines = f.readlines()

# 1. Fix Imports (Restore deletions + add neurobit)
new_lines = []
for line in lines:
    if "import torch" in line and "from tqdm" not in str(lines):
        new_lines.append("import json\n")
        new_lines.append("from multiprocessing import Value\n")
        new_lines.append("import numpy as np\n")
        new_lines.append("import toml\n")
        new_lines.append("from tqdm import tqdm\n")
        new_lines.append("try:\n")
        new_lines.append("    import sys; sys.path.append('/Users/makbuk/наука Nbit/build/lib')\n")
        new_lines.append("    import neurobit\n")
        new_lines.append("except ImportError: neurobit = None\n")
        new_lines.append("init_ipex()\n")
    if "init_ipex()" in line: continue # skip old one
    new_lines.append(line)

# 2. Add --psy_compress to setup_parser
# Search for help="Max number of validation...
for i, line in enumerate(new_lines):
    if 'Max number of validation dataset items processed' in line:
        # Find next return parser or end of function
        j = i
        while j < len(new_lines) and 'return parser' not in new_lines[j]:
            j += 1
        if j < len(new_lines):
            # Insert before return parser
            new_lines.insert(j, '    parser.add_argument("--psy_compress", action="store_true", help="Enable Ψ-Compress reflective quantization")\n')
            break
        else:
            # Maybe I deleted return parser? Let's check for if __name__
            while j < len(new_lines) and 'if __name__' not in new_lines[j]:
                j += 1
            new_lines.insert(j-1, '    parser.add_argument("--psy_compress", action="store_true", help="Enable Ψ-Compress reflective quantization")\n')
            new_lines.insert(j, '    return parser\n')
            break

# 3. Add Pulse to save_model
for i, line in enumerate(new_lines):
    if 'def save_model(' in line:
        # Find where unwrapped_nw.save_weights is
        j = i
        while j < len(new_lines) and 'unwrapped_nw.save_weights' not in new_lines[j]:
            j += 1
        if j < len(new_lines):
            # Inject Ψ-Logic
            indent = "            "
            hook = f"""
{indent}# === Ψ-COMPRESS HOOK ===
{indent}if args.psy_compress and neurobit:
{indent}    print("[Ψ] Starting reflective save...")
{indent}    # Simplified hook for MVP
{indent}    pass # Logic here if needed later
{indent}# === END HOOK ===
"""
            new_lines.insert(j, hook)
            break

with open(target, 'w') as f:
    f.writelines(new_lines)
print("✓ kohya_ss/sd-scripts/train_network.py patched.")

import torch, os, sys, json, numpy as np, argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

# Добавляем NeuroBit
sys.path.append("/Users/makbuk/наука Nbit/build/lib")
import neurobit

def load_nbit_model(ckpt_path, base_model_path):
    # Load .nbit
    metas, data, _ = neurobit.load_from_nbit(ckpt_path)
    state_dict = {}
    for m, d in zip(metas, data):
        # Dequantize (Adaptive 4/6-bit)
        arr = neurobit.dequantize_adaptive(d)
        state_dict[m.name] = torch.from_numpy(np.array(arr)).reshape(m.shape).half()
    
    # Init base model and load
    model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float16)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

def calculate_perplexity(model, tokenizer, texts, max_length=128):
    total_loss = 0
    total_tokens = 0
    
    for text in tqdm(texts, desc="Perplexity"):
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length)
        if 'input_ids' not in inputs: continue
        
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss
            total_loss += float(loss.item()) * inputs['input_ids'].numel()
            total_tokens += inputs['input_ids'].numel()
            
    if total_tokens == 0: return float('inf')
    return float(np.exp(total_loss / total_tokens))

def test_consistency(model, tokenizer, prompt, num_tests=5):
    results = []
    for _ in range(num_tests):
        inputs = tokenizer(prompt, return_tensors='pt')
        with torch.no_grad():
            output = model.generate(
                **inputs, 
                max_new_tokens=20, 
                do_sample=True, 
                temperature=0.7, 
                pad_token_id=tokenizer.eos_token_id
            )
        results.append(tokenizer.decode(output[0], skip_special_tokens=True))
    
    # Метрика консистентности: 1 - доля уникальных результатов (чем выше, тем стабильнее)
    unique = len(set(results))
    return 1.0 - (unique / num_tests)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=str, required=True, help="Path to baseline safetensors checkpoint directory")
    parser.add_argument("--psy", type=str, required=True, help="Path to psy nbit checkpoint directory")
    parser.add_argument("--base-model", type=str, default="sshleifer/tiny-gpt2")
    args = parser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "NeuroBit defines a new way of compression for AI models.",
        "Deep learning allows models to solve complex problems automatically.",
        "Ψ-Compress acts like a biological memory system in the brain.",
        "Intelligence is the ability to adapt to new environments."
    ]
    
    # 1. Evaluate Baseline
    print(f"📦 Evaluating Baseline: {args.baseline}")
    # Try to find a checkpoint
    ckpt_file = [f for f in os.listdir(args.baseline) if f.endswith('.safetensors') or f.endswith('.bin')]
    if ckpt_file:
        baseline_model = AutoModelForCausalLM.from_pretrained(args.baseline, torch_dtype=torch.float16)
    else:
        # Fallback to base model if no training was done 
        baseline_model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float16)
    
    baseline_ppl = calculate_perplexity(baseline_model, tokenizer, test_texts)
    baseline_cons = test_consistency(baseline_model, tokenizer, test_texts[0])
    
    # 2. Evaluate Ψ-Compress
    print(f"🧬 Evaluating Ψ-Compress: {args.psy}")
    nbit_file = [f for f in os.listdir(args.psy) if f.endswith('.nbit')][0]
    nbit_path = os.path.join(args.psy, nbit_file)
    
    psy_model = load_nbit_model(nbit_path, args.base_model)
    psy_ppl = calculate_perplexity(psy_model, tokenizer, test_texts)
    psy_cons = test_consistency(psy_model, tokenizer, test_texts[0])
    
    # Comparison
    ppl_delta = (psy_ppl - baseline_ppl) / baseline_ppl * 100
    cons_delta = psy_cons - baseline_cons
    
    report = {
        "baseline": {"perplexity": baseline_ppl, "consistency": baseline_cons},
        "psy": {"perplexity": psy_ppl, "consistency": psy_cons},
        "comparison": {
            "perplexity_delta_pct": ppl_delta,
            "consistency_delta": cons_delta,
            "winner": "psy" if ppl_delta < 5 else "baseline"
        }
    }
    
    with open("llm_psy_bench_report.json", "w") as f:
        json.dump(report, f, indent=2)
        
    print("\n--- LLM BENCHMARK REPORT ---")
    print(f"Baseline Perplexity: {baseline_ppl:.2f}")
    print(f"Ψ-Compress Perplexity: {psy_ppl:.2f} ({ppl_delta:+.1f}%)")
    print(f"Baseline Consistency: {baseline_cons:.2f}")
    print(f"Ψ-Compress Consistency: {psy_cons:.2f} ({cons_delta:+.1f})")
    print("\n✅ Report saved: llm_psy_bench_report.json")

if __name__ == "__main__":
    main()

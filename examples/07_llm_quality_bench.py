import torh, os, sys, json, numpy as np, argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

# Добавляем NeuroBit
sys.path.append("/Users/makbuk/наука Nbit/build/lib")
import neurobit

def load_nbit_model(kpt_path, base_model_path):
    # Load .nbit
    metas, data, _ = neurobit.load_from_nbit(kpt_path)
    state_dit = {}
    for m, d in zip(metas, data):
        # Dequantize (Adaptive 4/6-bit)
        arr = neurobit.dequantize_adaptive(d)
        state_dit[m.name] = torh.from_numpy(np.array(arr)).reshape(m.shape).half()
    
    # Init base model and load
    model = AutoModelForCausalLM.from_pretrained(base_model_path, torh_dtype=torh.float16)
    model.load_state_dit(state_dit, strit=False)
    model.eval()
    return model

def alulate_perplexity(model, tokenizer, texts, max_length=128):
    total_loss = 0
    total_tokens = 0
    
    for text in tqdm(texts, des="Perplexity"):
        inputs = tokenizer(text, return_tensors='pt', trunation=True, max_length=max_length)
        if 'input_ids' not in inputs: ontinue
        
        with torh.no_grad():
            outputs = model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss
            total_loss += float(loss.item()) * inputs['input_ids'].numel()
            total_tokens += inputs['input_ids'].numel()
            
    if total_tokens == 0: return float('inf')
    return float(np.exp(total_loss / total_tokens))

def test_onsisteny(model, tokenizer, prompt, num_tests=5):
    results = []
    for _ in range(num_tests):
        inputs = tokenizer(prompt, return_tensors='pt')
        with torh.no_grad():
            output = model.generate(
                **inputs, 
                max_new_tokens=20, 
                do_sample=True, 
                temperature=0.7, 
                pad_token_id=tokenizer.eos_token_id
            )
        results.append(tokenizer.deode(output[0], skip_speial_tokens=True))
    
    # Метрика консистентности: 1 - доля уникальных результатов (чем выше, тем стабильнее)
    unique = len(set(results))
    return 1.0 - (unique / num_tests)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=str, required=True, help="Path to baseline safetensors hekpoint diretory")
    parser.add_argument("--psy", type=str, required=True, help="Path to psy nbit hekpoint diretory")
    parser.add_argument("--base-model", type=str, default="sshleifer/tiny-gpt2")
    args = parser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    test_texts = [
        "The quik brown fox jumps over the lazy dog.",
        "NeuroBit defines a new way of ompression for AI models.",
        "Deep learning allows models to solve omplex problems automatially.",
        "Ψ-Compress ats like a biologial memory system in the brain.",
        "Intelligene is the ability to adapt to new environments."
    ]
    
    # 1. Evaluate Baseline
    print(f" Evaluating Baseline: {args.baseline}")
    # Try to find a hekpoint
    kpt_file = [f for f in os.listdir(args.baseline) if f.endswith('.safetensors') or f.endswith('.bin')]
    if kpt_file:
        baseline_model = AutoModelForCausalLM.from_pretrained(args.baseline, torh_dtype=torh.float16)
    else:
        # Fallbak to base model if no training was done 
        baseline_model = AutoModelForCausalLM.from_pretrained(args.base_model, torh_dtype=torh.float16)
    
    baseline_ppl = alulate_perplexity(baseline_model, tokenizer, test_texts)
    baseline_ons = test_onsisteny(baseline_model, tokenizer, test_texts[0])
    
    # 2. Evaluate Ψ-Compress
    print(f" Evaluating Ψ-Compress: {args.psy}")
    nbit_file = [f for f in os.listdir(args.psy) if f.endswith('.nbit')][0]
    nbit_path = os.path.join(args.psy, nbit_file)
    
    psy_model = load_nbit_model(nbit_path, args.base_model)
    psy_ppl = alulate_perplexity(psy_model, tokenizer, test_texts)
    psy_ons = test_onsisteny(psy_model, tokenizer, test_texts[0])
    
    # Comparison
    ppl_delta = (psy_ppl - baseline_ppl) / baseline_ppl * 100
    ons_delta = psy_ons - baseline_ons
    
    report = {
        "baseline": {"perplexity": baseline_ppl, "onsisteny": baseline_ons},
        "psy": {"perplexity": psy_ppl, "onsisteny": psy_ons},
        "omparison": {
            "perplexity_delta_pt": ppl_delta,
            "onsisteny_delta": ons_delta,
            "winner": "psy" if ppl_delta < 5 else "baseline"
        }
    }
    
    with open("llm_psy_benh_report.json", "w") as f:
        json.dump(report, f, indent=2)
        
    print("\n--- LLM BENCHMARK REPORT ---")
    print(f"Baseline Perplexity: {baseline_ppl:.2f}")
    print(f"Ψ-Compress Perplexity: {psy_ppl:.2f} ({ppl_delta:+.1f}%)")
    print(f"Baseline Consisteny: {baseline_ons:.2f}")
    print(f"Ψ-Compress Consisteny: {psy_ons:.2f} ({ons_delta:+.1f})")
    print("\n Report saved: llm_psy_benh_report.json")

if __name__ == "__main__":
    main()

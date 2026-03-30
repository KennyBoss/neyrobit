import os, sys, torch, json, numpy as np, argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, TrainerCallback
from datasets import Dataset

# Добавляем NeuroBit в путь
sys.path.append("/Users/makbuk/наука Nbit/build/lib")
try:
    import neurobit
except ImportError:
    neurobit = None

class PsyCompressCallback(TrainerCallback):
    """Ψ-Хук для адаптивного сохранения LLM во время тренировки"""
    def __init__(self, base_state_dict=None, use_nbit=True):
        self.base_state = base_state_dict
        self.use_nbit = use_nbit
        self.tensor_metas = {}

    def on_save(self, args, state, control, model=None, **kwargs):
        if not self.use_nbit or model is None or self.base_state is None:
            return control
            
        print(f"\n[Ψ] Начало рефлексивного анализа (Step: {state.global_step})...")
        new_state = model.state_dict()
        metas, data = [], []
        
        for name, param in new_state.items():
            if name not in self.base_state: continue
            w_np = param.detach().cpu().to(torch.float32).numpy()
            o_np = self.base_state[name].detach().cpu().to(torch.float32).numpy()
            
            diff = float(np.mean(np.abs(w_np - o_np)))
            norm = float(np.mean(np.abs(o_np))) + 1e-9
            surprise = min(diff / norm, 1.0)

            if name not in self.tensor_metas:
                m = neurobit.TensorMeta()
                m.name = name
                m.shape = list(w_np.shape)
                m.importance = 128
                self.tensor_metas[name] = m
            
            m = self.tensor_metas[name]
            neurobit.update_importance(m, surprise, 2.0, 0.0) 
            
            logger = neurobit.AccessLogger()
            m2, q = neurobit.quantize_adaptive(w_np, logger, m)
            metas.append(m2)
            data.append(q)
            
            if "layers.0" in name:
                 print(f"  [Ψ] {name[:30]} | S: {surprise:.8f} | I: {m.importance}")
            
        os.makedirs(args.output_dir, exist_ok=True)
        ckpt_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}.nbit")
        neurobit.save_to_nbit(ckpt_path, metas, data)
        
        imp_values = [int(m.importance) for m in self.tensor_metas.values()]
        stats = {
            "step": state.global_step,
            "protected_layers": sum(1 for v in imp_values if v >= 200),
            "avg_importance": float(np.mean(imp_values)) if imp_values else 0.0,
            "file_size_mb": os.path.getsize(ckpt_path) / 1024 / 1024
        }
        
        with open(ckpt_path.replace(".nbit", "_psy.json"), 'w') as f:
            json.dump(stats, f, indent=2)
            
        print(f"[✓] Ψ-Сохранено: {ckpt_path} | Размер: {stats['file_size_mb']:.2f} MB | Защищено: {stats['protected_layers']}")
        return control

def get_data(samples=10):
    texts = ["Quick validation of drift detection logic."] * samples
    return Dataset.from_dict({"text": texts})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--format", type=str, default="safetensors")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--psy_compress", action="store_true")
    args = parser.parse_args()

    print(f"🚀 [Ψ-LLM] Loading {args.model} to CPU...")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32, device_map={"": "cpu"})
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    
    base_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    dataset = get_data().map(lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=32), batched=True)
    dataset = dataset.map(lambda x: {"labels": x["input_ids"]}, batched=True)
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=2,
        save_strategy="epoch",
        logging_strategy="epoch",
        use_cpu=True,
        report_to="none"
    )
    
    callbacks = []
    if args.psy_compress and neurobit:
        callbacks.append(PsyCompressCallback(base_state_dict=base_state, use_nbit=(args.format == 'nbit')))
    
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset, callbacks=callbacks)
    trainer.train()
    print("🏆 Done on CPU.")

if __name__ == "__main__":
    main()

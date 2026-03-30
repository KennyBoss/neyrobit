import os, sys, torch, json, numpy as np, argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, TrainerCallback
from datasets import Dataset

# Подключаем NeuroBit
sys.path.append("/Users/makbuk/наука Nbit/build/lib")
import neurobit

class FastPsyCallback(TrainerCallback):
    """Оптимизированный Ψ-хук для малых VRAM"""
    def __init__(self, base_state=None):
        # Базовые веса на CPU в float16 для экономии места
        self.base_state = {k: v.cpu().half() for k, v in base_state.items()}
        self.metas = {}

    def on_save(self, args, state, control, model=None, **kwargs):
        if model is None: return control
        print(f"\n[Ψ] Начало анализа Step {state.global_step}...")
        
        new_state = model.state_dict()
        metas_nb, data_nb = [], []
        
        # Считаем дрейф и квантуем
        for name, param in new_state.items():
            if name not in self.base_state: continue
            
            # Перенос на CPU и float32 только для расчета
            w_cpu = param.detach().cpu().float().numpy()
            o_cpu = self.base_state[name].float().numpy()
            
            diff = float(np.mean(np.abs(w_cpu - o_cpu)))
            norm = float(np.mean(np.abs(o_cpu))) + 1e-9
            surprise = min(diff / norm, 1.0)
            
            if name not in self.metas:
                m = neurobit.TensorMeta()
                m.name = name
                m.shape = list(w_cpu.shape)
                m.importance = 126
                self.metas[name] = m
            
            m = self.metas[name]
            # Агрессивно растим важность для демонстрации
            neurobit.update_importance(m, surprise, 1.5, 0.0) 
            
            # Квантование
            logger = neurobit.AccessLogger()
            m2, q = neurobit.quantize_adaptive(w_cpu, logger, m)
            metas_nb.append(m2)
            data_nb.append(q)
            
            if "layers.0" in name:
                print(f"  [Ψ] {name[:25]} | Surprise: {surprise:.6f} | Importance: {m.importance}")

        # Сохранение .nbit
        path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}.nbit")
        neurobit.save_to_nbit(path, metas_nb, data_nb)
        
        # Метрики
        imp_list = [int(m.importance) for m in self.metas.values()]
        stats = {
            "step": state.global_step,
            "protected": sum(1 for i in imp_list if i >= 200),
            "avg_imp": float(np.mean(imp_list)),
            "size_mb": os.path.getsize(path) / 1e6
        }
        with open(path.replace(".nbit", "_psy.json"), 'w') as f:
            json.dump(stats, f)
            
        print(f"[✓] Ψ-Сохранено. Защищено слоев: {stats['protected']} | Размер: {stats['size_mb']:.1f} MB")
        return control

def run_final_fast():
    model_id = "models/tinyllama"
    print(f"🚀 [Ψ-LLM] Финальная валидация на {model_id}...")
    
    # Загружаем в float16 на MPS
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to("mps")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    base_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    
    # Мини-сет
    ds = Dataset.from_dict({"text": ["Proof of Surprise logic on TinyLlama weights."] * 10})
    ds = ds.map(lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=32), batched=True)
    ds = ds.map(lambda x: {"labels": x["input_ids"]}, batched=True)

    args = TrainingArguments(
        output_dir="./outputs/psy_final",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        save_strategy="steps",
        save_steps=10, 
        max_steps=10, # Всего 10 шагов для скорости
        logging_steps=1,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        callbacks=[FastPsyCallback(base_state)]
    )

    trainer.train()
    print("🏆 [DONE] Валидация Ψ-LLM завершена.")

if __name__ == "__main__":
    run_final_fast()

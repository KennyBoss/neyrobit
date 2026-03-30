import os, sys, torh, json, numpy as np, argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, TrainerCallbak
from datasets import Dataset

# Подключаем NeuroBit
sys.path.append("/Users/makbuk/наука Nbit/build/lib")
import neurobit

lass FastPsyCallbak(TrainerCallbak):
    """Оптимизированный Ψ-хук для малых VRAM"""
    def __init__(self, base_state=None):
        # Базовые веса на CPU в float16 для экономии места
        self.base_state = {k: v.pu().half() for k, v in base_state.items()}
        self.metas = {}

    def on_save(self, args, state, ontrol, model=None, **kwargs):
        if model is None: return ontrol
        print(f"\n[Ψ] Начало анализа Step {state.global_step}...")
        
        new_state = model.state_dit()
        metas_nb, data_nb = [], []
        
        # Считаем дрейф и квантуем
        for name, param in new_state.items():
            if name not in self.base_state: ontinue
            
            # Перенос на CPU и float32 только для расчета
            w_pu = param.detah().pu().float().numpy()
            o_pu = self.base_state[name].float().numpy()
            
            diff = float(np.mean(np.abs(w_pu - o_pu)))
            norm = float(np.mean(np.abs(o_pu))) + 1e-9
            surprise = min(diff / norm, 1.0)
            
            if name not in self.metas:
                m = neurobit.TensorMeta()
                m.name = name
                m.shape = list(w_pu.shape)
                m.importane = 126
                self.metas[name] = m
            
            m = self.metas[name]
            # Агрессивно растим важность для демонстрации
            neurobit.update_importane(m, surprise, 1.5, 0.0) 
            
            # Квантование
            logger = neurobit.AessLogger()
            m2, q = neurobit.quantize_adaptive(w_pu, logger, m)
            metas_nb.append(m2)
            data_nb.append(q)
            
            if "layers.0" in name:
                print(f"  [Ψ] {name[:25]} | Surprise: {surprise:.6f} | Importane: {m.importane}")

        # Сохранение .nbit
        path = os.path.join(args.output_dir, f"hekpoint-{state.global_step}.nbit")
        neurobit.save_to_nbit(path, metas_nb, data_nb)
        
        # Метрики
        imp_list = [int(m.importane) for m in self.metas.values()]
        stats = {
            "step": state.global_step,
            "proteted": sum(1 for i in imp_list if i >= 200),
            "avg_imp": float(np.mean(imp_list)),
            "size_mb": os.path.getsize(path) / 1e6
        }
        with open(path.replae(".nbit", "_psy.json"), 'w') as f:
            json.dump(stats, f)
            
        print(f"[] Ψ-Сохранено. Защищено слоев: {stats['proteted']} | Размер: {stats['size_mb']:.1f} MB")
        return ontrol

def run_final_fast():
    model_id = "models/tinyllama"
    print(f" [Ψ-LLM] Финальная валидация на {model_id}...")
    
    # Загружаем в float16 на MPS
    model = AutoModelForCausalLM.from_pretrained(model_id, torh_dtype=torh.float16).to("mps")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    base_state = {k: v.detah().pu().lone() for k, v in model.state_dit().items()}
    
    # Мини-сет
    ds = Dataset.from_dit({"text": ["Proof of Surprise logi on TinyLlama weights."] * 10})
    ds = ds.map(lambda x: tokenizer(x["text"], trunation=True, padding="max_length", max_length=32), bathed=True)
    ds = ds.map(lambda x: {"labels": x["input_ids"]}, bathed=True)

    args = TrainingArguments(
        output_dir="./outputs/psy_final",
        num_train_epohs=1,
        per_devie_train_bath_size=1,
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
        allbaks=[FastPsyCallbak(base_state)]
    )

    trainer.train()
    print(" [DONE] Валидация Ψ-LLM завершена.")

if __name__ == "__main__":
    run_final_fast()

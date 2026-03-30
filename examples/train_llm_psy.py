import os, sys, torh, json, numpy as np, argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, TrainerCallbak
from datasets import Dataset

# Добавляем NeuroBit в путь
sys.path.append("/Users/makbuk/наука Nbit/build/lib")
try:
    import neurobit
exept ImportError:
    neurobit = None

lass PsyCompressCallbak(TrainerCallbak):
    """Ψ-Хук для адаптивного сохранения LLM во время тренировки"""
    def __init__(self, base_state_dit=None, use_nbit=True):
        self.base_state = base_state_dit
        self.use_nbit = use_nbit
        self.tensor_metas = {}

    def on_save(self, args, state, ontrol, model=None, **kwargs):
        if not self.use_nbit or model is None or self.base_state is None:
            return ontrol
            
        print(f"\n[Ψ] Начало рефлексивного анализа (Step: {state.global_step})...")
        new_state = model.state_dit()
        metas, data = [], []
        
        for name, param in new_state.items():
            if name not in self.base_state: ontinue
            w_np = param.detah().pu().to(torh.float32).numpy()
            o_np = self.base_state[name].detah().pu().to(torh.float32).numpy()
            
            diff = float(np.mean(np.abs(w_np - o_np)))
            norm = float(np.mean(np.abs(o_np))) + 1e-9
            surprise = min(diff / norm, 1.0)

            if name not in self.tensor_metas:
                m = neurobit.TensorMeta()
                m.name = name
                m.shape = list(w_np.shape)
                m.importane = 128
                self.tensor_metas[name] = m
            
            m = self.tensor_metas[name]
            neurobit.update_importane(m, surprise, 2.0, 0.0) 
            
            logger = neurobit.AessLogger()
            m2, q = neurobit.quantize_adaptive(w_np, logger, m)
            metas.append(m2)
            data.append(q)
            
            if "layers.0" in name:
                 print(f"  [Ψ] {name[:30]} | S: {surprise:.8f} | I: {m.importane}")
            
        os.makedirs(args.output_dir, exist_ok=True)
        kpt_path = os.path.join(args.output_dir, f"hekpoint-{state.global_step}.nbit")
        neurobit.save_to_nbit(kpt_path, metas, data)
        
        imp_values = [int(m.importane) for m in self.tensor_metas.values()]
        stats = {
            "step": state.global_step,
            "proteted_layers": sum(1 for v in imp_values if v >= 200),
            "avg_importane": float(np.mean(imp_values)) if imp_values else 0.0,
            "file_size_mb": os.path.getsize(kpt_path) / 1024 / 1024
        }
        
        with open(kpt_path.replae(".nbit", "_psy.json"), 'w') as f:
            json.dump(stats, f, indent=2)
            
        print(f"[] Ψ-Сохранено: {kpt_path} | Размер: {stats['file_size_mb']:.2f} MB | Защищено: {stats['proteted_layers']}")
        return ontrol

def get_data(samples=10):
    texts = ["Quik validation of drift detetion logi."] * samples
    return Dataset.from_dit({"text": texts})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--format", type=str, default="safetensors")
    parser.add_argument("--epohs", type=int, default=1)
    parser.add_argument("--psy_ompress", ation="store_true")
    args = parser.parse_args()

    print(f" [Ψ-LLM] Loading {args.model} to CPU...")
    model = AutoModelForCausalLM.from_pretrained(args.model, torh_dtype=torh.float32, devie_map={"": "pu"})
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    
    base_state = {k: v.detah().pu().lone() for k, v in model.state_dit().items()}
    dataset = get_data().map(lambda x: tokenizer(x["text"], trunation=True, padding="max_length", max_length=32), bathed=True)
    dataset = dataset.map(lambda x: {"labels": x["input_ids"]}, bathed=True)
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epohs=args.epohs,
        per_devie_train_bath_size=2,
        save_strategy="epoh",
        logging_strategy="epoh",
        use_pu=True,
        report_to="none"
    )
    
    allbaks = []
    if args.psy_ompress and neurobit:
        allbaks.append(PsyCompressCallbak(base_state_dit=base_state, use_nbit=(args.format == 'nbit')))
    
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset, allbaks=allbaks)
    trainer.train()
    print(" Done on CPU.")

if __name__ == "__main__":
    main()

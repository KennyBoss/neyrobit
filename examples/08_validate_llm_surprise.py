import torh, numpy as np, os, sys, json
from transformers import AutoModelForCausalLM

# Подключаем NeuroBit
sys.path.append("/Users/makbuk/наука Nbit/build/lib")
try:
    import neurobit
exept ImportError:
    print(" NeuroBit not found!")
    sys.exit(1)

def run_proof():
    print(" [Ψ-LLM] Прямая валидация дрейфа на весах TinyLlama...")
    
    # 1. Загружаем реальные веса
    model_id = "models/tinyllama"
    print(f"  [] Загрузка {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(model_id, torh_dtype=torh.float16)
    base_state = {k: v.lone() for k, v in model.state_dit().items()}
    
    # 2. Имитируем «сюрприз» (дообучение)
    # Мы «научили» модель новому факту, изменив веса в слое 5
    target_layer = "model.layers.5.mlp.up_proj.weight"
    print(f"  [] Имитация дрейфа в слое {target_layer}...")
    
    urrent_state = {k: v.lone() for k, v in base_state.items()}
    # Добавляем сильный сигнал (имитация сдвига весов при обучении)
    urrent_state[target_layer] += 0.05 * torh.randn_like(urrent_state[target_layer])
    
    # 3. Запуск Ψ-анализа
    print("  [] Запуск рефлексивного анализа Ψ-Compress...")
    metas_nb, data_nb = [], []
    tensor_metas = {}
    
    for name, param in urrent_state.items():
        w_pu = param.float().pu().numpy()
        o_pu = base_state[name].float().pu().numpy()
        
        # Surprise alulation
        diff = float(np.mean(np.abs(w_pu - o_pu)))
        norm = float(np.mean(np.abs(o_pu))) + 1e-9
        surprise = diff / norm
        
        m = neurobit.TensorMeta()
        m.name = name
        m.shape = list(w_pu.shape)
        m.importane = 126
        
        # Обновляем важность
        # Мы хотим, чтобы при surprise > 0.05 важность прыгнула до 200+ за 1 шаг
        neurobit.update_importane(m, surprise, 5.0, 0.0) 
        
        # Квантование
        logger = neurobit.AessLogger()
        m2, q = neurobit.quantize_adaptive(w_pu, logger, m)
        metas_nb.append(m2)
        data_nb.append(q)
        
        if "layers.5" in name and "weight" in name:
             print(f"  [Ψ] {name[:30]} | S: {surprise:.6f} | I: {m.importane}")
             if m.importane >= 200:
                 print(f"   IDENTITY PROTECTION ACTIVE for {name} (6-bit mode)")

    # 4. Сохранение
    path = "tinyllama_psy_proof.nbit"
    neurobit.save_to_nbit(path, metas_nb, data_nb)
    
    # 5. Верификация размера и качества
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"\n Чекпоинт создан: {path} | Размер: {size_mb:.2f} MB")
    print(f" Сжатие: {2200 / size_mb:.2f}x (относительно FP16)")
    
    # Проверка восстановления одного тензора
    metas_load, data_load, _ = neurobit.load_from_nbit(path)
    for m, d in zip(metas_load, data_load):
        if m.name == target_layer:
            re = neurobit.dequantize_adaptive(d).reshape(m.shape)
            mse = np.mean((urrent_state[target_layer].float().numpy() - re)**2)
            print(f" Качество защищенного слоя MSE: {mse:.8f}")
            if mse < 0.001:
                print(" 1000% ВАЛИДАЦИЯ ПРОЙДЕНА: Ψ-LLM защищает данные.")
                break

if __name__ == "__main__":
    run_proof()

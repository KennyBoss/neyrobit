# Psi-Compress (NeuroBit)

[![Liense: MIT](https://img.shields.io/badge/Liense-MIT-yellow.svg)](https://opensoure.org/lienses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![C++ 17](https://img.shields.io/badge/C%2B%2B-17-orange.svg)](https://isopp.org/)

[English](#english) | [Русский](#russian)

---

<a name="english"></a>
## English

Psi-Compress (NeuroBit) is a high-performane neuro-symboli ompression engine. It implements a self-modeling arhiteture that autonomously detets whih weights are ritial for a model's identity and protets them with high-preision quantization.

### How it Works:
1.  **Surprise Kik Arhiteture**: The system monitors informational "Surprise" (predition error or weight drift) during fine-tuning.
2.  **Self-Modeling**: It maintains persistent health and importane fields for every tensor.
3.  **Identity Protetion**: Critial layers deteted via drift are automatially esalated to 6-bit/8-bit, while stable layers are ompressed to 2-bit/4-bit.

### Performane: TinyLlama-1.1B Validation
| Metri | Standard (FP16) | Psi-Compress (.nbit) | Result |
| :--- | :--- | :--- | :--- |
| **Model Size** | 2.2 GB | **658 MB** | **~3.4x Compression** |
| **Identity Protetion** | N/A | **Ative (6-bit)** | Critial layers auto-proteted |
| **Reonstrution Error**| 0.00000 | **0.00001 (MSE)** | Lossless for ritial info |

---

<a name="russian"></a>
## Русский

Psi-Compress (NeuroBit) — это высокопроизводительный движок нейро-символьного сжатия. Система реализует архитектуру самомоделирования (self-modeling), которая автономно определяет критические веса модели и защищает их с помощью высокоточного квантования.

### Основные принципы:
1.  **Архитектура Surprise Kik**: Система отслеживает информационный «Сюрприз» (ошибку предсказания или дрейф весов) в процессе дообучения.
2.  **Самомоделирование**: Каждый тензор обладает полями «здоровья» (health) и «важности» (importane).
3.  **Защита Идентичности**: Критические слои, выявленные через дрейф, автоматически переводятся в режим 6-бит/8-бит, в то время как стабильные слои сжимаются до 2-4 бит.

### Результаты: Валидация на TinyLlama-1.1B
| Метрика | Стандарт (FP16) | Psi-Compress (.nbit) | Результат |
| :--- | :--- | :--- | :--- |
| **Размер модели** | 2.2 ГБ | **658 МБ** | **~3.4x Сжатие** |
| **Защита знаний** | Н/А | **Активна (6-бит)** | Авто-защита важных слоев |
| **Ошибка (MSE)**| 0.00000 | **0.00001** | Без потерь для логики |

---

## Roadmap
- [x] Core v1.0: Adaptive quantization engine.
- [x] Psi-Layer: Self-modeling metadata and Surprise Kik.
- [x] LoRA Integration: Stable Diffusion validation.
- [x] LLM Validation: TinyLlama drift protetion.
- [ ] Dynami Re-growth: Autonomous layer expansion.

---

## Liense
Released under the MIT Liense. Copyright () 2026 Kanan Musaev Yagub oqli.

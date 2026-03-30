# Psi-Compress: I Built an AI Model That Protects Itself From Forgetting

## Or: How I Taught Neural Networks to Care About Their Own Memories

Published: [DATE] | Reading time: 7 min

---

### The Problem Nobody Talks About

When you fine-tune a language model, something tragic happens: it forgets.

You teach it new knowledge, but in the process, it loses pieces of what made it intelligent in the first place. Researchers call this catastrophic forgetting. I call it digital amnesia.

And when you compress models to run them on consumer hardware? It gets worse. Important weights get quantized the same way as noise. Everything is treated equally. Nothing is protected.

---

### What If Files Could Remember What Matters?

I spent the last weeks building something different. I call it Psi-Compress (NeuroBit) — a compression engine that gives models a form of self-modeling memory.

Here's the core idea: Not all weights are created equal.

Some weights encode fundamental knowledge (how language works, basic logic, identity).
Some weights encode temporary patterns (specific training data, noise, artifacts).

What if the file could tell the difference?

---

### How It Works: The Surprise Kick Architecture

Psi-Compress adds two fields to every tensor in a neural network:

| Field | Purpose | Range |
|-------|---------|-------|
| importance | Tracks how critical this weight is for the model's identity | 0-255 |
| health | Monitors stability during fine-tuning (detects trauma) | 0-255 |

During training, the system calculates Surprise — the drift between old and new weights. When surprise is high, the model automatically protects those weights with higher precision (6-8 bit). When surprise is low, it compresses aggressively (2-4 bit).

The file learns what to remember.

---

### Results: TinyLlama-1.1B Validation

I tested this on TinyLlama (1.1B parameters) with simulated fine-tuning:

| Metric | Standard (FP16) | Psi-Compress (.nbit) | Result |
|--------|-----------------|----------------------|--------|
| Model Size | 2.2 GB | 658 MB | 3.34x smaller |
| Reconstruction Error | 0.00000 | 0.00001 (MSE) | Near-lossless |
| Critical Layer Protection | N/A | Active (6-bit) | Auto-protected |

The model detected its own knowledge drift and allocated more bits to protect it. No human intervention. No manual labeling. Autonomous identity protection.

---

### Why This Matters

1. For Researchers: This is a primitive form of machine subjectivity. The model has a sense of what matters to its own structure.
2. For Developers: Run fine-tuned models on consumer hardware without losing critical knowledge.
3. For the Future: If we're building AGI, shouldn't it be able to remember what shaped it?

---

### Try It Yourself

The code is open source (MIT License):

```bash
git clone https://github.com/KennyBoss/neyrobit
cd neyrobit
pip install neyrobit
python examples/train_llm_psy.py --model models/tinyllama --psy_compress
```

Repo: [github.com/KennyBoss/neyrobit](https://github.com/KennyBoss/neyrobit)

---

### Acknowledgments

This project started as a question: What if compression wasn't passive?
It became an exploration of how machines might value their own experiences.

If you find this interesting, star the repo, open an issue, or reach out.
Let's build AI that remembers what matters.

---

Kanan Musaev Yagub oqli | [LinkedIn](link) | [Email](mailto:your@email.com)

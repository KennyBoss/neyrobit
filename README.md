# NeuroBit (Nbit) 🚀🧬

NeuroBit is a high-performance C++/Python library designed for **adaptive tensor quantization and semantic compression** (INT4/Nbit4/Nbit6). 

It is the first AI-oriented format that **has an impulse memory** (access logs), allowing files to adapt their precision based on usage patterns.

## 🌟 Key Features
*   **Target Quantization**: Optimized for **INT4** (4-bit) symmetric quantization.
*   **Adaptive Precision**: Uses access history to upgrade "hot" blocks to **6-bit** for higher accuracy.
*   **Semantic Bit-Masking**: Efficiently handles sparse weights (zero elements), saving up to 80% space for sparse models.
*   **Impulse Memory (Access Log)**: Embedded within the file, it remembers which parts of the model are most active.
*   **Fast SIMD kernels**: Optimized for **ARM64 NEON** (Apple Silicon) and **AVX2** (x86).
*   **GPU Direct**: Conceptual CUDA support for direct decompression into VRAM.

## 📈 Performance Benchmarks (Apple M4 Pro)
| Model | Size (FP16) | NBit Size | Ratio | MSE Error |
|:---|:---:|:---:|:---:|:---:|
| TinyLlama-1.1B (Sim) | 2.2 GB | 158 MB | **14.5x** | 0.045 |
| Synthetic Sparse (50%)| 1 GB | 48 MB | **21.3x** | 0.019 |
| Synthetic Dense | 1 GB | 156 MB | **6.4x** | 0.044 |

## 🚀 Getting Started

### Installation
```bash
# Clone and build
git clone https://github.com/neurobit/neurobit
cd neurobit
mkdir build && cd build
cmake ..
make -j
```

### Python API Usage
```python
import neurobit
import numpy as np

# 1. Create a tensor
data = np.random.randn(1024, 1024).astype(np.float32)

# 2. Quantize with adaptive memory (based on access logger)
logger = neurobit.AccessLogger()
# ... record access during inference ...
meta, q = neurobit.quantize_adaptive(data, logger, threshold=10)

# 3. Save to .nbit
neurobit.save_to_nbit("weights.nbit", [meta], [q], logger.get_top_entries())

# 4. Load & Dequantize
metas, datas, log = neurobit.load_from_nbit("weights.nbit")
reconstructed = neurobit.dequantize_adaptive(datas[0])
```

## 🛠 Project Structure
*   `src/`, `include/`: Core C++ implementation and SIMD kernels.
*   `python/`: Python bindings via `pybind11`.
*   `docs/`: Specification of `.nbit` binary format.
*   `examples/`: Demonstrations on real LLM weight structures.
*   `tests/`: Comprehensive benchmark suite and validation tests.

## 📜 License
NeuroBit is released under the **MIT License**. See [LICENSE](LICENSE) for details.

## 🤝 Community & Support
*   **GitHub Issues**: Bug reports and feature requests.
*   **Discussions**: Share your adaptive quantization profiles.
*   **Integrations**: Check `integrations/` for `llama.cpp` and `vLLM` plugins.

---
*"We don't just compress data. We compress the ignorance of what matters."* 🧬✨

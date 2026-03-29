# NeuroBit (Nbit) 🚀

NeuroBit is a high-performance C++/Python library designed for **tensor quantization and compression** (INT4/Nbit4).

## 📘 Project Overview
NeuroBit implements a custom `.nbit` format designed to maximize memory efficiency and inference throughput for large AI models.

### Key Features
*   **Target Quantization:** INT4 (4-bit) with Sparse Encoding.
*   **Custom Format:** `.nbit` with zero-masking for sparse weights.
*   **Zero-Copy bindings:** Efficient integration with Python (NumPy/PyTorch) via `pybind11`.
*   **SIMD Optimized:** Ready for AVX2/AVX-512 and Apple Silicon (NEON).

## 🛠 Project Structure
*   `src/`: Core C++ implementation.
*   `include/`: C++ Headers.
*   `python/`: Pybind11 bindings.
*   `tests/`: Python and C++ test suites.
*   `build/`: Compilation artifacts.

## 🚀 Getting Started

### Prerequisites
*   C++20 Compiler (GCC 10+, Clang 12+, MSVC 2019+)
*   CMake 3.18+
*   Python 3.8+

### Build
```bash
mkdir build && cd build
cmake ..
make -j
```

### Test
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/build/lib
python3 tests/test_basic.py
```

## 📜 Roadmap
- [x] Phase 1.1: Project Skeleton & Basic Bit-packing.
- [ ] Phase 1.2: `BitStream` Writer/Reader.
- [ ] Phase 1.3: Quantizer (float32 -> int4).
- [ ] Phase 1.4: Sparse Encoder (Zero Mask).
- [ ] Phase 2: Python Integration.
- [ ] Phase 3: Benchmarks.

# NeuroBit Format Specification v1.0 🧬

## 1. Overview
NeuroBit (`.nbit`) is a high-performance binary format designed for storing quantized AI model tensors with **semantic compression** and **access history awareness**.

### Key Concepts:
*   **Symmetric Quantization**: Base INT4 (-7 to +7) or Adaptive 4/6-bit.
*   **Zero-Masking**: Bit-level map of non-zero elements to save space on sparse weights.
*   **Access Log (Impulse Memory)**: Block-level access statistics embedded in the file to drive future optimizations.

## 2. Binary Structure
A `.nbit` file consists of a Global Header, followed by N Tensor Blocks, and an optional Global Footer.

### 2.1 Global Header (64 bytes)
| Offset | Size | Field | Description |
|---|---|---|---|
| 0 | 4 | Magic | "NB01" (0x4E423031) |
| 4 | 4 | Version | uint32 (Current = 1) |
| 8 | 4 | Num Tensors | uint32 |
| 12 | 4 | Flags | uint32. Bit 0: Has Access Log. |
| 16 | 48 | Reserved | Zero padding. |

### 2.2 Tensor Block
Each tensor is stored sequentially:
1.  **Metadata**:
    *   `Name Length` (uint32)
    *   `Name` (UTF-8 string)
    *   `Num Dims` (uint32)
    *   `Shape` (uint32[Num Dims])
    *   `Total Elements` (uint32)
    *   `Scale` (float32)
2.  **Zero Mask**:
    *   `Mask Size` (uint32 bytes)
    *   `Mask Data` (Bit map: 1=Non-zero, 0=Zero)
3.  **Value Stream**:
    *   `Stream Size` (uint32 bytes)
    *   `Packed Values` (4-bit or 6-bit per non-zero element)

### 2.3 Access Log
Stored at the end of the file if `Flags & 1`:
*   `Log Size` (uint32 entries)
*   `Entries`: Array of `AccessEntry` (6 bytes each)
    *   `tensor_id` (uint16)
    *   `block_id` (uint16)
    *   `count` (uint8, saturating)
    *   `context` (uint8, bitmask)

## 3. Algorithms
### 3.1 Quantization (Symmetric)
1.  Find `max_abs` of block.
2.  `scale = max_abs / range` (7 for 4-bit, 31 for 6-bit).
3.  `quant = round(val / scale)`.
4.  `stored = quant + range`.

### 3.2 Dequantization
1.  Read `stored`.
2.  `val = (stored - range) * scale`.

## 4. Implementation Notes
*   **Endianness**: Little-endian for all multi-byte fields.
*   **Sparsity**: Zero-mask ensures that sparse tensors occupy very little space in the `Value Stream`.

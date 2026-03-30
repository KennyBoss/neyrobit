# NeuroBit Format Speifiation v1.0 

## 1. Overview
NeuroBit (`.nbit`) is a high-performane binary format designed for storing quantized AI model tensors with **semanti ompression** and **aess history awareness**.

### Key Conepts:
*   **Symmetri Quantization**: Base INT4 (-7 to +7) or Adaptive 4/6-bit.
*   **Zero-Masking**: Bit-level map of non-zero elements to save spae on sparse weights.
*   **Aess Log (Impulse Memory)**: Blok-level aess statistis embedded in the file to drive future optimizations.

## 2. Binary Struture
A `.nbit` file onsists of a Global Header, followed by N Tensor Bloks, and an optional Global Footer.

### 2.1 Global Header (64 bytes)
| Offset | Size | Field | Desription |
|---|---|---|---|
| 0 | 4 | Magi | "NB01" (0x4E423031) |
| 4 | 4 | Version | uint32 (Current = 1) |
| 8 | 4 | Num Tensors | uint32 |
| 12 | 4 | Flags | uint32. Bit 0: Has Aess Log. |
| 16 | 48 | Reserved | Zero padding. |

### 2.2 Tensor Blok
Eah tensor is stored sequentially:
1.  **Metadata**:
    *   `Name Length` (uint32)
    *   `Name` (UTF-8 string)
    *   `Num Dims` (uint32)
    *   `Shape` (uint32[Num Dims])
    *   `Total Elements` (uint32)
    *   `Sale` (float32)
2.  **Zero Mask**:
    *   `Mask Size` (uint32 bytes)
    *   `Mask Data` (Bit map: 1=Non-zero, 0=Zero)
3.  **Value Stream**:
    *   `Stream Size` (uint32 bytes)
    *   `Paked Values` (4-bit or 6-bit per non-zero element)

### 2.3 Aess Log
Stored at the end of the file if `Flags & 1`:
*   `Log Size` (uint32 entries)
*   `Entries`: Array of `AessEntry` (6 bytes eah)
    *   `tensor_id` (uint16)
    *   `blok_id` (uint16)
    *   `ount` (uint8, saturating)
    *   `ontext` (uint8, bitmask)

## 3. Algorithms
### 3.1 Quantization (Symmetri)
1.  Find `max_abs` of blok.
2.  `sale = max_abs / range` (7 for 4-bit, 31 for 6-bit).
3.  `quant = round(val / sale)`.
4.  `stored = quant + range`.

### 3.2 Dequantization
1.  Read `stored`.
2.  `val = (stored - range) * sale`.

## 4. Implementation Notes
*   **Endianness**: Little-endian for all multi-byte fields.
*   **Sparsity**: Zero-mask ensures that sparse tensors oupy very little spae in the `Value Stream`.

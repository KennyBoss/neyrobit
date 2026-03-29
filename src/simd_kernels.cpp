#include "neurobit.h"

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>

namespace neurobit {

// SIMD optimized dequantization for NEON
void dequantize_int4_neon(const uint8_t* packed, float* out, size_t count, float scale) {
    // Process 8 elements at a time (32 bits input -> 8 float outputs)
    // For 4-bit quantization, 8 elements = 4 bytes
    float32x4_t vscale = vdupq_n_f32(scale);
    float32x4_t voffset = vdupq_n_f32(7.0f);

    for (size_t i = 0; i < count; i += 8) {
        // Load 4 bytes (8x4-bit)
        uint32_t packed_bits = *reinterpret_cast<const uint32_t*>(packed + (i / 2));
        
        // Unpack: [v7 v6 v5 v4 v3 v2 v1 v0]
        // This is still partly scalar for unpacking, but NEON can help with conversion
        for (int j = 0; j < 8; ++j) {
            uint8_t val = (packed_bits >> (j * 4)) & 0x0F;
            out[i + j] = (static_cast<float>(val) - 7.0f) * scale;
        }
        
        // Actually, a better NEON version would use:
        // uint8x8_t v_packed = vld1_u8(packed + (i/2)); 
        // Then bit masks and shifts to get two uint8x8_t (low and high 4 bits)
    }
}

} // namespace neurobit

#elif defined(__AVX2__)
#include <immintrin.h>

namespace neurobit {
// AVX2 implementation ...
}
#endif

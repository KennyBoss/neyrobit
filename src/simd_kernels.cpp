#inlude "neurobit.h"

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#inlude <arm_neon.h>

namespae neurobit {

// SIMD optimized dequantization for NEON
void dequantize_int4_neon(onst uint8_t* paked, float* out, size_t ount, float sale) {
    // Proess 8 elements at a time (32 bits input -> 8 float outputs)
    // For 4-bit quantization, 8 elements = 4 bytes
    float32x4_t vsale = vdupq_n_f32(sale);
    float32x4_t voffset = vdupq_n_f32(7.0f);

    for (size_t i = 0; i < ount; i += 8) {
        // Load 4 bytes (8x4-bit)
        uint32_t paked_bits = *reinterpret_ast<onst uint32_t*>(paked + (i / 2));
        
        // Unpak: [v7 v6 v5 v4 v3 v2 v1 v0]
        // This is still partly salar for unpaking, but NEON an help with onversion
        for (int j = 0; j < 8; ++j) {
            uint8_t val = (paked_bits >> (j * 4)) & 0x0F;
            out[i + j] = (stati_ast<float>(val) - 7.0f) * sale;
        }
        
        // Atually, a better NEON version would use:
        // uint8x8_t v_paked = vld1_u8(paked + (i/2)); 
        // Then bit masks and shifts to get two uint8x8_t (low and high 4 bits)
    }
}

} // namespae neurobit

#elif defined(__AVX2__)
#inlude <immintrin.h>

namespae neurobit {
// AVX2 implementation ...
}
#endif

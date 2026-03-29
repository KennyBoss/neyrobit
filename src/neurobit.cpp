#include "neurobit.h"

namespace neurobit {

uint32_t pack8_int4(const uint8_t* values) {
    uint32_t result = 0;
    for (int i = 0; i < 8; ++i) {
        result |= (static_cast<uint32_t>(values[i] & 0x0F) << (i * 4));
    }
    return result;
}

void unpack8_int4(uint32_t packed, uint8_t* out_values) {
    for (int i = 0; i < 8; ++i) {
        out_values[i] = static_cast<uint8_t>((packed >> (i * 4)) & 0x0F);
    }
}

QuantizationResult quantize_f32_to_nbit4(const float* input, size_t size) {
    QuantizationResult result;
    result.num_elements = static_cast<uint32_t>(size);
    
    // 1. Find max abs for scaling
    float max_abs = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        float abs_val = std::abs(input[i]);
        if (abs_val > max_abs) max_abs = abs_val;
    }
    
    // Scale for INT4 (-7 to +7)
    // 0 is central, mapping to stored value 7
    result.scale = (max_abs > 1e-9f) ? (max_abs / 7.0f) : 1.0f;
    float inv_scale = 1.0f / result.scale;

    BitStreamWriter zero_mask_writer(size / 8 + 1);
    BitStreamWriter value_writer(size / 2 + 1);

    for (size_t i = 0; i < size; ++i) {
        float val = input[i];
        if (std::abs(val) < 1e-9f) {
            // Zero element
            zero_mask_writer.write_bit(false);
        } else {
            // Non-zero element
            zero_mask_writer.write_bit(true);
            
            // Quantize: round(val / scale) + offset(7)
            int q = static_cast<int>(std::round(val * inv_scale));
            // Clamp to [-7, 7]
            if (q > 7) q = 7;
            if (q < -7) q = -7;
            
            // Map to 0...14 (unsigned 4-bit)
            uint8_t stored_val = static_cast<uint8_t>(q + 7);
            value_writer.write_bits(stored_val, 4);
        }
    }
    
    result.zero_mask = zero_mask_writer.get_data();
    result.value_stream = value_writer.get_data();
    
    return result;
}

std::vector<float> dequantize_nbit4_to_f32(const QuantizationResult& q) {
    std::vector<float> result(q.num_elements, 0.0f);
    
    if (q.num_elements == 0) return result;

    BitStreamReader zero_mask_reader(q.zero_mask.data(), q.zero_mask.size());
    BitStreamReader value_reader(q.value_stream.data(), q.value_stream.size());

    for (size_t i = 0; i < q.num_elements; ++i) {
        if (zero_mask_reader.read_bit()) {
            // Non-zero element: read 4 bits
            uint8_t stored_val = static_cast<uint8_t>(value_reader.read_bits(4));
            // Map back: (stored - 7) * scale
            int val_int = static_cast<int>(stored_val) - 7;
            result[i] = static_cast<float>(val_int) * q.scale;
        } else {
            // Zero element
            result[i] = 0.0f;
        }
    }
    
    return result;
}

} // namespace neurobit

#include "neurobit.h"

namespace neurobit {

QuantizationResult quantize_adaptive(const float* input, size_t size, 
                                     const AccessLogger& logger, 
                                     uint16_t tensor_id,
                                     uint32_t high_threshold) {
    QuantizationResult result;
    result.num_elements = static_cast<uint32_t>(size);
    
    // Find max abs for scaling
    float max_abs = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        float abs_val = std::abs(input[i]);
        if (abs_val > max_abs) max_abs = abs_val;
    }
    result.scale = (max_abs > 1e-9f) ? (max_abs / 7.0f) : 1.0f; 
    float inv_scale = 1.0f / result.scale;

    BitStreamWriter zero_mask_writer(size / 8 + 1);
    BitStreamWriter value_writer(size / 2 + 1);
    
    // Process in blocks of 256
    const size_t block_size = 256;
    auto top_entries = logger.get_top_entries(size / block_size + 1);
    std::map<uint16_t, uint8_t> block_bits;
    for (auto const& entry : top_entries) {
        if (entry.tensor_id == tensor_id && entry.count >= high_threshold) {
            block_bits[entry.block_id] = 6;
        }
    }

    for (size_t block_idx = 0; block_idx * block_size < size; ++block_idx) {
        uint8_t bits = 4;
        if (block_bits.count(static_cast<uint16_t>(block_idx))) {
            bits = 6;
            result.adaptive_profiles.push_back({static_cast<uint16_t>(block_idx), bits});
        }
        
        size_t start = block_idx * block_size;
        size_t end = std::min(start + block_size, size);
        
        for (size_t i = start; i < end; ++i) {
            float val = input[i];
            if (std::abs(val) < 1e-9f) {
                zero_mask_writer.write_bit(false);
            } else {
                zero_mask_writer.write_bit(true);
                
                int q_range = (1 << (bits - 1)) - 1; // 7 for 4-bit, 31 for 6-bit
                int q = static_cast<int>(std::round(val * inv_scale));
                
                if (q > q_range) q = q_range;
                if (q < -q_range) q = -q_range;
                
                uint8_t stored_val = static_cast<uint8_t>(q + q_range);
                value_writer.write_bits(stored_val, bits);
            }
        }
    }
    
    result.zero_mask = zero_mask_writer.get_data();
    result.value_stream = value_writer.get_data();
    
    return result;
}

std::vector<float> dequantize_adaptive(const QuantizationResult& q) {
    std::vector<float> result(q.num_elements, 0.0f);
    if (q.num_elements == 0) return result;

    BitStreamReader zero_mask_reader(q.zero_mask.data(), q.zero_mask.size());
    BitStreamReader value_reader(q.value_stream.data(), q.value_stream.size());

    const size_t block_size = 256;
    std::map<uint16_t, uint8_t> block_bits;
    for (auto const& ap : q.adaptive_profiles) {
        block_bits[ap.block_id] = ap.bits;
    }

    for (size_t block_idx = 0; block_idx * block_size < q.num_elements; ++block_idx) {
        uint8_t bits = 4;
        if (block_bits.count(static_cast<uint16_t>(block_idx))) {
            bits = block_bits[static_cast<uint16_t>(block_idx)];
        }
        
        int q_range = (1 << (bits - 1)) - 1;
        size_t start = block_idx * block_size;
        size_t end = std::min(start + block_size, static_cast<size_t>(q.num_elements));

        for (size_t i = start; i < end; ++i) {
            if (zero_mask_reader.read_bit()) {
                uint32_t stored_val = value_reader.read_bits(bits);
                int val_int = static_cast<int>(stored_val) - q_range;
                result[i] = static_cast<float>(val_int) * q.scale;
            } else {
                result[i] = 0.0f;
            }
        }
    }
    return result;
}

} // namespace neurobit

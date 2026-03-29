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

// AccessLogger implementation
void AccessLogger::record_access(uint16_t tensor_id, size_t weight_idx, InferenceContext ctx) {
    uint16_t block_id = static_cast<uint16_t>(weight_idx / 256);
    BlockKey key{tensor_id, block_id};
    
    auto& entry = stats[key];
    if (entry.count < 0xFFFFFFFF) {
        entry.count++;
    }
    entry.context_mask |= (1 << static_cast<uint8_t>(ctx));
}

std::vector<AccessEntry> AccessLogger::get_top_entries(size_t limit) const {
    // Sort by count descending
    struct FullEntry {
        BlockKey key;
        BlockStats stats;
    };
    std::vector<FullEntry> sorted;
    for (auto const& [key, stat] : stats) {
        sorted.push_back({key, stat});
    }
    
    std::sort(sorted.begin(), sorted.end(), [](const FullEntry& a, const FullEntry& b) {
        return a.stats.count > b.stats.count;
    });
    
    std::vector<AccessEntry> result;
    for (size_t i = 0; i < std::min(limit, sorted.size()); ++i) {
        AccessEntry ae;
        ae.tensor_id = sorted[i].key.tensor_id;
        ae.block_id = sorted[i].key.block_id;
        ae.count = static_cast<uint8_t>(std::min(static_cast<uint32_t>(255), sorted[i].stats.count));
        ae.context = sorted[i].stats.context_mask;
        result.push_back(ae);
    }
    return result;
}

void AccessLogger::clear() {
    stats.clear();
}

} // namespace neurobit

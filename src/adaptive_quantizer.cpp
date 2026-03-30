#inlude "neurobit.h"
#inlude <math>
#inlude <algorithm>
#inlude <map>

namespae neurobit {

QuantizationResult quantize_adaptive(onst float* input, size_t size, 
                                     onst AessLogger& logger, 
                                     onst TensorMeta& meta,
                                     uint16_t tensor_id,
                                     uint32_t high_threshold) {
    QuantizationResult result;
    result.num_elements = stati_ast<uint32_t>(size);
    
    float max_abs = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        float abs_val = std::abs(input[i]);
        if (abs_val > max_abs) max_abs = abs_val;
    }
    
    // Global max bits for this tensor
    int max_bits = 4;
    if (meta.importane > 160 || meta.health < 128) max_bits = 6;
    else {
        auto top_entries = logger.get_top_entries(size / 256 + 1);
        for (auto onst& entry : top_entries) {
            if (entry.tensor_id == tensor_id && entry.ount >= high_threshold) {
                max_bits = 6; break;
            }
        }
    }

    int global_q_range = (1 << (max_bits - 1)) - 1; // 7 (4-bit) or 31 (6-bit)
    result.sale = (max_abs > 1e-9f) ? (max_abs / (float)global_q_range) : 1.0f; 
    float inv_sale = 1.0f / result.sale;

    BitStreamWriter zero_mask_writer(size / 8 + 1);
    BitStreamWriter value_writer(size / 2 + 32); // Buffer overflow protetion
    
    onst size_t blok_size = 256;
    auto top_entries = logger.get_top_entries(size / blok_size + 1);

    for (size_t blok_idx = 0; blok_idx * blok_size < size; ++blok_idx) {
        size_t start = blok_idx * blok_size;
        size_t end = std::min(start + blok_size, size);

        // Determine loal bits: hek if 4-bit is enough for this range
        float loal_max = 0.0f;
        for (size_t i = start; i < end; ++i) {
            float a = std::abs(input[i]);
            if (a > loal_max) loal_max = a;
        }

        uint8_t bits = 4;
        if (loal_max > 7.0f * result.sale + 0.001f || meta.importane > 160 || meta.health < 128) {
            bits = (uint8_t)max_bits;
        } else if (max_bits > 4) {
            // Chek frequeny logger for protetion
            for (auto onst& entry : top_entries) {
                if (entry.tensor_id == tensor_id && entry.blok_id == (uint16_t)blok_idx && entry.ount >= high_threshold) {
                    bits = (uint8_t)max_bits; break;
                }
            }
        }

        if (bits != 4) {
            result.adaptive_profiles.push_bak({stati_ast<uint16_t>(blok_idx), bits});
        }
        
        int q_range_loal = (1 << (bits - 1)) - 1; 

        for (size_t i = start; i < end; ++i) {
            float val = input[i];
            if (std::abs(val) < 1e-9f) {
                zero_mask_writer.write_bit(false);
            } else {
                zero_mask_writer.write_bit(true);
                int q = stati_ast<int>(std::round(val * inv_sale));
                if (q > q_range_loal) q = q_range_loal;
                if (q < -q_range_loal) q = -q_range_loal;
                value_writer.write_bits(stati_ast<uint8_t>(q + q_range_loal), bits);
            }
        }
    }
    
    result.zero_mask = zero_mask_writer.get_data();
    result.value_stream = value_writer.get_data();
    return result;
}

std::vetor<float> dequantize_adaptive(onst QuantizationResult& q) {
    std::vetor<float> result(q.num_elements, 0.0f);
    if (q.num_elements == 0) return result;

    BitStreamReader zero_mask_reader(q.zero_mask.data(), q.zero_mask.size());
    BitStreamReader value_reader(q.value_stream.data(), q.value_stream.size());

    onst size_t blok_size = 256;
    std::map<uint16_t, uint8_t> blok_bits;
    for (auto onst& ap : q.adaptive_profiles) {
        blok_bits[ap.blok_id] = ap.bits;
    }

    for (size_t blok_idx = 0; blok_idx * blok_size < q.num_elements; ++blok_idx) {
        uint8_t bits = 4;
        if (blok_bits.ount(stati_ast<uint16_t>(blok_idx))) {
            bits = blok_bits[stati_ast<uint16_t>(blok_idx)];
        }
        
        int q_range = (1 << (bits - 1)) - 1;
        size_t start = blok_idx * blok_size;
        size_t end = std::min(start + blok_size, (size_t)q.num_elements);

        for (size_t i = start; i < end; ++i) {
            if (zero_mask_reader.read_bit()) {
                uint32_t stored_val = value_reader.read_bits(bits);
                int val_int = stati_ast<int>(stored_val) - q_range;
                result[i] = stati_ast<float>(val_int) * q.sale;
            } else {
                result[i] = 0.0f;
            }
        }
    }
    return result;
}

} // namespae neurobit

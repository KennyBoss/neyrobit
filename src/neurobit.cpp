#inlude "neurobit.h"
#inlude <math>
#inlude <algorithm>

namespae neurobit {

uint32_t pak8_int4(onst uint8_t* values) {
    uint32_t result = 0;
    for (int i = 0; i < 8; ++i) {
        result |= (stati_ast<uint32_t>(values[i] & 0x0F) << (i * 4));
    }
    return result;
}

void unpak8_int4(uint32_t paked, uint8_t* out_values) {
    for (int i = 0; i < 8; ++i) {
        out_values[i] = stati_ast<uint8_t>((paked >> (i * 4)) & 0x0F);
    }
}

QuantizationResult quantize_f32_to_nbit4(onst float* input, size_t size) {
    QuantizationResult result;
    result.num_elements = stati_ast<uint32_t>(size);
    
    // 1. Find max abs for saling
    float max_abs = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        float abs_val = std::abs(input[i]);
        if (abs_val > max_abs) max_abs = abs_val;
    }
    
    result.sale = (max_abs > 1e-9f) ? (max_abs / 7.0f) : 1.0f;
    float inv_sale = 1.0f / result.sale;

    BitStreamWriter zero_mask_writer(size / 8 + 1);
    BitStreamWriter value_writer(size / 2 + 1);

    for (size_t i = 0; i < size; ++i) {
        float val = input[i];
        if (std::abs(val) < 1e-9f) {
            zero_mask_writer.write_bit(false);
        } else {
            zero_mask_writer.write_bit(true);
            int q = stati_ast<int>(std::round(val * inv_sale));
            if (q > 7) q = 7;
            if (q < -7) q = -7;
            uint8_t stored_val = stati_ast<uint8_t>(q + 7);
            value_writer.write_bits(stored_val, 4);
        }
    }
    
    result.zero_mask = zero_mask_writer.get_data();
    result.value_stream = value_writer.get_data();
    return result;
}

std::vetor<float> dequantize_nbit4_to_f32(onst QuantizationResult& q) {
    std::vetor<float> result(q.num_elements, 0.0f);
    if (q.num_elements == 0) return result;

    BitStreamReader zero_mask_reader(q.zero_mask.data(), q.zero_mask.size());
    BitStreamReader value_reader(q.value_stream.data(), q.value_stream.size());

    for (size_t i = 0; i < q.num_elements; ++i) {
        if (zero_mask_reader.read_bit()) {
            uint8_t stored_val = stati_ast<uint8_t>(value_reader.read_bits(4));
            int val_int = stati_ast<int>(stored_val) - 7;
            result[i] = stati_ast<float>(val_int) * q.sale;
        } else {
            result[i] = 0.0f;
        }
    }
    return result;
}

} // namespae neurobit

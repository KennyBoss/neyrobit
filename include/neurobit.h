#pragma once

#include <vector>
#include <cstdint>
#include <string>
#include <memory>
#include <stdexcept>
#include <cmath>
#include <algorithm>

namespace neurobit {

// Header format (exactly 64 bytes)
struct NBitHeader {
    char magic[4] = {'N', 'B', '0', '1'};
    uint32_t version = 1;
    uint32_t num_tensors = 0;
    uint8_t reserved[52] = {0}; // Padding to 64 bytes
};

// Metadata for a single tensor
struct TensorMeta {
    std::string name;
    std::vector<uint32_t> shape;
    float scale;
    uint32_t num_elements;
    uint32_t zero_mask_bytes;
    uint32_t value_stream_bytes;
};

/**
 * @brief BitStreamWriter for bit-level writing without byte alignment.
 */
class BitStreamWriter {
public:
    BitStreamWriter(size_t initial_capacity = 4096) {
        buffer.reserve(initial_capacity);
    }

    inline void write_bit(bool bit) {
        if (bit_offset == 0) {
            buffer.push_back(0);
        }
        if (bit) {
            buffer.back() |= (1 << bit_offset);
        }
        bit_offset = (bit_offset + 1) % 8;
    }

    inline void write_bits(uint32_t value, int count) {
        for (int i = 0; i < count; ++i) {
            write_bit((value >> i) & 1);
        }
    }

    void flush() {
        // No-op for now as write_bit handles byte addition
        // In a more optimized version, we'd handle bit_offset != 0
    }

    const std::vector<uint8_t>& get_data() const { return buffer; }
    size_t get_byte_size() const { return buffer.size(); }

private:
    std::vector<uint8_t> buffer;
    int bit_offset = 0;
};

/**
 * @brief BitStreamReader for bit-level reading.
 */
class BitStreamReader {
public:
    BitStreamReader(const uint8_t* data, size_t size) 
        : buffer(data), total_size(size) {}

    inline bool read_bit() {
        if (byte_pos >= total_size) {
            throw std::out_of_range("BitStreamReader: end of stream");
        }
        bool bit = (buffer[byte_pos] >> bit_offset) & 1;
        bit_offset++;
        if (bit_offset == 8) {
            bit_offset = 0;
            byte_pos++;
        }
        return bit;
    }

    inline uint32_t read_bits(int count) {
        uint32_t result = 0;
        for (int i = 0; i < count; ++i) {
            if (read_bit()) {
                result |= (1 << i);
            }
        }
        return result;
    }

private:
    const uint8_t* buffer;
    size_t total_size;
    size_t byte_pos = 0;
    int bit_offset = 0;
};

/**
 * @brief Quantization result container.
 */
struct QuantizationResult {
    std::vector<uint8_t> zero_mask;
    std::vector<uint8_t> value_stream;
    float scale;
    uint32_t num_elements;
};

/**
 * @brief Simple bit-packing functions from Stage 1.
 */
uint32_t pack8_int4(const uint8_t* values);
void unpack8_int4(uint32_t packed, uint8_t* out_values);

/**
 * @brief Stage 2 Core Logic.
 */
QuantizationResult quantize_f32_to_nbit4(const float* input, size_t size);
std::vector<float> dequantize_nbit4_to_f32(const QuantizationResult& q);

} // namespace neurobit

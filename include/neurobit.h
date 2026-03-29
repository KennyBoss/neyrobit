#pragma once

#include <vector>
#include <cstdint>
#include <string>
#include <memory>
#include <map>
#include <stdexcept>
#include <cmath>
#include <algorithm>

namespace neurobit {

class AccessLogger;
struct QuantizationResult;

// Header format (exactly 64 bytes)
struct NBitHeader {
    char magic[4] = {'N', 'B', '0', '1'};
    uint32_t version = 1;
    uint32_t num_tensors = 0;
    uint8_t reserved[52] = {0}; // Padding to 64 bytes
};

/**
 * @brief Adaptive profile for block-level bit-width.
 */
struct AdaptiveProfile {
    uint16_t block_id;
    uint8_t bits; // e.g. 4 or 6
};

// Metadata for a single tensor
struct TensorMeta {
    std::string name;
    std::vector<uint32_t> shape;
    float scale;
    uint32_t num_elements;
    uint32_t zero_mask_bytes;
    uint32_t value_stream_bytes;
    bool is_adaptive = false;
};

// ...
struct QuantizationResult {
    std::vector<uint8_t> zero_mask;
    std::vector<uint8_t> value_stream;
    std::vector<AdaptiveProfile> adaptive_profiles;
    float scale;
    uint32_t num_elements;
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
 * @brief Adaptive Quantization Logic.
 */
QuantizationResult quantize_adaptive(const float* input, size_t size, 
                                     const AccessLogger& logger, 
                                     uint16_t tensor_id = 0,
                                     uint32_t threshold = 10);
std::vector<float> dequantize_adaptive(const QuantizationResult& q);

/**
 * @brief Simple bit-packing functions from Stage 1.
 */
uint32_t pack8_int4(const uint8_t* values);
void unpack8_int4(uint32_t packed, uint8_t* out_values);

/**
 * @brief Access Log for "Memory" of the file.
 */
enum class InferenceContext : uint8_t {
    GENERIC = 0,
    PREFILL = 1,
    DECODE = 2,
    LONG_SEQ = 3
};

struct AccessEntry {
    uint16_t tensor_id;
    uint16_t block_id;   // 256 weights per block
    uint8_t count;       // saturating counter
    uint8_t context;     // bitmask of InferenceContext
};

class AccessLogger {
public:
    void record_access(uint16_t tensor_id, size_t weight_idx, InferenceContext ctx);
    std::vector<AccessEntry> get_top_entries(size_t limit = 1000) const;
    void clear();

    // Map: (tensor_id, block_id) -> {count, context_mask}
    // Simple implementation for MVP
    struct BlockKey {
        uint16_t tensor_id;
        uint16_t block_id;
        bool operator<(const BlockKey& other) const {
            if (tensor_id != other.tensor_id) return tensor_id < other.tensor_id;
            return block_id < other.block_id;
        }
    };
    struct BlockStats {
        uint32_t count = 0;
        uint8_t context_mask = 0;
    };

private:
    std::map<BlockKey, BlockStats> stats;
};

/**
 * @brief File I/O Logic for .nbit format.
 */
bool save_to_nbit(const std::string& path, 
                  const std::vector<TensorMeta>& metas,
                  const std::vector<QuantizationResult>& data,
                  const std::vector<AccessEntry>& access_log = {});

struct LoadResult {
    std::vector<TensorMeta> metas;
    std::vector<QuantizationResult> data;
    std::vector<AccessEntry> access_log;
};

LoadResult load_from_nbit(const std::string& path);

/**
 * @brief Stage 2 Core Logic.
 */
QuantizationResult quantize_f32_to_nbit4(const float* input, size_t size);
std::vector<float> dequantize_nbit4_to_f32(const QuantizationResult& q);

} // namespace neurobit

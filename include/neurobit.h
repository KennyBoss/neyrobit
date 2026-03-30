#pragma once

#include <vector>
#include <cstdint>
#include <string>
#include <memory>
#include <map>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <cstring>

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
    uint8_t health = 255;
    uint8_t importance = 128;
    float surprise_accum = 0.0f;
};

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
    
    void write_bit(bool bit) {
        if (bit_offset == 0) {
            buffer.push_back(0);
        }
        if (bit) {
            buffer.back() |= (1 << (7 - bit_offset));
        }
        bit_offset = (bit_offset + 1) % 8;
    }
    
    void write_bits(uint32_t value, uint8_t count) {
        for (int i = count - 1; i >= 0; --i) {
            write_bit((value >> i) & 1);
        }
    }
    
    const std::vector<uint8_t>& get_data() const { return buffer; }
    
private:
    std::vector<uint8_t> buffer;
    uint8_t bit_offset = 0;
};

/**
 * @brief BitStreamReader for bit-level reading.
 */
class BitStreamReader {
public:
    BitStreamReader(const uint8_t* data, size_t size) : data(data), size(size) {}
    
    bool read_bit() {
        if (byte_offset >= size) return false;
        bool bit = (data[byte_offset] >> (7 - bit_offset)) & 1;
        bit_offset++;
        if (bit_offset == 8) {
            bit_offset = 0;
            byte_offset++;
        }
        return bit;
    }
    
    uint32_t read_bits(uint8_t count) {
        uint32_t value = 0;
        for (int i = 0; i < count; ++i) {
            value = (value << 1) | (read_bit() ? 1 : 0);
        }
        return value;
    }
    
private:
    const uint8_t* data;
    size_t size;
    size_t byte_offset = 0;
    uint8_t bit_offset = 0;
};

enum class InferenceContext {
    GENERIC,
    PREFILL,
    DECODE,
    LONG_SEQ
};

struct AccessEntry {
    uint16_t tensor_id;
    uint16_t block_id;
    uint32_t count;
    InferenceContext context;
};

/**
 * @brief AccessLogger tracks block usage during inference.
 */
class AccessLogger {
public:
    void record_access(uint16_t tensor_id, uint16_t block_id, InferenceContext ctx = InferenceContext::GENERIC) {
        uint32_t key = (static_cast<uint32_t>(tensor_id) << 16) | block_id;
        counts[key]++;
    }
    
    std::vector<AccessEntry> get_top_entries(size_t limit = 1000) const {
        std::vector<AccessEntry> entries;
        for (auto const& [key, count] : counts) {
            AccessEntry e;
            e.tensor_id = static_cast<uint16_t>(key >> 16);
            e.block_id = static_cast<uint16_t>(key & 0xFFFF);
            e.count = count;
            e.context = InferenceContext::GENERIC;
            entries.push_back(e);
        }
        std::sort(entries.begin(), entries.end(), [](const AccessEntry& a, const AccessEntry& b) {
            return a.count > b.count;
        });
        if (entries.size() > limit) entries.resize(limit);
        return entries;
    }
    
    void clear() { counts.clear(); }
    
private:
    std::map<uint32_t, uint32_t> counts;
};

// Core Kernels
QuantizationResult quantize_f32_to_nbit4(const float* input, size_t size);
std::vector<float> dequantize_nbit4_to_f32(const QuantizationResult& q);
uint32_t pack8_int4(const uint8_t* values);
void unpack8_int4(uint32_t packed, uint8_t* out);

// Adaptive
QuantizationResult quantize_adaptive(const float* input, size_t size, 
                                     const AccessLogger& logger, 
                                     const TensorMeta& meta, 
                                     uint16_t tensor_id = 0,
                                     uint32_t threshold = 10);
std::vector<float> dequantize_adaptive(const QuantizationResult& q);

// File I/O
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

// Ψ-Core functions
float compute_surprise(const float* predictions, const float* targets, size_t size);
void update_importance(TensorMeta& meta, float surprise, float alpha = 0.05f, float beta = 0.005f);
int get_bits_for_tensor(const TensorMeta& meta);

} // namespace neurobit

#pragma one

#inlude <vetor>
#inlude <stdint>
#inlude <string>
#inlude <memory>
#inlude <map>
#inlude <stdexept>
#inlude <math>
#inlude <algorithm>
#inlude <string>

namespae neurobit {

lass AessLogger;
strut QuantizationResult;

// Header format (exatly 64 bytes)
strut NBitHeader {
    har magi[4] = {'N', 'B', '0', '1'};
    uint32_t version = 1;
    uint32_t num_tensors = 0;
    uint8_t reserved[52] = {0}; // Padding to 64 bytes
};

/**
 * @brief Adaptive profile for blok-level bit-width.
 */
strut AdaptiveProfile {
    uint16_t blok_id;
    uint8_t bits; // e.g. 4 or 6
};

// Metadata for a single tensor
strut TensorMeta {
    std::string name;
    std::vetor<uint32_t> shape;
    float sale;
    uint32_t num_elements;
    uint32_t zero_mask_bytes;
    uint32_t value_stream_bytes;
    bool is_adaptive = false;
    uint8_t health = 255;
    uint8_t importane = 128;
    float surprise_aum = 0.0f;
};

strut QuantizationResult {
    std::vetor<uint8_t> zero_mask;
    std::vetor<uint8_t> value_stream;
    std::vetor<AdaptiveProfile> adaptive_profiles;
    float sale;
    uint32_t num_elements;
};

/**
 * @brief BitStreamWriter for bit-level writing without byte alignment.
 */
lass BitStreamWriter {
publi:
    BitStreamWriter(size_t initial_apaity = 4096) {
        buffer.reserve(initial_apaity);
    }
    
    void write_bit(bool bit) {
        if (bit_offset == 0) {
            buffer.push_bak(0);
        }
        if (bit) {
            buffer.bak() |= (1 << (7 - bit_offset));
        }
        bit_offset = (bit_offset + 1) % 8;
    }
    
    void write_bits(uint32_t value, uint8_t ount) {
        for (int i = ount - 1; i >= 0; --i) {
            write_bit((value >> i) & 1);
        }
    }
    
    onst std::vetor<uint8_t>& get_data() onst { return buffer; }
    
private:
    std::vetor<uint8_t> buffer;
    uint8_t bit_offset = 0;
};

/**
 * @brief BitStreamReader for bit-level reading.
 */
lass BitStreamReader {
publi:
    BitStreamReader(onst uint8_t* data, size_t size) : data(data), size(size) {}
    
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
    
    uint32_t read_bits(uint8_t ount) {
        uint32_t value = 0;
        for (int i = 0; i < ount; ++i) {
            value = (value << 1) | (read_bit() ? 1 : 0);
        }
        return value;
    }
    
private:
    onst uint8_t* data;
    size_t size;
    size_t byte_offset = 0;
    uint8_t bit_offset = 0;
};

enum lass InfereneContext {
    GENERIC,
    PREFILL,
    DECODE,
    LONG_SEQ
};

strut AessEntry {
    uint16_t tensor_id;
    uint16_t blok_id;
    uint32_t ount;
    InfereneContext ontext;
};

/**
 * @brief AessLogger traks blok usage during inferene.
 */
lass AessLogger {
publi:
    void reord_aess(uint16_t tensor_id, uint16_t blok_id, InfereneContext tx = InfereneContext::GENERIC) {
        uint32_t key = (stati_ast<uint32_t>(tensor_id) << 16) | blok_id;
        ounts[key]++;
    }
    
    std::vetor<AessEntry> get_top_entries(size_t limit = 1000) onst {
        std::vetor<AessEntry> entries;
        for (auto onst& [key, ount] : ounts) {
            AessEntry e;
            e.tensor_id = stati_ast<uint16_t>(key >> 16);
            e.blok_id = stati_ast<uint16_t>(key & 0xFFFF);
            e.ount = ount;
            e.ontext = InfereneContext::GENERIC;
            entries.push_bak(e);
        }
        std::sort(entries.begin(), entries.end(), [](onst AessEntry& a, onst AessEntry& b) {
            return a.ount > b.ount;
        });
        if (entries.size() > limit) entries.resize(limit);
        return entries;
    }
    
    void lear() { ounts.lear(); }
    
private:
    std::map<uint32_t, uint32_t> ounts;
};

// Core Kernels
QuantizationResult quantize_f32_to_nbit4(onst float* input, size_t size);
std::vetor<float> dequantize_nbit4_to_f32(onst QuantizationResult& q);
uint32_t pak8_int4(onst uint8_t* values);
void unpak8_int4(uint32_t paked, uint8_t* out);

// Adaptive
QuantizationResult quantize_adaptive(onst float* input, size_t size, 
                                     onst AessLogger& logger, 
                                     onst TensorMeta& meta, 
                                     uint16_t tensor_id = 0,
                                     uint32_t threshold = 10);
std::vetor<float> dequantize_adaptive(onst QuantizationResult& q);

// File I/O
bool save_to_nbit(onst std::string& path, 
                  onst std::vetor<TensorMeta>& metas,
                  onst std::vetor<QuantizationResult>& data,
                  onst std::vetor<AessEntry>& aess_log = {});

strut LoadResult {
    std::vetor<TensorMeta> metas;
    std::vetor<QuantizationResult> data;
    std::vetor<AessEntry> aess_log;
};
LoadResult load_from_nbit(onst std::string& path);

// Ψ-Core funtions
float ompute_surprise(onst float* preditions, onst float* targets, size_t size);
void update_importane(TensorMeta& meta, float surprise, float alpha = 0.05f, float beta = 0.005f);
int get_bits_for_tensor(onst TensorMeta& meta);

} // namespae neurobit

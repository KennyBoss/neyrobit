#include "neurobit.h"
#include <fstream>
#include <iostream>

namespace neurobit {

// Internal helper for writing raw values
template<typename T>
void write_raw(std::ostream& os, const T& value) {
    os.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

// Internal helper for reading raw values
template<typename T>
void read_raw(std::istream& is, T& value) {
    is.read(reinterpret_cast<char*>(&value), sizeof(T));
}

bool save_to_nbit(const std::string& path, 
                  const std::vector<TensorMeta>& metas,
                  const std::vector<QuantizationResult>& data,
                  const std::vector<AccessEntry>& access_log) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs.is_open()) return false;

    // 1. Write Header
    NBitHeader header;
    header.num_tensors = static_cast<uint32_t>(metas.size());
    if (!access_log.empty()) {
        uint32_t flags = 1; // has_access_log
        // Map Flags field (bytes 12-15)
        std::memcpy(&header.reserved[0], &flags, sizeof(uint32_t));
    }
    ofs.write(reinterpret_cast<const char*>(&header), sizeof(NBitHeader));

    // 2. Write Tensors
    for (size_t i = 0; i < metas.size(); ++i) {
        const auto& meta = metas[i];
        const auto& q = data[i];

        // Meta: Name Length + Name
        uint32_t name_len = static_cast<uint32_t>(meta.name.size());
        write_raw(ofs, name_len);
        ofs.write(meta.name.data(), name_len);

        // Meta: Shape
        uint32_t num_dims = static_cast<uint32_t>(meta.shape.size());
        write_raw(ofs, num_dims);
        for (uint32_t dim : meta.shape) {
            write_raw(ofs, dim);
        }

        // Meta: Stats
        write_raw(ofs, q.num_elements);
        write_raw(ofs, q.scale);

        // Data: Zero Mask
        uint32_t zero_mask_size = static_cast<uint32_t>(q.zero_mask.size());
        write_raw(ofs, zero_mask_size);
        ofs.write(reinterpret_cast<const char*>(q.zero_mask.data()), zero_mask_size);

        // Data: Value Stream
        uint32_t value_stream_size = static_cast<uint32_t>(q.value_stream.size());
        write_raw(ofs, value_stream_size);
        ofs.write(reinterpret_cast<const char*>(q.value_stream.data()), value_stream_size);
    }

    // 3. Write Access Log (if any)
    if (!access_log.empty()) {
        uint32_t log_size = static_cast<uint32_t>(access_log.size());
        write_raw(ofs, log_size);
        for (const auto& entry : access_log) {
            ofs.write(reinterpret_cast<const char*>(&entry), sizeof(AccessEntry));
        }
    }

    return true;
}

LoadResult load_from_nbit(const std::string& path) {
    LoadResult result;
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) return result;

    // 1. Read Header
    NBitHeader header;
    ifs.read(reinterpret_cast<char*>(&header), sizeof(NBitHeader));
    
    // Check magic
    if (std::string(header.magic, 4) != "NB01") {
        throw std::runtime_error("Invalid file format");
    }

    uint32_t flags = 0;
    std::memcpy(&flags, &header.reserved[0], sizeof(uint32_t));
    bool has_access_log = (flags & 1);

    // 2. Read Tensors
    for (uint32_t i = 0; i < header.num_tensors; ++i) {
        TensorMeta meta;
        QuantizationResult q;

        // Meta: Name
        uint32_t name_len;
        read_raw(ifs, name_len);
        meta.name.resize(name_len);
        ifs.read(&meta.name[0], name_len);

        // Meta: Shape
        uint32_t num_dims;
        read_raw(ifs, num_dims);
        meta.shape.resize(num_dims);
        for (uint32_t j = 0; j < num_dims; ++j) {
            read_raw(ifs, meta.shape[j]);
        }

        // Meta: Stats
        read_raw(ifs, q.num_elements);
        read_raw(ifs, q.scale);
        meta.num_elements = q.num_elements;
        meta.scale = q.scale;

        // Data: Zero Mask
        uint32_t zero_mask_size;
        read_raw(ifs, zero_mask_size);
        q.zero_mask.resize(zero_mask_size);
        ifs.read(reinterpret_cast<char*>(q.zero_mask.data()), zero_mask_size);

        // Data: Value Stream
        uint32_t value_stream_size;
        read_raw(ifs, value_stream_size);
        q.value_stream.resize(value_stream_size);
        ifs.read(reinterpret_cast<char*>(q.value_stream.data()), value_stream_size);

        result.metas.push_back(meta);
        result.data.push_back(q);
    }

    // 3. Read Access Log
    if (has_access_log) {
        uint32_t log_size;
        read_raw(ifs, log_size);
        result.access_log.resize(log_size);
        for (uint32_t i = 0; i < log_size; ++i) {
            ifs.read(reinterpret_cast<char*>(&result.access_log[i]), sizeof(AccessEntry));
        }
    }

    return result;
}

} // namespace neurobit

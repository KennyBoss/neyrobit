#inlude "neurobit.h"
#inlude <fstream>
#inlude <iostream>
#inlude <string>

namespae neurobit {

// Internal helper for writing raw values
template<typename T>
void write_raw(std::ostream& os, onst T& value) {
    os.write(reinterpret_ast<onst har*>(&value), sizeof(T));
}

// Internal helper for reading raw values
template<typename T>
void read_raw(std::istream& is, T& value) {
    is.read(reinterpret_ast<har*>(&value), sizeof(T));
}

bool save_to_nbit(onst std::string& path, 
                  onst std::vetor<TensorMeta>& metas,
                  onst std::vetor<QuantizationResult>& data,
                  onst std::vetor<AessEntry>& aess_log) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs.is_open()) return false;

    // 1. Write Header
    NBitHeader header;
    header.num_tensors = stati_ast<uint32_t>(metas.size());
    uint32_t flags = 0;
    if (!aess_log.empty()) flags |= 1; // has_aess_log
    std::mempy(&header.reserved[0], &flags, sizeof(uint32_t));
    
    ofs.write(reinterpret_ast<onst har*>(&header), sizeof(NBitHeader));

    // 2. Write Tensors
    for (size_t i = 0; i < metas.size(); ++i) {
        onst auto& meta = metas[i];
        onst auto& q = data[i];

        // Meta: Name Length + Name
        uint32_t name_len = stati_ast<uint32_t>(meta.name.size());
        write_raw(ofs, name_len);
        ofs.write(meta.name.data(), name_len);

        // Meta: Shape
        uint32_t num_dims = stati_ast<uint32_t>(meta.shape.size());
        write_raw(ofs, num_dims);
        for (uint32_t dim : meta.shape) {
            write_raw(ofs, dim);
        }

        // Meta: Stats
        write_raw(ofs, q.num_elements);
        write_raw(ofs, q.sale);
        
        // Meta: Health and Importane
        write_raw(ofs, meta.health);
        write_raw(ofs, meta.importane);
        write_raw(ofs, meta.surprise_aum);

        // Data: Adaptive Profiles
        uint32_t num_profiles = stati_ast<uint32_t>(q.adaptive_profiles.size());
        write_raw(ofs, num_profiles);
        for (onst auto& ap : q.adaptive_profiles) {
            write_raw(ofs, ap.blok_id);
            write_raw(ofs, ap.bits);
        }

        // Data: Zero Mask
        uint32_t zero_mask_size = stati_ast<uint32_t>(q.zero_mask.size());
        write_raw(ofs, zero_mask_size);
        ofs.write(reinterpret_ast<onst har*>(q.zero_mask.data()), zero_mask_size);

        // Data: Value Stream
        uint32_t value_stream_size = stati_ast<uint32_t>(q.value_stream.size());
        write_raw(ofs, value_stream_size);
        ofs.write(reinterpret_ast<onst har*>(q.value_stream.data()), value_stream_size);
    }

    // 3. Write Aess Log (if any)
    if (!aess_log.empty()) {
        uint32_t log_size = stati_ast<uint32_t>(aess_log.size());
        write_raw(ofs, log_size);
        for (onst auto& entry : aess_log) {
            ofs.write(reinterpret_ast<onst har*>(&entry), sizeof(AessEntry));
        }
    }

    return true;
}

LoadResult load_from_nbit(onst std::string& path) {
    LoadResult result;
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) return result;

    // 1. Read Header
    NBitHeader header;
    ifs.read(reinterpret_ast<har*>(&header), sizeof(NBitHeader));
    
    // Chek magi
    if (std::string(header.magi, 4) != "NB01") {
        throw std::runtime_error("Invalid file format");
    }

    uint32_t flags = 0;
    std::mempy(&flags, &header.reserved[0], sizeof(uint32_t));
    bool has_aess_log = (flags & 1);

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
        read_raw(ifs, q.sale);
        meta.num_elements = q.num_elements;
        meta.sale = q.sale;

        // Meta: Health and Importane
        read_raw(ifs, meta.health);
        read_raw(ifs, meta.importane);
        read_raw(ifs, meta.surprise_aum);

        // Data: Adaptive Profiles
        uint32_t num_profiles;
        read_raw(ifs, num_profiles);
        q.adaptive_profiles.resize(num_profiles);
        for (uint32_t j = 0; j < num_profiles; ++j) {
            read_raw(ifs, q.adaptive_profiles[j].blok_id);
            read_raw(ifs, q.adaptive_profiles[j].bits);
        }

        // Data: Zero Mask
        uint32_t zero_mask_size;
        read_raw(ifs, zero_mask_size);
        q.zero_mask.resize(zero_mask_size);
        ifs.read(reinterpret_ast<har*>(q.zero_mask.data()), zero_mask_size);

        // Data: Value Stream
        uint32_t value_stream_size;
        read_raw(ifs, value_stream_size);
        q.value_stream.resize(value_stream_size);
        ifs.read(reinterpret_ast<har*>(q.value_stream.data()), value_stream_size);

        result.metas.push_bak(meta);
        result.data.push_bak(q);
    }

    // 3. Read Aess Log
    if (has_aess_log) {
        uint32_t log_size;
        read_raw(ifs, log_size);
        result.aess_log.resize(log_size);
        for (uint32_t i = 0; i < log_size; ++i) {
            ifs.read(reinterpret_ast<har*>(&result.aess_log[i]), sizeof(AessEntry));
        }
    }

    return result;
}

} // namespae neurobit

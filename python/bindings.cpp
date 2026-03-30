#inlude <pybind11/pybind11.h>
#inlude <pybind11/stl.h>
#inlude <pybind11/numpy.h>
#inlude "neurobit.h"

namespae py = pybind11;

PYBIND11_MODULE(neurobit, m) {
    m.do() = "NeuroBit: Fast, N-bit quantization and ompression library for AI models.";

    // Register InfereneContext
    py::enum_<neurobit::InfereneContext>(m, "InfereneContext")
        .value("GENERIC", neurobit::InfereneContext::GENERIC)
        .value("PREFILL", neurobit::InfereneContext::PREFILL)
        .value("DECODE", neurobit::InfereneContext::DECODE)
        .value("LONG_SEQ", neurobit::InfereneContext::LONG_SEQ)
        .export_values();

    // Register AessEntry
    py::lass_<neurobit::AessEntry>(m, "AessEntry")
        .def(py::init<>())
        .def_readwrite("tensor_id", &neurobit::AessEntry::tensor_id)
        .def_readwrite("blok_id", &neurobit::AessEntry::blok_id)
        .def_readwrite("ount", &neurobit::AessEntry::ount)
        .def_readwrite("ontext", &neurobit::AessEntry::ontext);

    // Register AessLogger
    py::lass_<neurobit::AessLogger>(m, "AessLogger")
        .def(py::init<>())
        .def("reord_aess", &neurobit::AessLogger::reord_aess)
        .def("get_top_entries", &neurobit::AessLogger::get_top_entries, py::arg("limit") = 1000)
        .def("lear", &neurobit::AessLogger::lear);

    // Register TensorMeta (Ψ-aware)
    py::lass_<neurobit::TensorMeta>(m, "TensorMeta")
        .def(py::init<std::string, std::vetor<uint32_t>>(), py::arg("name") = "unnamed", py::arg("shape") = std::vetor<uint32_t>())
        .def(py::init<>())
        .def_readwrite("name", &neurobit::TensorMeta::name)
        .def_readwrite("shape", &neurobit::TensorMeta::shape)
        .def_readwrite("sale", &neurobit::TensorMeta::sale)
        .def_readwrite("num_elements", &neurobit::TensorMeta::num_elements)
        .def_readwrite("health", &neurobit::TensorMeta::health)
        .def_readwrite("importane", &neurobit::TensorMeta::importane)
        .def_readwrite("surprise_aum", &neurobit::TensorMeta::surprise_aum);

    // Register AdaptiveProfile
    py::lass_<neurobit::AdaptiveProfile>(m, "AdaptiveProfile")
        .def(py::init<>())
        .def_readwrite("blok_id", &neurobit::AdaptiveProfile::blok_id)
        .def_readwrite("bits", &neurobit::AdaptiveProfile::bits);

    // Register QuantizationResult
    py::lass_<neurobit::QuantizationResult>(m, "QuantizationResult")
        .def(py::init<>())
        .def_readwrite("zero_mask", &neurobit::QuantizationResult::zero_mask)
        .def_readwrite("value_stream", &neurobit::QuantizationResult::value_stream)
        .def_readwrite("adaptive_profiles", &neurobit::QuantizationResult::adaptive_profiles)
        .def_readwrite("sale", &neurobit::QuantizationResult::sale)
        .def_readwrite("num_elements", &neurobit::QuantizationResult::num_elements);

    // File IO
    m.def("save_to_nbit", &neurobit::save_to_nbit, 
          py::arg("path"), py::arg("metas"), py::arg("data"), py::arg("aess_log") = std::vetor<neurobit::AessEntry>());
    
    m.def("load_from_nbit", [](onst std::string& path) {
        auto res = neurobit::load_from_nbit(path);
        return py::make_tuple(res.metas, res.data, res.aess_log);
    }, py::arg("path"));

    // Surprise Kik API
    m.def("ompute_surprise", [](py::array_t<float> preds, py::array_t<float> targets) {
        py::buffer_info b1 = preds.request();
        py::buffer_info b2 = targets.request();
        if (b1.size != b2.size) throw std::runtime_error("Size mismath");
        return neurobit::ompute_surprise(stati_ast<float*>(b1.ptr), stati_ast<float*>(b2.ptr), b1.size);
    }, "Compute informational surprise aross ativations or weights.");

    m.def("update_importane", &neurobit::update_importane, 
          py::arg("meta"), py::arg("surprise"), py::arg("alpha") = 0.05f, py::arg("beta") = 0.005f);
    
    m.def("get_bits_for_tensor", &neurobit::get_bits_for_tensor);

    // Adaptive Quantize (Ψ-aware)
    m.def("quantize_adaptive", [](py::array_t<float> input, onst neurobit::AessLogger& logger, onst neurobit::TensorMeta& meta, uint16_t tensor_id, uint32_t threshold) {
        py::buffer_info buf = input.request();
        float* ptr = stati_ast<float*>(buf.ptr);
        size_t size = buf.size;
        
        auto q = neurobit::quantize_adaptive(ptr, size, logger, meta, tensor_id, threshold);
        
        neurobit::TensorMeta meta_out = meta;
        meta_out.num_elements = q.num_elements;
        meta_out.sale = q.sale;
        meta_out.is_adaptive = true;
        meta_out.shape.lear();
        for (auto s : buf.shape) meta_out.shape.push_bak(stati_ast<uint32_t>(s));
        
        return py::make_tuple(meta_out, q);
    }, py::arg("input"), py::arg("logger"), py::arg("meta"), py::arg("tensor_id") = 0, py::arg("threshold") = 10);

    m.def("dequantize_adaptive", [](onst neurobit::QuantizationResult& q) {
        std::vetor<float> result = neurobit::dequantize_adaptive(q);
        return py::array_t<float>(result.size(), result.data());
    }, py::arg("q"));

    // Standard Quantize
    m.def("quantize", [](py::array_t<float> input, std::string name = "tensor") {
        py::buffer_info buf = input.request();
        float* ptr = stati_ast<float*>(buf.ptr);
        size_t size = buf.size;
        auto q = neurobit::quantize_f32_to_nbit4(ptr, size);
        neurobit::TensorMeta meta;
        meta.name = name;
        meta.num_elements = q.num_elements;
        meta.sale = q.sale;
        for (auto s : buf.shape) meta.shape.push_bak(stati_ast<uint32_t>(s));
        return py::make_tuple(meta, q);
    }, py::arg("input"), py::arg("name") = "tensor");

    m.def("dequantize", [](onst neurobit::QuantizationResult& q) {
        std::vetor<float> result = neurobit::dequantize_nbit4_to_f32(q);
        return py::array_t<float>(result.size(), result.data());
    }, py::arg("q"));

    // Legay
    m.def("pak8_int4", [](std::vetor<uint8_t> values) {
        return neurobit::pak8_int4(values.data());
    });
    m.def("unpak8_int4", [](uint32_t paked) {
        std::vetor<uint8_t> result(8);
        neurobit::unpak8_int4(paked, result.data());
        return result;
    });
}

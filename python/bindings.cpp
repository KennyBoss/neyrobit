#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "neurobit.h"

namespace py = pybind11;

PYBIND11_MODULE(neurobit, m) {
    m.doc() = "NeuroBit: Fast, N-bit quantization and compression library for AI models.";

    // Register InferenceContext
    py::enum_<neurobit::InferenceContext>(m, "InferenceContext")
        .value("GENERIC", neurobit::InferenceContext::GENERIC)
        .value("PREFILL", neurobit::InferenceContext::PREFILL)
        .value("DECODE", neurobit::InferenceContext::DECODE)
        .value("LONG_SEQ", neurobit::InferenceContext::LONG_SEQ)
        .export_values();

    // Register AccessEntry
    py::class_<neurobit::AccessEntry>(m, "AccessEntry")
        .def(py::init<>())
        .def_readwrite("tensor_id", &neurobit::AccessEntry::tensor_id)
        .def_readwrite("block_id", &neurobit::AccessEntry::block_id)
        .def_readwrite("count", &neurobit::AccessEntry::count)
        .def_readwrite("context", &neurobit::AccessEntry::context);

    // Register AccessLogger
    py::class_<neurobit::AccessLogger>(m, "AccessLogger")
        .def(py::init<>())
        .def("record_access", &neurobit::AccessLogger::record_access)
        .def("get_top_entries", &neurobit::AccessLogger::get_top_entries, py::arg("limit") = 1000)
        .def("clear", &neurobit::AccessLogger::clear);

    // Register TensorMeta (Ψ-aware)
    py::class_<neurobit::TensorMeta>(m, "TensorMeta")
        .def(py::init<std::string, std::vector<uint32_t>>(), py::arg("name") = "unnamed", py::arg("shape") = std::vector<uint32_t>())
        .def(py::init<>())
        .def_readwrite("name", &neurobit::TensorMeta::name)
        .def_readwrite("shape", &neurobit::TensorMeta::shape)
        .def_readwrite("scale", &neurobit::TensorMeta::scale)
        .def_readwrite("num_elements", &neurobit::TensorMeta::num_elements)
        .def_readwrite("health", &neurobit::TensorMeta::health)
        .def_readwrite("importance", &neurobit::TensorMeta::importance)
        .def_readwrite("surprise_accum", &neurobit::TensorMeta::surprise_accum);

    // Register AdaptiveProfile
    py::class_<neurobit::AdaptiveProfile>(m, "AdaptiveProfile")
        .def(py::init<>())
        .def_readwrite("block_id", &neurobit::AdaptiveProfile::block_id)
        .def_readwrite("bits", &neurobit::AdaptiveProfile::bits);

    // Register QuantizationResult
    py::class_<neurobit::QuantizationResult>(m, "QuantizationResult")
        .def(py::init<>())
        .def_readwrite("zero_mask", &neurobit::QuantizationResult::zero_mask)
        .def_readwrite("value_stream", &neurobit::QuantizationResult::value_stream)
        .def_readwrite("adaptive_profiles", &neurobit::QuantizationResult::adaptive_profiles)
        .def_readwrite("scale", &neurobit::QuantizationResult::scale)
        .def_readwrite("num_elements", &neurobit::QuantizationResult::num_elements);

    // File IO
    m.def("save_to_nbit", &neurobit::save_to_nbit, 
          py::arg("path"), py::arg("metas"), py::arg("data"), py::arg("access_log") = std::vector<neurobit::AccessEntry>());
    
    m.def("load_from_nbit", [](const std::string& path) {
        auto res = neurobit::load_from_nbit(path);
        return py::make_tuple(res.metas, res.data, res.access_log);
    }, py::arg("path"));

    // Surprise Kick API
    m.def("compute_surprise", [](py::array_t<float> preds, py::array_t<float> targets) {
        py::buffer_info b1 = preds.request();
        py::buffer_info b2 = targets.request();
        if (b1.size != b2.size) throw std::runtime_error("Size mismatch");
        return neurobit::compute_surprise(static_cast<float*>(b1.ptr), static_cast<float*>(b2.ptr), b1.size);
    }, "Compute informational surprise across activations or weights.");

    m.def("update_importance", &neurobit::update_importance, 
          py::arg("meta"), py::arg("surprise"), py::arg("alpha") = 0.05f, py::arg("beta") = 0.005f);
    
    m.def("get_bits_for_tensor", &neurobit::get_bits_for_tensor);

    // Adaptive Quantize (Ψ-aware)
    m.def("quantize_adaptive", [](py::array_t<float> input, const neurobit::AccessLogger& logger, const neurobit::TensorMeta& meta, uint16_t tensor_id, uint32_t threshold) {
        py::buffer_info buf = input.request();
        float* ptr = static_cast<float*>(buf.ptr);
        size_t size = buf.size;
        
        auto q = neurobit::quantize_adaptive(ptr, size, logger, meta, tensor_id, threshold);
        
        neurobit::TensorMeta meta_out = meta;
        meta_out.num_elements = q.num_elements;
        meta_out.scale = q.scale;
        meta_out.is_adaptive = true;
        meta_out.shape.clear();
        for (auto s : buf.shape) meta_out.shape.push_back(static_cast<uint32_t>(s));
        
        return py::make_tuple(meta_out, q);
    }, py::arg("input"), py::arg("logger"), py::arg("meta"), py::arg("tensor_id") = 0, py::arg("threshold") = 10);

    m.def("dequantize_adaptive", [](const neurobit::QuantizationResult& q) {
        std::vector<float> result = neurobit::dequantize_adaptive(q);
        return py::array_t<float>(result.size(), result.data());
    }, py::arg("q"));

    // Standard Quantize
    m.def("quantize", [](py::array_t<float> input, std::string name = "tensor") {
        py::buffer_info buf = input.request();
        float* ptr = static_cast<float*>(buf.ptr);
        size_t size = buf.size;
        auto q = neurobit::quantize_f32_to_nbit4(ptr, size);
        neurobit::TensorMeta meta;
        meta.name = name;
        meta.num_elements = q.num_elements;
        meta.scale = q.scale;
        for (auto s : buf.shape) meta.shape.push_back(static_cast<uint32_t>(s));
        return py::make_tuple(meta, q);
    }, py::arg("input"), py::arg("name") = "tensor");

    m.def("dequantize", [](const neurobit::QuantizationResult& q) {
        std::vector<float> result = neurobit::dequantize_nbit4_to_f32(q);
        return py::array_t<float>(result.size(), result.data());
    }, py::arg("q"));

    // Legacy
    m.def("pack8_int4", [](std::vector<uint8_t> values) {
        return neurobit::pack8_int4(values.data());
    });
    m.def("unpack8_int4", [](uint32_t packed) {
        std::vector<uint8_t> result(8);
        neurobit::unpack8_int4(packed, result.data());
        return result;
    });
}

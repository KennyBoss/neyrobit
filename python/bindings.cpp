#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "neurobit.h"

namespace py = pybind11;

PYBIND11_MODULE(neurobit, m) {
    m.doc() = "NeuroBit: Fast, N-bit quantization and compression library for AI models.";

    // Register InferenceContext enum
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

    // Register TensorMeta
    py::class_<neurobit::TensorMeta>(m, "TensorMeta")
        .def(py::init<>())
        .def_readwrite("name", &neurobit::TensorMeta::name)
        .def_readwrite("shape", &neurobit::TensorMeta::shape)
        .def_readwrite("scale", &neurobit::TensorMeta::scale)
        .def_readwrite("num_elements", &neurobit::TensorMeta::num_elements);

    // File IO
    m.def("save_to_nbit", &neurobit::save_to_nbit, 
          py::arg("path"), py::arg("metas"), py::arg("data"), py::arg("access_log") = std::vector<neurobit::AccessEntry>());
    
    m.def("load_from_nbit", [](const std::string& path) {
        auto res = neurobit::load_from_nbit(path);
        return py::make_tuple(res.metas, res.data, res.access_log);
    }, py::arg("path"), "Load .nbit file and return (metas, data, access_log)");

    // Register AdaptiveProfile
    py::class_<neurobit::AdaptiveProfile>(m, "AdaptiveProfile")
        .def(py::init<>())
        .def_readwrite("block_id", &neurobit::AdaptiveProfile::block_id)
        .def_readwrite("bits", &neurobit::AdaptiveProfile::bits);

    // Register QuantizationResult struct
    py::class_<neurobit::QuantizationResult>(m, "QuantizationResult")
        .def(py::init<>())
        .def_readwrite("zero_mask", &neurobit::QuantizationResult::zero_mask)
        .def_readwrite("value_stream", &neurobit::QuantizationResult::value_stream)
        .def_readwrite("adaptive_profiles", &neurobit::QuantizationResult::adaptive_profiles)
        .def_readwrite("scale", &neurobit::QuantizationResult::scale)
        .def_readwrite("num_elements", &neurobit::QuantizationResult::num_elements);

    // Adaptive Quantize
    m.def("quantize_adaptive", [](py::array_t<float> input, const neurobit::AccessLogger& logger, uint16_t tensor_id, uint32_t threshold = 10) {
        py::buffer_info buf = input.request();
        float* ptr = static_cast<float*>(buf.ptr);
        size_t size = buf.size;
        
        return neurobit::quantize_adaptive(ptr, size, logger, tensor_id, threshold);
    }, py::arg("input"), py::arg("logger"), py::arg("tensor_id") = 0, py::arg("threshold") = 10);

    m.def("dequantize_adaptive", &neurobit::dequantize_adaptive);

    // Quantize function with NumPy support
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
    }, py::arg("input"), py::arg("name") = "tensor", "Quantize float32 array and return (meta, quantization_result).");

    // Dequantize function returning NumPy array
    m.def("dequantize", [](const neurobit::QuantizationResult& q) {
        std::vector<float> result = neurobit::dequantize_nbit4_to_f32(q);
        return py::array_t<float>(result.size(), result.data());
    }, py::arg("q"), "Dequantize NBit format back to float32 NumPy array.");

    // Simple pack function (from Stage 1)
    m.def("pack8_int4", [](std::vector<uint8_t> values) {
        if (values.size() != 8) {
            throw std::runtime_error("Input vector must have exactly 8 elements");
        }
        return neurobit::pack8_int4(values.data());
    }, "Pack 8 integers (0-15) into a single 32-bit integer.");

    // Simple unpack function (from Stage 1)
    m.def("unpack8_int4", [](uint32_t packed) {
        std::vector<uint8_t> result(8);
        neurobit::unpack8_int4(packed, result.data());
        return result;
    }, "Unpack a 32-bit integer into 8 integers (0-15).");
}

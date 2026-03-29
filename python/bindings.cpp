#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "neurobit.h"

namespace py = pybind11;

PYBIND11_MODULE(neurobit, m) {
    m.doc() = "NeuroBit: Fast, N-bit quantization and compression library for AI models.";

    // Register QuantizationResult struct
    py::class_<neurobit::QuantizationResult>(m, "QuantizationResult")
        .def(py::init<>())
        .def_readwrite("zero_mask", &neurobit::QuantizationResult::zero_mask)
        .def_readwrite("value_stream", &neurobit::QuantizationResult::value_stream)
        .def_readwrite("scale", &neurobit::QuantizationResult::scale)
        .def_readwrite("num_elements", &neurobit::QuantizationResult::num_elements);

    // Quantize function with NumPy support
    m.def("quantize", [](py::array_t<float> input) {
        py::buffer_info buf = input.request();
        if (buf.ndim != 1) {
            throw std::runtime_error("Input array must be 1D for quantization MVP");
        }
        float* ptr = static_cast<float*>(buf.ptr);
        size_t size = buf.shape[0];
        
        return neurobit::quantize_f32_to_nbit4(ptr, size);
    }, py::arg("input"), "Quantize float32 array to NBit format (INT4 + ZeroMask).");

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

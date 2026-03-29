#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "neurobit.h"

namespace py = pybind11;

PYBIND11_MODULE(neurobit, m) {
    m.doc() = "NeuroBit: Fast, N-bit quantization and compression library for AI models.";

    // Simple pack function
    m.def("pack8_int4", [](std::vector<uint8_t> values) {
        if (values.size() != 8) {
            throw std::runtime_error("Input vector must have exactly 8 elements");
        }
        return neurobit::pack8_int4(values.data());
    }, "Pack 8 integers (0-15) into a single 32-bit integer.");

    // Simple unpack function
    m.def("unpack8_int4", [](uint32_t packed) {
        std::vector<uint8_t> result(8);
        neurobit::unpack8_int4(packed, result.data());
        return result;
    }, "Unpack a 32-bit integer into 8 integers (0-15).");
}

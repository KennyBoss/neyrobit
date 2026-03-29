#include "neurobit.h"

namespace neurobit {

uint32_t pack8_int4(const uint8_t* values) {
    uint32_t result = 0;
    for (int i = 0; i < 8; ++i) {
        // Only keep the lower 4 bits of each value
        result |= (static_cast<uint32_t>(values[i] & 0x0F) << (i * 4));
    }
    return result;
}

void unpack8_int4(uint32_t packed, uint8_t* out_values) {
    for (int i = 0; i < 8; ++i) {
        out_values[i] = static_cast<uint8_t>((packed >> (i * 4)) & 0x0F);
    }
}

} // namespace neurobit

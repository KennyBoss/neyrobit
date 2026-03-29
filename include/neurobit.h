#pragma once

#include <vector>
#include <cstdint>
#include <iostream>

namespace neurobit {

/**
 * @brief Pack 8 values (4-bit each) into a 32-bit integer.
 * 
 * Each of the 8 values must be between 0 and 15 (inclusive).
 * They are packed as: [v7 v6 v5 v4 v3 v2 v1 v0] (v0 is lowest 4 bits).
 */
uint32_t pack8_int4(const uint8_t* values);

/**
 * @brief Unpack a 32-bit integer into 8 values (4-bit each).
 */
void unpack8_int4(uint32_t packed, uint8_t* out_values);

} // namespace neurobit

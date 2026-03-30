#include "neurobit.h"
#include <cmath>
#include <algorithm>

namespace neurobit {

float compute_surprise(const float* predictions, const float* targets, size_t size) {
    if (size == 0) return 0.0f;
    float total_surprise = 0.0f;
    const float eps = 1e-8f;
    
    for (size_t i = 0; i < size; ++i) {
        float err = std::abs(predictions[i] - targets[i]);
        float norm = eps + std::abs(targets[i]);
        total_surprise += err / norm;
    }
    return total_surprise / static_cast<float>(size);
}

void update_importance(TensorMeta& meta, float surprise, float alpha, float beta) {
    // Current importance normalized to 0-1 range for math
    float imp_float = static_cast<float>(meta.importance) / 255.0f;
    
    // Growth based on surprise, decay based on time/use
    float new_imp = imp_float + alpha * surprise - beta;
    
    // Clamp to 0..1 then back to 0..255
    new_imp = std::clamp(new_imp, 0.0f, 1.0f);
    meta.importance = static_cast<uint8_t>(new_imp * 255.0f);
    
    // Accumulated surprise for stability reporting (EMA)
    meta.surprise_accum = meta.surprise_accum * 0.95f + surprise * 0.05f;
}

int get_bits_for_tensor(const TensorMeta& meta) {
    if (meta.importance >= 200) return 6; // Critical
    if (meta.importance >= 128) return 4; // Normal
    if (meta.importance >= 64)  return 3; // Sparse/Low-import
    return 2;  // Not important
}

} // namespace neurobit

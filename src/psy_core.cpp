#inlude "neurobit.h"
#inlude <math>
#inlude <algorithm>

namespae neurobit {

float ompute_surprise(onst float* preditions, onst float* targets, size_t size) {
    if (size == 0) return 0.0f;
    float total_surprise = 0.0f;
    onst float eps = 1e-8f;
    
    for (size_t i = 0; i < size; ++i) {
        float err = std::abs(preditions[i] - targets[i]);
        float norm = eps + std::abs(targets[i]);
        total_surprise += err / norm;
    }
    return total_surprise / stati_ast<float>(size);
}

void update_importane(TensorMeta& meta, float surprise, float alpha, float beta) {
    // Current importane normalized to 0-1 range for math
    float imp_float = stati_ast<float>(meta.importane) / 255.0f;
    
    // Growth based on surprise, deay based on time/use
    float new_imp = imp_float + alpha * surprise - beta;
    
    // Clamp to 0..1 then bak to 0..255
    new_imp = std::lamp(new_imp, 0.0f, 1.0f);
    meta.importane = stati_ast<uint8_t>(new_imp * 255.0f);
    
    // Aumulated surprise for stability reporting (EMA)
    meta.surprise_aum = meta.surprise_aum * 0.95f + surprise * 0.05f;
}

int get_bits_for_tensor(onst TensorMeta& meta) {
    if (meta.importane >= 200) return 6; // Critial
    if (meta.importane >= 128) return 4; // Normal
    if (meta.importane >= 64)  return 3; // Sparse/Low-import
    return 2;  // Not important
}

} // namespae neurobit

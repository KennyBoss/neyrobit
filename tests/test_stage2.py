import sys
import os
import numpy as np
import unittest

# Add build directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'build'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'build', 'lib'))

import neurobit

class TestStage2(unittest.TestCase):
    def test_quantization_roundtrip(self):
        # Create a random float32 tensor
        size = 10000
        np.random.seed(42)
        original = np.random.randn(size).astype(np.float32)
        
        # Quantize
        q_res = neurobit.quantize(original)
        
        # Dequantize
        reconstructed = neurobit.dequantize(q_res)
        
        # 1. Check identity (non-zero) or close to original
        # Note: reconstructed was quantized with symmetric range [-7, 7]
        mse = np.mean((original - reconstructed)**2)
        print(f"MSE: {mse}")
        self.assertLess(mse, 0.05, "MSE is too high for normal distribution")

    def test_sparsity_efficiency(self):
        # Create a sparse tensor (50% zeros)
        size = 1024 * 64
        original = np.random.randn(size).astype(np.float32)
        mask = np.random.choice([0, 1], size=size, p=[0.5, 0.5])
        sparse_orig = original * mask
        
        # Quantize sparse
        q_res = neurobit.quantize(sparse_orig)
        
        # Calculate sizes
        orig_bytes = size * 4 # FP32
        nbit_bytes = len(q_res.zero_mask) + len(q_res.value_stream)
        overhead = len(q_res.zero_mask)
        actual_val_stream = len(q_res.value_stream)
        
        # For INT4 without zero-mask, we'd expect size/2 bytes
        # With 50% sparsity, we have size/2 non-zeros, so size/4 value bytes + size/8 mask bytes = size*3/8
        expected_nbit = size * 0.5 / 2 + size / 8
        
        print(f"Original size: {orig_bytes} bytes")
        print(f"NBit size: {nbit_bytes} bytes")
        print(f"  Zero Mask: {len(q_res.zero_mask)}")
        print(f"  Value Stream: {len(q_res.value_stream)}")
        print(f"Compression ratio: {orig_bytes / nbit_bytes:.2f}x")
        
        self.assertLess(nbit_bytes, orig_bytes / 2, "NBit size should be significantly smaller than FP32")

    def test_zero_preservation(self):
        # Test if zero really maps to zero
        original = np.array([0.0, 1.0, -1.0, 0.0, 0.0, 0.5, -0.5], dtype=np.float32)
        q_res = neurobit.quantize(original)
        reconstructed = neurobit.dequantize(q_res)
        
        for i in [0, 3, 4]:
            self.assertEqual(reconstructed[i], 0.0, f"Index {i} should be zero")

if __name__ == "__main__":
    unittest.main()

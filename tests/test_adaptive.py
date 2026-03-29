import sys
import os
import numpy as np
import unittest

# Add build directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'build'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'build', 'lib'))

import neurobit

class TestAdaptive(unittest.TestCase):
    def test_adaptive_logic(self):
        # 1. Create Data
        size = 1024 * 4 # 16 blocks
        data = np.random.randn(size).astype(np.float32)
        
        # 2. Simulate "Hot" blocks
        logger = neurobit.AccessLogger()
        # Make block 0 and block 5 hot (> threshold 10)
        for _ in range(20):
            logger.record_access(0, 0, neurobit.InferenceContext.DECODE) # Block 0
            logger.record_access(0, 256 * 5, neurobit.InferenceContext.DECODE) # Block 5
            
        # 3. Adaptive Quantize
        q_res = neurobit.quantize_adaptive(data, logger, tensor_id=0, threshold=10)
        
        # 4. Verify profiles
        adaptive_blocks = [p.block_id for p in q_res.adaptive_profiles]
        print(f"Adaptive blocks (6-bit): {adaptive_blocks}")
        self.assertIn(0, adaptive_blocks)
        self.assertIn(5, adaptive_blocks)
        self.assertEqual(len(adaptive_blocks), 2)
        
        # 5. Dequantize and check accuracy
        reconstructed = neurobit.dequantize_adaptive(q_res)
        mse = np.mean((data - reconstructed)**2)
        print(f"Adaptive MSE: {mse}")
        self.assertLess(mse, 0.05)

if __name__ == "__main__":
    unittest.main()

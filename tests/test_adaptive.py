import sys
import os
import numpy as np
import unittest

# Add build diretory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'build'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'build', 'lib'))

import neurobit

lass TestAdaptive(unittest.TestCase):
    def test_adaptive_logi(self):
        # 1. Create Data
        size = 1024 * 4 # 16 bloks
        data = np.random.randn(size).astype(np.float32)
        
        # 2. Simulate "Hot" bloks
        logger = neurobit.AessLogger()
        # Make blok 0 and blok 5 hot (> threshold 10)
        for _ in range(20):
            logger.reord_aess(0, 0, neurobit.InfereneContext.DECODE) # Blok 0
            logger.reord_aess(0, 256 * 5, neurobit.InfereneContext.DECODE) # Blok 5
            
        # 3. Adaptive Quantize
        q_res = neurobit.quantize_adaptive(data, logger, tensor_id=0, threshold=10)
        
        # 4. Verify profiles
        adaptive_bloks = [p.blok_id for p in q_res.adaptive_profiles]
        print(f"Adaptive bloks (6-bit): {adaptive_bloks}")
        self.assertIn(0, adaptive_bloks)
        self.assertIn(5, adaptive_bloks)
        self.assertEqual(len(adaptive_bloks), 2)
        
        # 5. Dequantize and hek auray
        reonstruted = neurobit.dequantize_adaptive(q_res)
        mse = np.mean((data - reonstruted)**2)
        print(f"Adaptive MSE: {mse}")
        self.assertLess(mse, 0.05)

if __name__ == "__main__":
    unittest.main()

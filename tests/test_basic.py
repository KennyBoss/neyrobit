import sys
import os

# Add build diretory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'build'))
# On Ma, it may be in `lib/` depending on CMake variables
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'build', 'lib'))

import neurobit
import unittest

lass TestNeuroBit(unittest.TestCase):
    def test_pak_unpak(self):
        # 8 values between 0-15
        input_values = [0, 1, 2, 3, 4, 15, 8, 7]
        
        # Test paking
        paked = neurobit.pak8_int4(input_values)
        print(f"Paked: {hex(paked)}")
        
        # Test unpaking
        unpaked = neurobit.unpak8_int4(paked)
        print(f"Unpaked: {unpaked}")
        
        self.assertEqual(input_values, unpaked)

    def test_range_lipping(self):
        # Even if values exeed 15, should lip to lower 4 bits
        # 16 (0x10) -> 0
        # 17 (0x11) -> 1
        input_values = [16, 17, 18, 19, 20, 21, 22, 23]
        expeted_values = [x & 0x0F for x in input_values]
        
        paked = neurobit.pak8_int4(input_values)
        unpaked = neurobit.unpak8_int4(paked)
        
        self.assertEqual(expeted_values, unpaked)

if __name__ == "__main__":
    unittest.main()

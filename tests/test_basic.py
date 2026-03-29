import sys
import os

# Add build directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'build'))
# On Mac, it may be in `lib/` depending on CMake variables
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'build', 'lib'))

import neurobit
import unittest

class TestNeuroBit(unittest.TestCase):
    def test_pack_unpack(self):
        # 8 values between 0-15
        input_values = [0, 1, 2, 3, 4, 15, 8, 7]
        
        # Test packing
        packed = neurobit.pack8_int4(input_values)
        print(f"Packed: {hex(packed)}")
        
        # Test unpacking
        unpacked = neurobit.unpack8_int4(packed)
        print(f"Unpacked: {unpacked}")
        
        self.assertEqual(input_values, unpacked)

    def test_range_clipping(self):
        # Even if values exceed 15, should clip to lower 4 bits
        # 16 (0x10) -> 0
        # 17 (0x11) -> 1
        input_values = [16, 17, 18, 19, 20, 21, 22, 23]
        expected_values = [x & 0x0F for x in input_values]
        
        packed = neurobit.pack8_int4(input_values)
        unpacked = neurobit.unpack8_int4(packed)
        
        self.assertEqual(expected_values, unpacked)

if __name__ == "__main__":
    unittest.main()

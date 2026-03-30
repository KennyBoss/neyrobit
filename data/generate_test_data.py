import numpy as np
import os

def generate_random_tensor(shape=(1024, 1024), filename='data/test_layer.npy'):
    print(f"Generating random tensor of shape {shape}...")
    tensor = np.random.randn(*shape).astype(np.float32)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    np.save(filename, tensor)
    print(f"Saved to {filename}")

if __name__ == "__main__":
    generate_random_tensor()

from utils import zca_whitening
from pathlib import Path
from glob import glob
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import h5py
import time

root = "data/upenn/cd02A"
root_path = Path(root)

files = list(glob(str(root_path / "*.npy")))

patch_size = 256
overlap = patch_size // 2


patches = []
for i, f in enumerate(tqdm(files, desc="Extracting patches", ncols=100)):
    im_full = np.load(f)
    h, w = im_full.shape
    for y in range((h - overlap) // overlap):
        for x in range((w - overlap) // overlap):
            patch = im_full[
                (y * overlap) : (y * overlap + patch_size),
                (x * overlap) : (x * overlap + patch_size),
            ].flatten()
            patches.append(patch)

patches = np.stack(patches)

print(f"Extracted {patches.shape[0]} patches of size {patch_size}x{patch_size}")

print("Whitening images via ZCA...")
start = time.time()
# ZCA whitening
patches = (patches - patches.mean(axis=1, keepdims=True)) / (
    patches.std(axis=1, keepdims=True) + 1e-8
)
# np.save("data/cd02A_patches.npy", patches.reshape(-1, patch_size, patch_size))

sigma = np.cov(patches, rowvar=True)
u, s, v = np.linalg.svd(sigma)
zca_matrix = np.dot(u, np.dot(np.diag(1.0 / np.sqrt(s + 1e-6)), u.T))
whitened = np.dot(zca_matrix, patches).reshape(-1, patch_size, patch_size)

elapsed = time.time() - start
print(
    f"Elapsed: {elapsed:.3f} seconds. {float(patches.shape[0]) / elapsed:.3f} patches/sec."
)

print("Saving objects...")
start = time.time()
output = np.stack([patches.reshape(-1, patch_size, patch_size), whitened])
np.save("data/cd02A_patches.npy", output)
elapsed = time.time() - start
print(
    f"Elapsed: {elapsed:.3f} seconds. {float(patches.shape[0]) / elapsed:.3f} patches/sec."
)

# with h5py.File("data/whitened_dataset.h5", mode="w") as f:
#     f.create_dataset("original", data=patches, compression="lzf")
#     f.create_dataset("whitened", data=whitened, compression="lzf")

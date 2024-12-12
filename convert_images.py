import os
from pathlib import Path
import numpy as np
import scipy.io
from tqdm import tqdm

if __name__ == "__main__":
    # Define root directory
    root = "/Users/somamizobuchi/Downloads/cd02A"
    root_path = Path(root)

    # Define output directory
    output_dir = Path("data/upenn/cd02A")
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all files ending with LUM.mat
    files = list(root_path.glob("*LUM.mat"))

    # Sort files to ensure consistent ordering
    files.sort()

    for idx, file_path in tqdm(
        enumerate(files), total=len(files), desc="Converting images"
    ):
        # Load .mat file
        mat_contents = scipy.io.loadmat(str(file_path))

        # Assuming 'LUM_Image' is the variable name in .mat file
        im = mat_contents["LUM_Image"]

        # Process image
        im[im < 0] = 0  # Replace negative values with 0
        im = np.log(im + 1e-5)
        im = (im - np.mean(im)) / np.std(im)

        # Create numbered filename with zero-padding
        output_filename = f"{root_path.name}_{idx:03d}.npy"  # This creates names like cd02A_000.npy, cd02A_001.npy, etc.
        output_path = output_dir / output_filename

        # Save as .npy file
        np.save(output_path, im.astype(np.float32))

import numpy as np
import os
from tqdm import tqdm
from utils import implay
from fixation_utils import generate_bm_fixations


if __name__ == "__main__":
    # Parameters
    num_movies = 50  # Number of images to generate
    NX = 128  # Size of each image
    NT = 128
    D = 20 / 3600
    alpha = 1.0  # Pink noise parameter
    fs = 1000
    ppd = 240

    # Create directory if it doesn't exist
    save_dir = "data/natural_noise"
    os.makedirs(save_dir, exist_ok=True)

    out = np.zeros([num_movies, NT, NX, NX])
    # Generate and save images
    for i in tqdm(range(num_movies), desc="Generating pink noise"):
        # Generate noise image
        noise = generate_bm_fixations(D, fs, ppd, NX, NT, alpha)
        out[i] = noise.transpose(2, 0, 1)

    implay(out[0].transpose(1, 2, 0), interval=10, repeat=True)

    print(out.shape)
    np.save("data/bm_fixation_videos.npy", out)
    print("Done! All images generated and saved.")

import numpy as np
import os
from tqdm import tqdm
from utils import implay
from fixation_utils import generate_pink_noise_movie, generate_3d_pink_noise


if __name__ == "__main__":
    # Parameters
    num_movies = 150  # Number of images to generate
    NX = 128  # Size of each image
    NT = 128  # Number of time frames
    alpha_x = 1  # Pink noise parameter
    alpha_t = 2  # Pink noise parameter

    # Create directory if it doesn't exist
    save_dir = "data/natural_noise"
    os.makedirs(save_dir, exist_ok=True)

    out = np.zeros([num_movies, NT, NX, NX])
    # Generate and save images
    for i in tqdm(range(num_movies), desc="Generating pink noise"):
        # Generate noise image
        # noise = generate_3d_pink_noise((NT, NX, NX), 100, 100, alpha)
        # noise = generate_3d_pink_noise(NX, 1, 1, 100, bbb)
        noise = generate_pink_noise_movie(NX, NT, alpha_x, alpha_t)
        out[i] = noise

    implay(out[0].transpose(1, 2, 0), interval=10, repeat=True)
    # out = out[:, :, :, :, np.newaxis]
    print(out.shape)
    np.save("data/pink_noise_videos.npy", out)
    print("Done! All images generated and saved.")

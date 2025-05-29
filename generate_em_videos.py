import numpy as np
import os
from tqdm import tqdm
from utils import implay
from fixation_utils import pink_noise_gray_image
from em_utils import generate_brownian_motion, generate_saccade
from scipy.stats import gamma


if __name__ == "__main__":
    # Parameters
    num_movies = 100  # Number of images to generate
    NX = 128  # Size of each image
    NT = 1024
    D = 20 / 3600
    alpha = 1.0  # Pink noise parameter
    fs = 1000
    ppd = 240

    # Create directory if it doesn't exist
    save_dir = "data/natural_noise"
    os.makedirs(save_dir, exist_ok=True)

    alpha = 1.5
    beta = 0.08

    imsize = 4096


    out = np.zeros([num_movies, NT, NX, NX])
    img = pink_noise_gray_image(imsize)
    # Generate and save images
    for i in tqdm(range(num_movies), desc="Generating eye movement videos"):
        # Generate image every 10 EMs
        if i % 10 == 0:
            img = pink_noise_gray_image(imsize)

        # Choose a random point in image to start
        eye_idx = np.random.randint((NX/2, imsize-3*NX/2), size=(2, 1))

        while eye_idx.shape[1] < NT:
            # Generate drift
            while True:
                drift_samples = gamma.rvs(alpha, loc=0, scale=beta) * fs
                drift_samples = drift_samples.astype(int)
                if drift_samples < 1:
                    continue
                drift = generate_brownian_motion(D, fs, drift_samples)
                drift = drift * ppd + eye_idx[:,-1:]
                if np.min(drift) >= 0 and np.max(drift + NX) < imsize:
                    break; 
            eye_idx = np.concat((eye_idx, drift.round().astype(int)), axis=1)

            # Generate saccade
            a = saccade_amp = gamma.rvs(1.41, scale=4.87)
            while True:
                theta = np.random.rand() * 360.0
                t, sacc_x, sacc_y, _ = generate_saccade(a, theta, fs)
                sacc = np.stack((sacc_x, sacc_y))
                sacc = sacc * ppd + eye_idx[:,-1:]
                if np.min(sacc) >= 0 and np.max(sacc + NX) < imsize:
                    break; 

            eye_idx = np.concat((eye_idx, sacc.round().astype(int)), axis=1)

        # Trim to fit
        eye_idx = eye_idx[:,:NT]

        for fi in range(min(eye_idx.shape[1], NT)):
            out[i,fi,:,:] = img[eye_idx[1,fi]:eye_idx[1,fi]+NX, eye_idx[0,fi]:eye_idx[0,fi]+NX]
        

    implay(out[0].transpose(1, 2, 0), interval=10, repeat=True)

    np.save("data/em_videos.npy", out)
    print("Done! All images generated and saved.")


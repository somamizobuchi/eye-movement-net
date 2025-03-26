import numpy as np
import os
from tqdm import tqdm
from utils import implay


def generate_pink_noise_movie(
    nx, nt, fs: float = 1.0, ppd: float = 1.0, alpha: float = 1.0
) -> np.ndarray:
    """
    Generate a pink noise image by filtering white Gaussian noise in the frequency domain.

    Args:
        size (tuple): Size of the output image (height/width, time)
        beta (float): Power law exponent (1.0 for pink noise, 0.0 for white noise, 2.0 for brown noise)

    Returns:
        numpy.ndarray: Pink noise image
    """
    # Create white Gaussian noise
    white_noise = np.random.normal(size=(nt, nx, nx))

    # Create frequency coordinates
    f = np.fft.fftfreq(nt, d=(1.0 / fs))
    k = np.fft.fftfreq(nx, d=(1.0 / ppd))

    # Create 3D frequency grid
    freq_t, freq_x, freq_y = np.meshgrid(f, k, k, indexing="ij")
    freq_distance = np.sqrt(freq_t**2 + freq_x**2 + freq_y**2)

    # Avoid division by zero at DC component
    freq_distance[0, 0, 0] = 1.0

    # Create power law filter
    H_pink = freq_distance ** (-alpha)
    H_pink[0, 0, 0] = 0.0  # Set DC component to 0

    # Transform noise to frequency domain
    # Apply filter
    filtered_fft = H_pink * np.exp(1j * 2 * np.pi * np.random.randn(nt, nx, nx))


    # Transform back to spatial domain
    pink_noise = np.fft.ifftn(filtered_fft)

    # Get real component and normalize
    pink_noise = np.real(pink_noise)
    # pink_noise = (pink_noise - pink_noise.min()) / (pink_noise.max() - pink_noise.min())

    return pink_noise


if __name__ == "__main__":
    # Parameters
    num_movies = 50  # Number of images to generate
    NX = 128  # Size of each image
    NT = 128
    alpha = 2.0  # Pink noise parameter

    # Create directory if it doesn't exist
    save_dir = "data/natural_noise"
    os.makedirs(save_dir, exist_ok=True)

    out = np.zeros([num_movies, NT, NX, NX])
    # Generate and save images
    for i in tqdm(range(num_movies), desc="Generating pink noise"):
        # Generate noise image
        noise = generate_pink_noise_movie(NX, NT, fs=10, ppd=1, alpha=alpha)
        out[i] = noise

    implay(out[0].transpose(1, 2, 0), interval=10, repeat=True)
    # out = out[:, :, :, :, np.newaxis]
    print(out.shape)
    np.save("data/pink_noise_videos.npy", out)
    print("Done! All images generated and saved.")

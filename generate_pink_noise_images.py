import numpy as np
import os
from tqdm import tqdm


def generate_pink_noise_image(size=(256, 256)):
    """
    Generate a pink noise image by filtering white Gaussian noise in the frequency domain.

    Args:
        size (tuple): Size of the output image (height, width)
        beta (float): Power law exponent (1.0 for pink noise, 0.0 for white noise, 2.0 for brown noise)

    Returns:
        numpy.ndarray: Pink noise image
    """
    # Create white Gaussian noise
    white_noise = np.random.normal(size=size)

    # Create frequency coordinates
    freq_rows = np.fft.fftfreq(size[0])
    freq_cols = np.fft.fftfreq(size[1])

    # Create 2D frequency grid
    freq_x, freq_y = np.meshgrid(freq_rows, freq_cols, indexing="ij")
    freq_distance = np.sqrt(freq_x**2 + freq_y**2)

    # Avoid division by zero at DC component
    freq_distance[0, 0] = 1.0

    # Create power law filter
    filter_shape = 1.0 / (freq_distance**beta)
    filter_shape[0, 0] = 0.0  # Set DC component to 0

    # Transform noise to frequency domain
    noise_fft = np.fft.fft2(white_noise)

    # Apply filter
    filtered_fft = noise_fft * filter_shape

    # Transform back to spatial domain
    pink_noise = np.fft.ifft2(filtered_fft)

    # Get real component and normalize
    pink_noise = np.real(pink_noise)
    # pink_noise = (pink_noise - pink_noise.min()) / (pink_noise.max() - pink_noise.min())

    return white_noise, pink_noise


if __name__ == "__main__":
    # Parameters
    num_images = 4000  # Number of images to generate
    image_size = (256, 256)  # Size of each image
    beta = 1.0  # Pink noise parameter

    # Create directory if it doesn't exist
    save_dir = "data/natural_noise"
    os.makedirs(save_dir, exist_ok=True)

    out = np.zeros([2, num_images, image_size[0], image_size[1]])
    # Generate and save images
    for i in tqdm(range(num_images), desc="Generating pink noise"):
        # Generate noise image
        noise, pink = generate_pink_noise_image(size=image_size)

        out[0, i] = pink
        out[1, i] = noise

        # Create filename with 4-digit zero padding
        # filename = f"{save_dir}/{i:04d}.npy"

        # Save the noise image
        # np.save(filename, noise.astype(np.float32))

        # Print progress
        # if (i + 1) % 10 == 0:
        #     print(f"Generated {i + 1}/{num_images} images")

    np.save("data/pink_noise.npy", out)

    print("Done! All images generated and saved.")

import numpy as np


def generate_bm_fixations(
    D: float, fs: int, ppd: float, roi_size: int, length: int, alpha: float
) -> np.ndarray:
    """
    Generates a sequence of fixations based on a Brownian motion model and creates a movie of the fixations.

    This function simulates eye movements within a region of interest (ROI)
    using a Brownian motion model. It first generates a 2D Brownian motion trace
    representing the gaze position over time. Then, it creates a spatial
    frequency-filtered noise image and extracts patches from this image
    centered around the gaze positions to simulate fixations.

    Args:
        D (float): Diffusion coefficient, controlling the speed of the Brownian motion (in degrees^2/second).
        fs (int): Sampling frequency, the number of samples per second (in Hz).
        ppd (float): Pixels per degree, the number of pixels corresponding to one degree of visual angle.
        roi_size (int): Size of the region of interest (ROI) in pixels (width and height).
        length (int): Length of the fixation sequence to generate (number of samples).
        alpha (float): Power-law exponent for the frequency response of the noise image.

    Returns:
        np.ndarray: A 3D array of shape (roi_size, roi_size, length) representing a movie of the fixations.
                    Each frame (along the third dimension) is a patch of size roi_size x roi_size
                    extracted from the filtered noise image, centered around the corresponding gaze position.
    """
    # Generate brownian motion trace
    gaze_pos = generate_brownian_motion(D, fs, length)
    gaze_pos = gaze_pos * ppd  # to pixels
    # Calculate min image size as a power of 2
    img_size = int(np.ceil(np.max(np.max(gaze_pos, axis=1) - np.min(gaze_pos, axis=1))))
    img_size = int(2 ** (nextpow2(img_size + roi_size)))
    # Discretize gaze position
    gaze_pos -= gaze_pos.min(axis=1, keepdims=True)
    gaze_pos = np.astype(gaze_pos, np.int32)
    # Generate 2D pink noise
    im = pink_noise_gray_image(img_size, alpha)

    # Crop ROI around gaze position
    mov = np.zeros([roi_size, roi_size, length], dtype=np.float32)
    for t in range(length):
        x = np.astype(gaze_pos[0, t], np.int32)
        y = np.astype(gaze_pos[1, t], np.int32)

        mov[:, :, t] = im[y : (y + roi_size), x : (x + roi_size)]

    return mov


def generate_brownian_motion(D: float, fs: int, length: int) -> np.ndarray:
    """
    Generates a 2D Brownian motion trace.

    Args:
        D (float): Diffusion coefficient.
        fs (int): Sampling frequency.
        length (int): Length of the trace (number of samples).

    Returns:
        np.ndarray: A 2D array of shape (2, length) representing the x and y
                    coordinates of the Brownian motion over time.
    """
    K = np.sqrt(2.0 * D / fs)
    eye_trace = K * np.random.randn(2, length - 1)
    eye_trace = np.concatenate([np.zeros((2, 1)), eye_trace], axis=1)
    eye_trace = np.cumsum(eye_trace, axis=1)
    return eye_trace


def nextpow2(x):
    """Returns the smallest integer exponent p such that 2**p >= abs(x)"""
    return np.ceil(np.log2(np.abs(x))) if x != 0 else 0


def pink_noise_gray_image(size: int, alpha: float = 1.0) -> np.ndarray:
    """
    Generates a pink noise (1/f) grayscale image of a given size.

    This function creates a pink noise image by generating white noise in the
    frequency domain and then applying a 1/f^alpha filter. The result is then
    transformed back to the spatial domain to produce the final image.

    Args:
        size (int): The size of the square image (width and height).
        alpha (float): The power-law exponent for the frequency filter.
                       alpha=1 corresponds to pink noise, alpha=0 to white noise,
                       and alpha=2 to Brownian noise.

    Returns:
        np.ndarray: A 2D numpy array representing the pink noise grayscale image.
    """
    k = np.fft.fftfreq(size)
    k[0] = 1.0
    kr = np.sqrt(k[:, None] ** 2 + k[None, :] ** 2)
    H = 1.0 / (kr**alpha)
    H[0, 0] = 0.0
    Im = H * np.exp(1j * 2 * np.pi * np.random.randn(size, size))

    return np.real(np.fft.ifft2(Im))


def generate_pink_noise_movie(
    nx,
    nt,
    alpha_x: float = 1.0,
    alpha_t: float = 1.0,
) -> np.ndarray:

    # Create frequency coordinates
    f = np.fft.fftfreq(nt)
    k = np.fft.fftfreq(nx)

    ff = abs(f)[:, None, None]
    kk = np.sqrt(k[:, None] ** 2 + k[None, :] ** 2)[None, :, :]

    ff[0] = 1.0
    kk[0, 0] = 1.0

    H_pink = 1.0 / (ff**alpha_t * kk**alpha_x)
    H_pink[0, :, :] = 0.0
    H_pink[:, 0, 0] = 0.0

    # Apply filter
    filtered_fft = H_pink * np.exp(1j * 2 * np.pi * np.random.randn(nt, nx, nx))

    # Transform back to spatial domain
    pink_noise = np.real(np.fft.ifftn(filtered_fft))

    pink_noise = (pink_noise - np.mean(pink_noise)) / np.std(pink_noise)

    return pink_noise

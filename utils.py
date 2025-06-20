import torch
import math

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm, vonmises
import torch
from typing import Any, Tuple, Union


def repeat_first_frame(x: torch.Tensor, n: int) -> torch.Tensor:
    y = torch.cat((x[0, :].unsqueeze(0).repeat(n, 1, 1), x), dim=0)
    return y


def frames_to_image(img_sz: Tuple[int, int, int], retinal_input: torch.Tensor, eye_px):
    assert retinal_input.shape[1] == eye_px.shape[1], "Temporal dimesnsion mismatch!"
    output = torch.zeros(img_sz)
    for batch in range(img_sz[0]):
        for t in range(eye_px.shape[1]):
            frame = retinal_input[batch, t, :]
            x_px = eye_px[batch, t, 0]
            y_px = eye_px[batch, t, 1]
            output[
                batch, y_px : (y_px + frame.shape[0]), x_px : (x_px + frame.shape[1])
            ] += frame
    return output


def accumulate_frames_at_positions(
    canvas_size: Tuple[int, int],
    frames: torch.Tensor,
    positions: torch.Tensor,
    crop_center: bool = False,
) -> torch.Tensor:
    """
    Accumulates multiple frames into a single output image by placing
    each frame at specified positions and summing overlapping regions.

    Args:
        canvas_size: Tuple of (height, width) for the output image
        frames: Tensor of frames to place, shape (num_frames, frame_height, frame_width)
        positions: Tensor of (x,y) positions, shape (num_frames, 2)

    Returns:
        Tensor of shape canvas_size containing the accumulated frames
    """
    assert frames.shape[0] == positions.shape[0], "Temporal dimension mismatch!"

    # Initialize empty output canvas
    canvas = torch.zeros(canvas_size, device=frames.device)

    # For each frame
    for t in range(positions.shape[0]):
        frame = frames[t, :]  # Get current frame
        x_pos = positions[t, 0]  # Get x position
        y_pos = positions[t, 1]  # Get y position

        # Add frame to canvas at specified position
        canvas[
            y_pos : (y_pos + frame.shape[0]), x_pos : (x_pos + frame.shape[1])
        ] += frame

    # Crop center
    if crop_center:
        mean_pos = positions.float().mean(0)
        _, min_idx = torch.square(mean_pos - positions.float()).sum(1).min(dim=0)
        center_pos = positions[min_idx, :]
        return canvas[
            center_pos[1] : center_pos[1] + frame.shape[0],
            center_pos[0] : center_pos[0] + frame.shape[1],
        ]

    return canvas


def apply_roi_mask(
    img: torch.Tensor,
    roi_positions: torch.Tensor,
    roi_size: Tuple[int, int],
    fill: float = 0.0,
    crop_center: bool = False,
    whiten_output: bool = False,
):
    """
    Masks an image to keep only specified regions of interest (ROIs), filling the rest.
    Uses direct array indexing which can be faster for smaller numbers of ROIs.

    Args:
        img: Input image tensor of shape (H, W)
        roi_positions: Tensor of shape (N, 2) containing (x, y) coordinates of ROI top-left corners
        roi_size: Tuple of (width, height) for all ROIs
        fill: Value to fill non-ROI areas with (default: 0.0)

    Returns:
        Tensor of same shape as input with areas outside ROIs filled
    """
    # Crop center
    if crop_center:
        mean_pos = roi_positions.float().mean(0)
        _, min_idx = torch.square(mean_pos - roi_positions.float()).sum(1).min(dim=0)
        center_pos = roi_positions[min_idx, :]
        return img[
            center_pos[1] : center_pos[1] + roi_size[1],
            center_pos[0] : center_pos[0] + roi_size[0],
        ]

    # Initialize boolean mask same size as image (all False)
    mask = torch.full(img.shape, False)

    # Set True for each ROI region using slice indexing
    for t in range(roi_positions.shape[0]):
        mask[
            roi_positions[t, 1] : roi_positions[t, 1] + roi_size[1],
            roi_positions[t, 0] : roi_positions[t, 0] + roi_size[0],
        ] = True

    # Fill all non-ROI areas with fill value
    img[~mask] = fill

    return img


def kernel_images(W, kernel_size, image_channels, rows=None, cols=None, spacing=1):
    """
    Return the kernels as tiled images for visualization
    :return: np.ndarray, shape = [rows * (kernel_size + spacing) - spacing, cols * (kernel_size + spacing) - spacing, 1]
    """

    W /= np.linalg.norm(W, axis=0, keepdims=True)
    W = W.reshape(image_channels, -1, W.shape[-1])

    if rows is None:
        rows = int(np.ceil(math.sqrt(W.shape[-1])))
    if cols is None:
        cols = int(np.ceil(W.shape[-1] / rows))

    kernels = np.ones(
        [
            3,
            rows * (kernel_size + spacing) - spacing,
            cols * (kernel_size + spacing) - spacing,
        ],
        dtype=np.float32,
    )
    coords = [(i, j) for i in range(rows) for j in range(cols)]

    Wt = W.transpose(2, 0, 1)

    for (i, j), weight in zip(coords, Wt):
        kernel = weight.reshape(image_channels, kernel_size, kernel_size) * 2 + 0.5
        x = i * (kernel_size + spacing)
        y = j * (kernel_size + spacing)
        kernels[:, x : x + kernel_size, y : y + kernel_size] = kernel

    return kernels.clip(0, 1)


def plot_convolution(weight: torch.Tensor):
    if torch.is_tensor(weight):
        weight = weight.numpy()
    weight = weight / np.linalg.norm(weight, axis=-1, keepdims=True)

    fig = plt.figure(figsize=(4, 4))
    plt.plot(weight[:, 0, :].T)
    plt.tight_layout()
    fig.canvas.draw()

    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    ncol, nrow = fig.canvas.get_width_height()
    buf = buf.reshape(ncol, nrow, 3)
    plt.close()

    return buf.transpose(2, 0, 1)


# create natural noise image
def natural_noise(size):
    im = np.random.normal(0.0, 100.0, (size, size))

    kx = np.arange(-size / 2, size / 2)
    ky = kx.reshape(-1, 1)

    kx[kx == 0] = 1
    ky[ky == 0] = 1
    kr = 1.0 / np.sqrt(kx**2 + ky**2)

    Im = np.fft.fftshift(np.fft.fft2(im))
    Im = kr * Im

    im = np.real(np.fft.ifft2(np.fft.fftshift(Im)))

    return im


def brownian_eye_trace(
    D: np.double, fs: float, n: int, rng: np.random.Generator = None
) -> np.array:
    """
    Creates simulated eye traces based on brownian motion

    Parameters
    ----------
    D : double
        diffusion constant in arcmin^2/sec
    fs : float
        sampling frequency in Hz
    n : int
        number of samples to generate

    Returns
    -------
    tuple
        A 2-by-n array of eye traces for x and y eye traces
    """
    if rng != None:
        trace = rng.normal(0.0, 1.0, (2, n - 1))
    else:
        trace = np.random.normal(0.0, 1.0, (2, n - 1))

    trace = np.concat((np.array([0, 0])[:, np.newaxis], trace), axis=1)

    K = np.sqrt(2.0 * D / fs)
    return np.cumsum(K * trace, axis=1)


def crop_image(img, roi_size: int, top_left: Tuple[int, int]):
    return img[
        top_left[1] : (top_left[1] + roi_size), top_left[0] : (top_left[0] + roi_size)
    ]


def implay(seq, interval=20, repeat=False, repeat_delay=-1, save_name: str = None):
    """
    Plays a sequence of gray images (2D arrays)

    Parameters
    ----------
    seq : Array
        The input sequence (x, y, t)
    interval : int
        Interval between frames in milliseconds
    """
    fig, ax = plt.subplots()
    video = []
    seq -= seq.min()
    seq /= seq.max()
    for i in range(0, seq.shape[2]):
        roi = seq[:, :, i]
        implt = ax.imshow(roi, animated=True, cmap="gray", vmin=0, vmax=1)
        if i == 0:
            ax.imshow(roi, cmap="gray")
        video.append([implt])

    ani = animation.ArtistAnimation(
        fig,
        video,
        interval=interval,
        blit=True,
        repeat=repeat,
        repeat_delay=repeat_delay,
    )

    if save_name is not None:
        writer = animation.FFMpegWriter(fps=20)
        ani.save(save_name, writer=writer)

    plt.show()

import numpy as np




# def generate_saccade(amplitude_deg: float, angle_radians: float, fs: int = 1000):
#     """
#     Generates a saccade from cumulative gaussian function

#     Parameters
#     ----------
#     amplitude_deg : float
#         The saccade amplitude in degrees
#     angle_radians : float
#         The saccade angle (direction) in radians
#     fs : int
#         The sampling frequency in Hz
#     """
#     # From gaussian
#     peak_velocity = 150 * np.sqrt(amplitude_deg)
#     sigma = 1 / ((peak_velocity / amplitude_deg) * np.sqrt(2 * np.pi))
#     t = np.arange(-sigma * 3, sigma * 3, 1 / fs)
#     pos = amplitude_deg * norm.cdf(t, loc=0, scale=sigma)
#     x = np.cos(angle_radians) * pos
#     y = np.sin(angle_radians) * pos
#     return np.vstack((x, y))


def gen_em_sequence(pre_saccade_drift_samples: int, fs: float, diffusion_const: float):
    """ """
    # Draw amplitude from Gamma distribution
    amplitude = np.random.gamma(1.8, 2)

    # Random direction from multinomial von Mises
    match np.random.randint(4):
        case 0:
            center = 0
            kappa = 10
        case 1:
            center = np.pi
            kappa = 10
        case 2:
            center = np.pi / 2.0
            kappa = 2
        case 3:
            center = -np.pi / 2.0
            kappa = 2
    direction = vonmises.rvs(loc=center, kappa=kappa)

    post_fixation_samples = int((0.15 + np.random.random() * 0.15) * fs)

    drift_pre = (
        brownian_eye_trace(diffusion_const, fs, pre_saccade_drift_samples) / 60.0
    )
    saccade = generate_saccade(amplitude, direction, fs)
    drift_post = brownian_eye_trace(diffusion_const, fs, post_fixation_samples) / 60.0

    trace = np.concat((drift_pre.T, saccade.T + drift_pre[:, -1]))
    trace = np.concat((trace, drift_post.T + trace[-1, :]))

    saccade_start_idx = drift_pre.shape[1] - 1
    drift_start_idx = saccade_start_idx + saccade.shape[1] - 1

    return (trace, amplitude, direction, saccade_start_idx, drift_start_idx)


def decorrelation_loss(input: torch.Tensor) -> torch.Tensor:
    R = torch.corrcoef(input)
    return torch.nn.functional.mse_loss(R, torch.eye(R.shape[0]))


def normalize_unit_variance(
    input: torch.Tensor, dim: int | None = None
) -> torch.Tensor:
    return (input - input.mean(dim=dim, keepdim=True)) / (
        input.std(dim=dim, keepdim=True) + 1e-8
    )


def rescale(input: torch.Tensor) -> torch.Tensor:
    return (input - input.min()) / (input.max() - input.min())


def zca_whitening(input: torch.Tensor) -> torch.Tensor:
    """
    Performs ZCA whitening on the input tensor

    Args:
        input (torch.Tensor): Input tensor of shape (batch_size, n_features)

    Returns:
        torch.Tensor: Whitened tensor of the same shape
    """
    mean = input.mean(dim=1, keepdim=True)
    std = input.std(dim=1, keepdim=True)
    input = (input - mean) / (std + 1e-8)

    sigma = torch.cov(input)

    u, s, _ = torch.linalg.svd(sigma)

    epsilon = 1e-6
    zca_matrix = torch.tensordot(
        u,
        torch.tensordot(torch.diag(1.0 / torch.sqrt(s + epsilon)), u.T, dims=1),
        dims=1,
    )

    return torch.tensordot(zca_matrix, input, dims=1)


def get_keyframe_indices(roi_positions: torch.Tensor) -> torch.Tensor:
    mean_pos = roi_positions.float().mean(dim=1, keepdim=True)
    distance = torch.square(mean_pos - roi_positions).sum(dim=2)
    return distance.min(1).indices


def accumulate_frames(frames: torch.Tensor, offset_indices: torch.Tensor):
    reconstructed = torch.zeros_like(frames[0])
    for t, offset in enumerate(offset_indices.T):
        reconstructed[
            (0 if offset[1] <= 0 else -offset[1]) : (
                offset[1] if offset[1] < 0 else None
            ),
            (0 if offset[0] <= 0 else -offset[0]) : (
                offset[0] if offset[0] < 0 else None
            ),
        ] += frames[
            t,
            (-offset[1] if offset[1] <= 0 else 0) : (
                None if offset[1] <= 0 else offset[1]
            ),
            (-offset[0] if offset[0] <= 0 else 0) : (
                None if offset[0] <= 0 else offset[0]
            ),
        ]

    return reconstructed / frames.shape[0]


def generate_rgc_impulse_response(cell_type='P', num_samples=100, fs=1000):
    """
    Generates the temporal impulse response for a primate retinal ganglion cell.

    This function is based on the linear cascade model described by
    Bernadete and Kaplan. It computes the response for either a
    Parvocellular (P) or Magnocellular (M) cell.

    Args:
        cell_type (str): The type of ganglion cell. Can be 'P' for Parvocellular
                         or 'M' for Magnocellular. Defaults to 'P'.
        num_samples (int): The total number of samples in the impulse response.
                           Defaults to 100.
        fs (int): The sampling frequency in Hertz. Defaults to 1000 Hz.

    Returns:
        tuple: A tuple containing:
            - t (numpy.ndarray): The time vector for the impulse response.
            - combined_response (numpy.ndarray): The impulse response waveform,
              representing the combined effect of the center and surround.
    """
    # --- 1. Set Parameters based on Cell Type ---
    if cell_type.upper() == 'P':
        # Parameters for P-cells (slower, more sustained response)
        n = 5  # Number of cascaded filter stages
        tau = 5.9 / 1000  # Time constant in seconds (5.9 ms)
        surround_delay = 3.5 / 1000 # Average surround delay in seconds (3.5 ms)
        surround_gain = 0.9 # Relative gain of the surround
    elif cell_type.upper() == 'M':
        # Parameters for M-cells (faster, more transient response)
        n = 3  # Number of cascaded filter stages
        tau = 4.0 / 1000  # Time constant in seconds (4.0 ms)
        surround_delay = 3.5 / 1000 # Average surround delay in seconds (3.5 ms)
        surround_gain = 0.95 # Relative gain of the surround
    else:
        raise ValueError("Invalid cell_type. Choose 'P' or 'M'.")

    # --- 2. Create Time Vector ---
    # Generate a time array based on the number of samples and sampling frequency
    t = np.arange(num_samples) / fs

    # --- 3. Calculate Center Impulse Response ---
    # This formula is derived from the gamma distribution, representing the
    # response of n cascaded low-pass filters.
    center_response = (t / tau)**(n - 1) * np.exp(-t / tau) * (1 / (tau * math.factorial(n - 1)))
    
    # Normalize the peak of the center response to 1 for easier interpretation
    if np.max(center_response) > 0:
        center_response /= np.max(center_response)

    # --- 4. Calculate Surround Impulse Response ---
    # The surround is modeled as a delayed, inverted, and scaled version
    # of the center response.
    t_surround = t - surround_delay
    # Ensure time is not negative for the surround calculation
    t_surround[t_surround < 0] = 0 
    
    surround_response = (t_surround / tau)**(n - 1) * np.exp(-t_surround / tau) * (1 / (tau * math.factorial(n - 1)))
    
    # Normalize and scale by the surround gain
    if np.max(surround_response) > 0:
        surround_response /= np.max(surround_response)
    surround_response *= surround_gain

    # --- 5. Combine Center and Surround ---
    # The final receptive field response is the difference between the center
    # and the antagonistic surround.
    combined_response = center_response - surround_response

    return t, combined_response


def generate_rgc_spatial_rf(cell_type='P', eccentricity=5.0, resolution=100, size_samples=100):
    """
    Generates the 2D spatial receptive field for a primate retinal ganglion cell.

    This function is based on the Difference of Gaussians (DoG) model, with
    parameters for center/surround size and gain from Croner & Kaplan (1995).
    It uses a lookup table and interpolation to determine the center size.

    Args:
        cell_type (str): 'P' for Parvocellular or 'M' for Magnocellular.
        eccentricity (float): The distance from the fovea in degrees of visual angle.
                              Affects the size of the receptive field. Defaults to 5.0.
        resolution (int): The resolution of the grid in samples (pixels) per degree
                          of visual angle. Defaults to 100.
        size_samples (int): The total width and height of the spatial grid in samples
                            (pixels). Defaults to 100.

    Returns:
        tuple: A tuple containing:
            - x_grid (numpy.ndarray): A 2D array of x-coordinates for the grid.
            - y_grid (numpy.ndarray): A 2D array of y-coordinates for the grid.
            - rf (numpy.ndarray): A 2D array representing the receptive field sensitivity.
    """
    # 1. Set constant parameters from Croner & Kaplan (1995), Table 1.
    # Data is digitized from the table for specific eccentricity ranges,
    # ignoring the overall summary row for the "0-40" range as requested.
    # We use the midpoint of each specific range for interpolation.
    p_cell_data = {
        # Midpoints for ranges: 0-5, 5-10, 10-20, 20-30, 30-40
        'ecc_mid': np.array([2.5, 7.5, 15, 25, 35]),
        'rc':      np.array([0.03, 0.05, 0.07, 0.09, 0.15]), # Center radius (deg)
        'rs':      np.array([0.18, 0.43, 0.54, 0.73, 0.65]), # Surround radius (deg)
        # Gain ratio is calculated from median Kc and Ks values (Ks/Kc)
        'gain_ratio': np.array([4.4/325.2, 0.7/114.7, 0.6/77.8, 0.8/57.2, 1.1/18.6])
    }
    m_cell_data = {
        # Midpoints for ranges: 0-10, 10-20, 20-30
        'ecc_mid': np.array([5, 15, 25]),
        'rc':      np.array([0.10, 0.18, 0.23]),
        'rs':      np.array([0.72, 1.19, 0.58]),
        'gain_ratio': np.array([1.1/148.0, 2.0/115.0, 1.6/63.8])
    }

    if cell_type.upper() == 'P':
        data = p_cell_data
    elif cell_type.upper() == 'M':
        data = m_cell_data
    else:
        raise ValueError("Invalid cell_type. Choose 'P' or 'M'.")

    # 2. Calculate RF parameters by interpolating from the lookup table
    # sigma_c and sigma_s are the standard deviations of the Gaussians, equivalent to rc and rs
    sigma_c = np.interp(eccentricity, data['ecc_mid'], data['rc'])
    sigma_s = np.interp(eccentricity, data['ecc_mid'], data['rs'])
    surround_gain = np.interp(eccentricity, data['ecc_mid'], data['gain_ratio'])

    # 3. Create the 2D spatial grid based on resolution and sample size
    degrees = size_samples / resolution
    half_degrees = degrees / 2
    x = np.linspace(-half_degrees, half_degrees, size_samples)
    y = np.linspace(-half_degrees, half_degrees, size_samples)
    x_grid, y_grid = np.meshgrid(x, y)

    # 4. Calculate the Center and Surround Gaussian profiles
    dist_sq = x_grid**2 + y_grid**2
    center_gauss = np.exp(-dist_sq / (2 * sigma_c**2))
    surround_gauss = np.exp(-dist_sq / (2 * sigma_s**2))

    # 5. Combine into the Difference of Gaussians (DoG) receptive field
    rf = center_gauss - surround_gain * surround_gauss

    return x_grid, y_grid, rf
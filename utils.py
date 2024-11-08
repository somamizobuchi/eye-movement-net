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

# matplotlib.use("agg")


def repeat_first_frame(x: torch.Tensor, n: int) -> torch.Tensor:
    y = torch.cat((x[0, :].unsqueeze(0).repeat(n, 1, 1), x), dim=0)
    return y


def frames_to_image(img_sz: Tuple[int, int, int], retinal_input: torch.Tensor, eye_px):
    assert (retinal_input.shape[1] == eye_px.shape[1]
            ), "Temporal dimesnsion mismatch!"
    output = torch.zeros(img_sz)
    for batch in range(img_sz[0]):
        for t in range(eye_px.shape[1]):
            frame = retinal_input[batch, t, :]
            x_px = eye_px[batch, t, 0]
            y_px = eye_px[batch, t, 1]
            output[batch, y_px:(y_px + frame.shape[0]),
                   x_px:(x_px + frame.shape[1])] += frame
    return output


def accumulate_frames_at_positions(
    canvas_size: Tuple[int, int],
    frames: torch.Tensor,
    positions: torch.Tensor,
    crop_center: bool = False
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
    assert (frames.shape[0] == positions.shape[0]
            ), "Temporal dimension mismatch!"

    # Initialize empty output canvas
    canvas = torch.zeros(canvas_size)

    # For each frame
    for t in range(positions.shape[0]):
        frame = frames[t, :]         # Get current frame
        x_pos = positions[t, 0]      # Get x position
        y_pos = positions[t, 1]      # Get y position

        # Add frame to canvas at specified position
        canvas[y_pos:(y_pos + frame.shape[0]),
               x_pos:(x_pos + frame.shape[1])] += frame

    # Crop center
    if crop_center:
        mean_pos = positions.float().mean(0)
        _, min_idx = torch.square(
            mean_pos - positions.float()).sum(1).min(dim=0)
        center_pos = positions[min_idx, :]
        return canvas[center_pos[1]:center_pos[1]+frame.shape[0], center_pos[0]:center_pos[0]+frame.shape[1]]

    return canvas


def apply_roi_mask(img: torch.Tensor, roi_positions: torch.Tensor, roi_size: Tuple[int, int], fill: float = 0., crop_center: bool = False):
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
    # Initialize boolean mask same size as image (all False)
    mask = torch.full(img.shape, False)

    # Set True for each ROI region using slice indexing
    for t in range(roi_positions.shape[0]):
        mask[roi_positions[t, 1]:roi_positions[t, 1]+roi_size[1],
             roi_positions[t, 0]:roi_positions[t, 0]+roi_size[0]] = True

    # Fill all non-ROI areas with fill value
    img[~mask] = fill

    # Crop center
    if crop_center:
        mean_pos = roi_positions.float().mean(0)
        _, min_idx = torch.square(
            mean_pos - roi_positions.float()).sum(1).min(dim=0)
        center_pos = roi_positions[min_idx, :]
        return img[center_pos[1]:center_pos[1]+roi_size[1], center_pos[0]:center_pos[0]+roi_size[0]]

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

    kernels = np.ones([3, rows * (kernel_size + spacing) - spacing,
                      cols * (kernel_size + spacing) - spacing], dtype=np.float32)
    coords = [(i, j) for i in range(rows) for j in range(cols)]

    Wt = W.transpose(2, 0, 1)

    for (i, j), weight in zip(coords, Wt):
        kernel = weight.reshape(
            image_channels, kernel_size, kernel_size) * 2 + 0.5
        x = i * (kernel_size + spacing)
        y = j * (kernel_size + spacing)
        kernels[:, x:x+kernel_size, y:y+kernel_size] = kernel

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

    kx = np.arange(-size/2, size/2)
    ky = kx.reshape(-1, 1)

    kx[kx == 0] = 1
    ky[ky == 0] = 1
    kr = 1. / np.sqrt(kx**2 + ky**2)

    Im = np.fft.fftshift(np.fft.fft2(im))
    Im = kr * Im

    im = np.real(np.fft.ifft2(np.fft.fftshift(Im)))

    return im


def brownian_eye_trace(D: np.double, fs: int, n: int, rng: np.random.Generator = None) -> np.array:
    """
    Creates simulated eye traces based on brownian motion

    Parameters
    ----------
    D : double
        diffusion constant in arcmin^2/sec
    fs : int 
        sampling frequency in Hz
    n : int
        number of samples to generate

    Returns
    -------
    tuple
        A 2-by-n array of eye traces for x and y eye traces
    """
    if rng != None:
        trace = rng.normal(0., 1., (2, n-1))
    else:
        trace = np.random.normal(0., 1., (2, n-1))

    trace = np.concat((np.array([0, 0])[:, np.newaxis], trace), axis=1)

    K = np.sqrt(2.*D / fs)
    return np.cumsum(K * trace, axis=1)


def crop_image(img, roi_size: int, top_left: Tuple[int, int]):
    return img[top_left[1]:(top_left[1]+roi_size), top_left[0]:(top_left[0]+roi_size)]


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
    for i in range(0, seq.shape[2]):
        roi = seq[:, :, i]
        implt = ax.imshow(roi, animated=True, cmap='gray')
        if i == 0:
            ax.imshow(roi, cmap='gray')
        video.append([implt])

    ani = animation.ArtistAnimation(
        fig, video, interval=interval, blit=True, repeat=repeat, repeat_delay=repeat_delay)

    if (save_name is not None):
        writer = animation.FFMpegWriter(fps=20)
        ani.save(save_name, writer=writer)

    plt.show()


def generate_saccade(amplitude_deg: float, angle_radians: float, fs: int = 1000):
    """
    Generates a saccade from cumulative gaussian function

    Parameters
    ----------
    amplitude_deg : float
        The saccade amplitude in degrees
    angle_radians : float
        The saccade angle (direction) in radians
    fs : int
        The sampling frequency in Hz
    """
    # From gaussian
    peak_velocity = 150 * np.sqrt(amplitude_deg)
    sigma = 1 / ((peak_velocity / amplitude_deg) * np.sqrt(2*np.pi))
    t = np.arange(-sigma*3, sigma*3, 1/fs)
    pos = amplitude_deg * norm.cdf(t, loc=0, scale=sigma)
    x = np.cos(angle_radians) * pos
    y = np.sin(angle_radians) * pos
    return np.vstack((x, y))


def gen_em_sequence(pre_saccade_drift_samples: int, fs: float, diffusion_const: float):
    """
    """
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
            center = np.pi / 2.
            kappa = 2
        case 3:
            center = -np.pi / 2.
            kappa = 2
    direction = vonmises.rvs(loc=center, kappa=kappa)

    post_fixation_samples = int((0.15 + np.random.random() * 0.15) * fs)

    drift_pre = brownian_eye_trace(
        diffusion_const, fs, pre_saccade_drift_samples) / 60.
    saccade = generate_saccade(amplitude, direction, fs)
    drift_post = brownian_eye_trace(
        diffusion_const, fs, post_fixation_samples) / 60.

    trace = np.concat((drift_pre.T, saccade.T + drift_pre[:, -1]))
    trace = np.concat((trace, drift_post.T + trace[-1, :]))

    saccade_start_idx = drift_pre.shape[1] - 1
    drift_start_idx = saccade_start_idx + saccade.shape[1] - 1

    return (trace, amplitude, direction, saccade_start_idx, drift_start_idx)


def mutual_information_loss(input: torch.Tensor, target: torch.Tensor, bins: int) -> torch.Tensor:
    joint = torch.stack((input.ravel(), target.ravel())).T
    hist2, _ = torch.histogramdd(joint, [bins, bins])
    pxy = hist2 / hist2.sum()
    px = pxy.sum(dim=1)
    py = pxy.sum(dim=0)
    px_py = px[:, None] * py[None, :]
    non_zero = pxy > 0
    return -torch.sum(pxy[non_zero] * torch.log(pxy[non_zero] / px_py[non_zero]))


def decorrelation_loss(input: torch.Tensor) -> torch.Tensor:
    R = torch.corrcoef(input)
    return torch.nn.functional.mse_loss(R, torch.eye(R.shape[0]))


def normalize_unit_variance(input: torch.Tensor) -> torch.Tensor:
    return (input - input.mean()) / input.std()


def rescale(input: torch.Tensor) -> torch.Tensor:
    return (input - input.min()) / (input.max() - input.min())

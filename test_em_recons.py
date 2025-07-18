import numpy as np
import matplotlib.pyplot as plt
from em_utils import generate_brownian_motion, generate_saccade
from fixation_utils import pink_noise_gray_image
from utils import implay
from datasets import ReconDataset

if __name__ == "__main__":
    dataset = ReconDataset(
        img_size=256,
        roi_size=32,
        total_samples=128,
        sampling_frequency=1000,
        diffusion_coefficient=20 / 3600,
        pixels_per_degree=240,
    )
    retinal_input, target, eye_trace = dataset[0]

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].imshow(target.numpy(), cmap="gray")
    axs[0].plot(
        eye_trace[0] + dataset.roi_size / 2,
        eye_trace[1] + dataset.roi_size / 2,
        marker="o",
        linestyle="-",
        markersize=2,
        color="red",
    )
    axs[0].set_title("Target Image")
    axs[1].plot(eye_trace[0], marker="o", linestyle="-", markersize=2)
    axs[1].plot(eye_trace[1], marker="o", linestyle="-", markersize=2)
    axs[1].set_title("Eye Trace")
    plt.tight_layout()
    plt.show()

    # implay(np.permute_dims(retinal_input.numpy(), (1, 2, 0)), 100)

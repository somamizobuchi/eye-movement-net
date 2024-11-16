import datetime
from pathlib import Path

from tqdm import tqdm, trange
from data import EMSequenceDataset, FixationDataset
from model import Encoder
import matplotlib.pyplot as plt
import numpy as np
from utils import (
    apply_roi_mask,
    accumulate_frames_at_positions,
    normalize_unit_variance,
    rescale,
)

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def main():
    kernel_size = 16
    kernel_length = 48
    n_kernels = 96
    fs = 640
    ppd = 180
    diffusion_constant = 15.0 / 3600.0  # deg^2/sec
    drift_samples = 256
    delay_samples = 1

    # Batch size or gradient accumulation steps
    batch_size = 8
    epoch_index = 1

    # Logging
    iterations = 1_000_000 // batch_size
    log_iterations = 1000 // batch_size
    checkpoint_iterations = 50_000 // batch_size

    # Hyper parameters
    alpha = 5  # Jerk energy (smoothness)
    beta = 0.01  # L2 loss

    model = Encoder(kernel_size, kernel_length, n_kernels, fs, delay_samples)

    load_checkpoint = True
    run = "20241113_1648"
    iteration = 187_500

    if load_checkpoint:
        state = torch.load(
            "./checkpoints/{}/{}.pt".format(run, iteration), weights_only=True
        )
        model.load_state_dict(state["model_state_dict"])
    else:
        run = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        Path("./checkpoints/{}".format(run)).mkdir(parents=True, exist_ok=True)
        iteration = 0

    writer = SummaryWriter(log_dir="logs/run_{}".format(run))

    # device = "mps" if torch.backends.mps.is_available() else "cpu"
    device = "cpu"
    model.to(device)

    dataset = FixationDataset(kernel_size, fs, ppd, drift_samples, diffusion_constant)

    data_loader = DataLoader(dataset, shuffle=True, batch_size=batch_size)

    optimizer = torch.optim.Adam(model.parameters())
    running_loss = 0.0
    last_loss = 0.0
    running_mse = 0.0
    running_jerk = 0.0
    running_fr = 0.0

    tq_range = trange(
        iteration, iteration + iterations, desc="Training progress", ncols=100
    )

    # Main loop
    for i in tq_range:

        retinal_input, image, eye_px = next(iter(data_loader))
        # normalize input
        retinal_input = normalize_unit_variance(retinal_input)
        image = normalize_unit_variance(image)

        # Zero the gradient
        optimizer.zero_grad()

        # Encode
        out, _ = model(retinal_input)

        # Spatiotopic reconstruction
        target = torch.stack(
            [
                apply_roi_mask(
                    image[b, :],
                    eye_px[b, model.kernel_length - 1 :, :],
                    (model.kernel_size, model.kernel_size),
                    crop_center=True,
                )
                for b in range(image.shape[0])
            ]
        )

        recons = (
            torch.stack(
                [
                    accumulate_frames_at_positions(
                        image[0].shape,
                        out[b, :],
                        eye_px[b, model.kernel_length - 1 :, :],
                        True,
                    )
                    for b in range(image.shape[0])
                ]
            )
            / out.shape[1]
        )

        # Compute the loss and its gradients
        loss_mse = torch.nn.functional.mse_loss(recons, target) / batch_size
        loss_jerk = alpha * model.jerk_energy_loss()
        loss_fr = beta * (
            model.spatial_kernels.square().mean()
            + model.temporal_kernels.square().mean()
        )
        loss = loss_mse + loss_jerk + loss_fr

        if torch.isnan(loss):
            print(f"NaN detected in loss at iteration {i}, skipping.")
            continue

        loss.backward()
        optimizer.step()

        # Gather data and report
        with torch.no_grad():
            running_loss += loss.item()
            running_mse += loss_mse.item()
            running_jerk += loss_jerk.item()
            running_fr += loss_fr.item()

        if i % log_iterations == log_iterations - 1:
            # Send spatiotemporal kernels
            with torch.no_grad():
                # plot temporal kernels
                fig, ax = plt.subplots()
                ax.plot(model.temporal_kernels_full().detach().numpy().T)
                writer.add_figure("Kernel/temporal", fig, i)

                # Add spatial kernels as images
                ims = model.spatial_kernels.reshape(
                    model.kernel_size, model.kernel_size, 1, -1
                )
                ims = (ims - ims.min()) / (ims.max() - ims.min())
                writer.add_images("Kernel/spatial", ims, i, dataformats="WHCN")

                # Visualize spatial reconstruction
                writer.add_images(
                    "Reconstruction/Spatial",
                    rescale(torch.concat((target, recons), dim=1))[:, None, :, :],
                    i,
                    dataformats="NCHW",
                )

                # Add histogram of ReLU slopes
                writer.add_histogram("Kernel/ReLU_Slope", model.non_linear.slopes, i)

                # visualize reconstruction
                # input = retinal_input[:, kernel_length-1:, :, :]
                # input = (input - input.min()) / (input.max() - input.min())
                # output = out
                # output = (output - output.min()) / \
                #     (output.max() - output.min())
                # tb_retinal_input = torch.cat((input, output), dim=3)
                # writer.add_video("Reconstruction/spatiotemporal",
                #                  vid_tensor=tb_retinal_input[:, :, None, :, :].repeat(1, 1, 3, 1, 1), global_step=i, fps=10)

                # Log loss metrics
                writer.add_scalar("Loss/train", running_loss / log_iterations, i)
                writer.add_scalar("Loss/mse", running_mse / log_iterations, i)
                writer.add_scalar("Loss/jerk", running_jerk / log_iterations, i)
                writer.add_scalar("Loss/firing", running_fr / log_iterations, i)

                # Update progress with loss
                tq_range.set_postfix_str(
                    "Loss={:.5f}".format(running_loss / log_iterations)
                )

                # reset running loss
                running_loss = 0.0
                running_mse = 0.0
                running_jerk = 0.0
                running_fr = 0.0

        if i % checkpoint_iterations == checkpoint_iterations - 1:
            torch.save(
                {
                    "epoch": epoch_index,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": last_loss,
                    "iteration": i + 1,
                },
                "checkpoints/{}/{}.pt".format(run, i + 1),
            )


if __name__ == "__main__":
    main()

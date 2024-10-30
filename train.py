import datetime
from pathlib import Path

from tqdm import tqdm, trange
from data import EMSequenceDataset
from model import Encoder
import matplotlib.pyplot as plt
import numpy as np
from utils import decorrelation_loss

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def main():
    kernel_size = 32
    kernel_length = 32
    n_kernels = 144
    fs = 480
    ppd = 180

    model = Encoder(
        kernel_size,
        kernel_length,
        n_kernels,
        fs
    )

    load_checkpoint = False
    run =  "20241029_0944"
    iteration = 100000

    if load_checkpoint:
        state = torch.load("./checkpoints/{}/{}.pt".format(run, iteration), weights_only=True)
        model.load_state_dict(state["model_state_dict"])
    else:
        run = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        Path("./checkpoints/{}".format(run)).mkdir(parents=True, exist_ok=True)
        iteration = 0
        model.initialize()

    writer = SummaryWriter(log_dir="logs/run_{}".format(run))

    # device = "mps" if torch.backends.mps.is_available() else "cpu"
    device = "cpu"
    model.to(device)

    dataset = EMSequenceDataset(
        kernel_size,
        fs,
        ppd        
    )

    data_loader = DataLoader(dataset,shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    running_loss = 0.
    last_loss = 0.
    running_mse = 0.
    running_fr = 0.
    running_decorr = 0.

    epoch_index = 1
    iterations = 1000000
    log_iterations = 1000 
    checkpoint_iterations = 25000
    alpha = 0.001
    transmission_delay_samples = 16

    n_accumulate_steps = 8

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    tq_range = trange(iteration, iteration + iterations, desc="Training progress", ncols=100)
    for i in tq_range:
        video = next(iter(data_loader)).to(device)
        video = (video - video.mean()) / video.std()

        # Make predictions for this batch
        out, fr = model(video)

        # Compute the loss and its gradients
        loss_mse = torch.nn.functional.mse_loss(out[:,:-transmission_delay_samples,:], video[:,kernel_length-1+transmission_delay_samples:,:,:])
        # loss_fr = torch.square(10. - fr.pow(2.).mean())
        loss_fr = fr.pow(2.).mean()
        # loss_decorr = decorrelation_loss(fr.squeeze())
        loss = (loss_mse + alpha * loss_fr) / n_accumulate_steps
        # loss = loss_mse / n_accumulate_steps
        if torch.isnan(loss):
            print(f"NaN detected in loss at iteration {i}, skipping.")
            continue

        loss.backward()

        # gradient accumulation
        if (i + 1) % n_accumulate_steps == 0:
            # Adjust learning weights
            optimizer.step()
            optimizer.zero_grad()


        # Gather data and report
        with torch.no_grad():
            running_loss += loss.item() * n_accumulate_steps
            running_mse += loss_mse.item()
            running_fr += loss_fr.item()
            # running_decorr += loss_decorr.item()

        if i % log_iterations == log_iterations-1:
            # Send spatiotemporal kernels
            with torch.no_grad():
                # plot temporal kernels
                fig, ax = plt.subplots()
                ax.plot(model.temporal_kernels.detach().numpy().T)
                writer.add_figure("Kernel/temporal", fig, i)

                # Add spatial kernels as images
                ims = model.spatial_kernels.clone().reshape(model.kernel_size, model.kernel_size, 1, -1)
                ims = (ims - ims.min()) / (ims.max() - ims.min())
                writer.add_images("Kernel/spatial", ims, i, dataformats='WHCN')

                # visualize reconstruction
                input = video[:,kernel_length-1+transmission_delay_samples:,:,:]
                input = (input - input.min()) / (input.max() - input.min())
                output = out[:,:-transmission_delay_samples,:]
                output = (output - output.min()) / (output.max() - output.min())
                tb_video = torch.cat((input, output), dim=3)
                writer.add_video("Reconstruction", vid_tensor=tb_video[:,:,None,:,:].repeat(1, 1, 3, 1, 1), global_step=i, fps=10)
            
                # Log loss metrics
                writer.add_scalar('Loss/train', running_loss / log_iterations, i)
                writer.add_scalar('Loss/mse', running_mse / log_iterations, i)
                writer.add_scalar('Loss/firing', alpha * running_fr / log_iterations, i)
                # writer.add_scalar('Loss/decorrelation', running_decorr/ n_accumulate_steps, i)
                tq_range.set_postfix_str("Loss={:.5f}".format(running_loss / log_iterations))

                # reset running loss
                running_loss = 0.
                running_mse = 0.
                running_fr = 0.
                running_decorr = 0.
        
        if i % checkpoint_iterations == checkpoint_iterations - 1:
            torch.save({
                'epoch': epoch_index,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': last_loss,
                'iteration': i+1
            }, "checkpoints/{}/{}.pt".format(
                run,
                i + 1
            ))

if __name__ == "__main__":
    main()
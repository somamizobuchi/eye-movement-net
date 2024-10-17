import datetime
from pathlib import Path

from tqdm import tqdm, trange
from data import EMSequenceDataset
from model import Encoder
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def main():
    kernel_size = 32
    kernel_length = 32
    n_kernels = 32
    fs = 360
    ppd = 180

    start_datetime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    writer = SummaryWriter(log_dir="logs/run_{}".format(start_datetime_str))

    # Create directory for checkpoints
    Path("./checkpoints/{}".format(start_datetime_str)).mkdir(parents=True, exist_ok=True)

    model = Encoder(
        kernel_size,
        kernel_length,
        n_kernels,
        fs
    )

    dataset = EMSequenceDataset(
        kernel_size,
        fs,
        ppd        
    )

    data_loader = DataLoader(dataset, shuffle=True)

    loss_fn = torch.nn.MSELoss()

    model.initialize()


    optimizer = torch.optim.Adam(model.parameters())

    optimizer.zero_grad()

    torch.autograd.set_detect_anomaly(True)

    running_loss = 0.
    last_loss = 0.

    epoch_index = 1
    iterations = 1000000
    log_iterations = 1000 
    checkpoint_iterations = 100000

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    tq_range = trange(iterations, desc="Training progress", ncols=100)
    for i in tq_range:
        video = next(iter(data_loader))
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        out, fr = model(video)

        # Compute the loss and its gradients
        # loss = loss_fn(out, video[:,(kernel_length//2):(-kernel_length//2+1),:,:])
        # loss.backward()

        loss = torch.nn.functional.mse_loss(out, video[:,(kernel_length//2):(-kernel_length//2+1),:,:])
        loss += fr
        # loss += model.temporal_kernels.abs().sum()
        loss.backward()


        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % log_iterations == log_iterations-1:
            # Send spatiotemporal kernels
            fig, ax = plt.subplots()
            ax.plot(model.temporal_kernels.detach().numpy().T)
            writer.add_figure("Kernel/temporal", fig, i)
            ims = model.spatial_kernels.clone().reshape(model.kernel_size, model.kernel_size, 1, -1)
            ims = (ims - ims.min()) / (ims.max() - ims.min())
            writer.add_images("Kernel/spatial", ims, i, dataformats='WHCN')
            
            last_loss = running_loss / log_iterations # loss per batch
            writer.add_scalar('Loss/train', np.log10(last_loss), i)
            tq_range.set_postfix_str("Loss={:.5f}".format(np.log10(running_loss / log_iterations)))
            running_loss = 0.
        
        if i % checkpoint_iterations == checkpoint_iterations - 1:
            torch.save({
                'epoch': epoch_index,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': last_loss,
                'iteration': i+1
            }, "checkpoints/{}/{}.pt".format(
                start_datetime_str,
                i + 1
            ))

if __name__ == "__main__":
    main()
import torch
import matplotlib.pyplot as plt
import numpy as np
from utils import rescale

from model import Encoder


run = "20241108_1958"
iteration = "87500"

kernel_size = 32
kernel_length = 32
n_kernels = 32
delay_samples = 1
fs = 480.0

state = torch.load("./checkpoints/{}/{}.pt".format(run, iteration), weights_only=True)

model = Encoder(kernel_size, kernel_length, n_kernels, fs, delay_samples)

model.load_state_dict(state["model_state_dict"])

# fig = plt.figure()
# gs = fig.add_gridspec(n_kernels, 2, hspace=0, wspace=0)
# axs = gs.subplots()

# fig, axs = plt.subplots(10, 2, gridspec_kw={"wspace": 0, "hspace": 0})
# for k in range(10):
#     axs[k, 0].imshow(
#         model.spatial_kernels[:, k].reshape([kernel_size, kernel_size]).detach().numpy()
#     )
#     axs[k, 1].plot(model.temporal_kernels_full()[k, :].detach().numpy())

#     axs[k, 0].axis("off")
#     axs[k, 1].axis("off")
# plt.show()

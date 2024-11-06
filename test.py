from data import EMSequenceDataset
import utils
import torch 
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from model import Encoder

import time

dataset = EMSequenceDataset(roi_size=64, f_samp_hz=512)
model = Encoder(spatial_kernel_size=64)

dataloader = DataLoader(dataset)

ri, img, pos, sacc_start_idx = next(iter(dataloader))

out, _ = model(ri)

target = utils.apply_roi_mask(img[0,:], pos[0,:], ri.shape[2:])
img = utils.frames_to_image(img.shape, out, pos[:,sacc_start_idx-1:,:])

fig, axs = plt.subplots(2)
axs[0].imshow(target)
axs[1].imshow(img[0,:].detach().numpy())
plt.show()


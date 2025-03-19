from data import WhitenedDataset, PinkNoise3Dataset
import torch
from torch.utils.data import DataLoader
from utils import accumulate_frames, implay

dataset = PinkNoise3Dataset(n_spatial=32, n_temporal=96)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

data = next(iter(dataloader))
print(data.shape)

implay(data[0].numpy().transpose(1, 2, 0), interval=30)

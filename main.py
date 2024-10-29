import torch
import numpy as np
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import Encoder
import tqdm
import utils


from data import EMSequenceDataset


def main():
    model = Encoder()

    dataset = EMSequenceDataset()
    dataloader = DataLoader(dataset=dataset)

    video = next(iter(dataloader))

    output = model(video)

    print(output.shape)

    return 0

    
if __name__ == "__main__":
    main()
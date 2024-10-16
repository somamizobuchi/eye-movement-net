import torch
import numpy as np
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
from torch.utils.tensorboard import SummaryWriter
from model import Encoder
import tqdm
import utils


from data import EMSequenceDataset


def main():
    writer = SummaryWriter("run", "test run")
    # writer.add_figure("Loss/")
    model = Encoder()
    writer.add_images("Kernels/spatial", 
                      model.spatial_kernels.unsqueeze(1).view(-1, 1, model.kernel_size, model.kernel_size),
                      1,
                      dataformats="NCWH")
    
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # model.to(device=device)

    # model(torch.zeros(128, 32, 32).to(device=device))

    # loss_fn = torch.nn.MSELoss()
    
    # dataset = EMSequenceDataset(roi_size=64)
    # training_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

    # for i in range(100):
        # seq = dataset[np.random.randint(100000)]
        # utils.implay(seq.permute(1, 2, 0))
        
    writer.close()

    return 0

    
if __name__ == "__main__":
    main()
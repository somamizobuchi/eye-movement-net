import torch
import numpy as np
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
from model import Encoder
import tqdm
import utils


from data import EMSequenceDataset


def main():
    model = Encoder()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device=device)

    loss_fn = torch.nn.MSELoss()
    
    # for t in tqdm.tqdm(np.arange(100), "Test speed"):
    #     model.forward(input)


    dataset = EMSequenceDataset(roi_size=64)
    training_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

    seq = dataset[0]

    utils.implay(seq.permute(1, 2, 0))


    return 0

    
if __name__ == "__main__":
    main()
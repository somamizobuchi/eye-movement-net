from sympy.solvers.diophantine.diophantine import reconstruct
from data import WhitenedDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from utils import implay, accumulate_frames

dataset = WhitenedDataset("data/whitened_dataset.h5")

# for i in tqdm(range(1000)):
#     input, target, target_idx = dataset[i]
input, target, target_idx, gaze_pos = dataset[int(torch.randint(0, len(dataset), [1,]).item())]

plt.figure()
plt.hist(target)
plt.show()

diff = gaze_pos - gaze_pos[:, target_idx][:,None]
reconstructed = accumulate_frames(input, diff)

fig, axs = plt.subplots(3,1)
axs[0].imshow(input[target_idx])
axs[1].imshow(target)
axs[2].imshow(reconstructed)
plt.show()


# fig, axs = plt.subplots(2,1)
# axs[0].imshow(original)
# axs[1].imshow(whitened)
# plt.show()

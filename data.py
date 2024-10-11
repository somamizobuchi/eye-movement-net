import torch
from torch.utils.data import Dataset
import numpy as np
import utils
from tqdm import tqdm
from glob import glob

class EMSequenceDataset(Dataset):
    def __init__(self,
                 roi_size: int = 32,
                 f_samp_hz: int = 240,
                 pixels_per_degree: float = 240.,
                 n_drift_pad: int = 32) -> None:
        super().__init__()
        
        self.n_drift_pad = n_drift_pad
        self.roi_size = roi_size
        self.fs = f_samp_hz
        self.ppd = pixels_per_degree

        # Load images
        root = "upenn"
        self.files = sorted(glob(f"data/{root}/*.npy"))
        print("Loading {} images from {} ...".format(len(self.files), root))
        self.images= []
        for file in tqdm(self.files, desc="Loading images", ncols=100):
            image = np.load(file, mmap_mode='r')
            self.images.append(image)

    def __len__() -> int:
        return 1000000

    def __getitem__(self, index: int) -> torch.Tensor:
        image_idx = np.random.randint(len(self.images))
        img = self.images[image_idx]
        h, w = img.shape
        start_idx = np.array([
            np.random.randint(w), 
            np.random.randint(h)])
        
        # Generate eye trace and make sure it's within frame
        while True:
            eye_deg = utils.gen_em_sequence(self.n_drift_pad, self.fs, 20)
            eye_px = np.astype(eye_deg[0] * self.ppd, np.int64)
            roi = eye_px.T + start_idx[:,np.newaxis]

            if np.all(np.max(roi + self.roi_size // 2, axis=1) < (w, h)) and np.all(np.min(roi - self.roi_size // 2, axis=1) >= 0):
                break;

        seq = np.zeros((roi.shape[1], self.roi_size, self.roi_size), np.float32)
        for t in range(0, seq.shape[0]):
            seq[t,:,:] = utils.crop_image(img, self.roi_size, roi[:,t])
        
        return torch.from_numpy(seq)



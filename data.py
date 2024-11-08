import torch
from torch.utils.data import Dataset
import numpy as np
import utils
from tqdm import tqdm
from glob import glob
from dataclasses import dataclass
from typing import Tuple

@dataclass
class EMStats:
    trace_deg: torch.Tensor
    trace_px: torch.Tensor
    amplitude: float
    angle_rad: float


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

    def __len__(self) -> int:
        return 1000000

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_idx = np.random.randint(len(self.images))
        img = self.images[image_idx]
        h, w = img.shape
        
        # Generate eye trace and make sure it's within frame
        while True:
            start_idx = np.array([
                np.random.randint(w - self.roi_size), 
                np.random.randint(h - self.roi_size)])
            # eye_deg, amp, theta, sacc_idx, drift_idx = utils.gen_em_sequence(self.n_drift_pad, self.fs, 20)
            eye_deg = utils.brownian_eye_trace(20./3600., self.fs, 128).T
            roi = np.astype(eye_deg * self.ppd, np.int64) + start_idx[None,:]

            if np.all((np.max(roi , axis=0) + self.roi_size) < (w, h)) and np.all(np.min(roi , axis=0) >= 0):
                break

        target = img[roi[:,1].min():roi[:,1].max()+self.roi_size, roi[:,0].min():roi[:,0].max()+self.roi_size].copy()
        roi -= roi.min(axis=0)
        seq = np.zeros([roi.shape[0], self.roi_size, self.roi_size], np.float32)
        for t in range(seq.shape[0]):
            seq[t,:,:] = utils.crop_image(target, self.roi_size, roi[t,:])

        return torch.from_numpy(seq), torch.from_numpy(target), torch.from_numpy(roi)


class FixationDataset(Dataset):
    def __init__(self,
                 roi_size: int = 32,
                 f_samp_hz: int = 240,
                 pixels_per_degree: float = 240.,
                 fixation_length: int = 128,
                 diffusion_constant: float = 20./3600.) -> None:
        super().__init__()
        
        self.fixation_legnth = fixation_length
        self.roi_size = roi_size
        self.fs = f_samp_hz
        self.ppd = pixels_per_degree
        self.D = diffusion_constant

        # Load images
        root = "upenn"
        self.files = sorted(glob(f"data/{root}/*.npy"))
        print("Loading {} images from {} ...".format(len(self.files), root))
        self.images= []
        for file in tqdm(self.files, desc="Loading images", ncols=100):
            image = np.load(file, mmap_mode='r')
            self.images.append(image)

    def __len__(self) -> int:
        return 1000000

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_idx = np.random.randint(len(self.images))
        img = self.images[image_idx]
        h, w = img.shape
        
        # Generate eye trace and make sure it's within frame
        while True:
            start_idx = np.array([
                np.random.randint(w - self.roi_size), 
                np.random.randint(h - self.roi_size)])
            # eye_deg, amp, theta, sacc_idx, drift_idx = utils.gen_em_sequence(self.n_drift_pad, self.fs, 20)
            eye_deg = utils.brownian_eye_trace(self.D, self.fs, self.fixation_legnth).T
            roi = np.astype(eye_deg * self.ppd, np.int64) + start_idx[None,:]

            if np.all((np.max(roi , axis=0) + self.roi_size) < (w, h)) and np.all(np.min(roi , axis=0) >= 0):
                break

        target = img[roi[:,1].min():roi[:,1].max()+self.roi_size, roi[:,0].min():roi[:,0].max()+self.roi_size].copy()
        roi -= roi.min(axis=0)
        seq = np.zeros([roi.shape[0], self.roi_size, self.roi_size], np.float32)
        for t in range(seq.shape[0]):
            seq[t,:,:] = utils.crop_image(target, self.roi_size, roi[t,:])

        return torch.from_numpy(seq), torch.from_numpy(target), torch.from_numpy(roi)
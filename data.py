import math
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
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
    def __init__(
        self,
        roi_size: int = 32,
        f_samp_hz: int = 240,
        pixels_per_degree: float = 240.0,
        n_drift_pad: int = 32,
    ) -> None:
        super().__init__()

        self.n_drift_pad = n_drift_pad
        self.roi_size = roi_size
        self.fs = f_samp_hz
        self.ppd = pixels_per_degree

        # Load images
        root = "natural_noise"
        self.files = sorted(glob(f"data/{root}/*.npy"))
        print("Loading {} images from {} ...".format(len(self.files), root))
        self.images = []
        for file in tqdm(self.files, desc="Loading images", ncols=100):
            image = np.load(file, mmap_mode="r")
            self.images.append(image)

    def __len__(self) -> int:
        return 1000000

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_idx = np.random.randint(len(self.images))
        img = self.images[image_idx]
        h, w = img.shape

        # Generate eye trace and make sure it's within frame
        while True:
            start_idx = np.array(
                [
                    np.random.randint(w - self.roi_size),
                    np.random.randint(h - self.roi_size),
                ]
            )
            # eye_deg, amp, theta, sacc_idx, drift_idx = utils.gen_em_sequence(self.n_drift_pad, self.fs, 20)
            eye_deg = utils.brownian_eye_trace(20.0 / 3600.0, self.fs, 128).T
            roi = np.astype(eye_deg * self.ppd, np.int64) + start_idx[None, :]

            if np.all((np.max(roi, axis=0) + self.roi_size) < (w, h)) and np.all(
                np.min(roi, axis=0) >= 0
            ):
                break

        target = img[
            roi[:, 1].min() : roi[:, 1].max() + self.roi_size,
            roi[:, 0].min() : roi[:, 0].max() + self.roi_size,
        ].copy()
        roi -= roi.min(axis=0)
        seq = np.zeros([roi.shape[0], self.roi_size, self.roi_size], np.float32)
        for t in range(seq.shape[0]):
            seq[t, :, :] = utils.crop_image(target, self.roi_size, roi[t, :])

        return torch.from_numpy(seq), torch.from_numpy(target), torch.from_numpy(roi)


class FixationDataset(Dataset):
    def __init__(
        self,
        roi_size: int = 32,
        f_samp_hz: int = 240,
        pixels_per_degree: float = 240.0,
        fixation_length: int = 128,
        diffusion_constant: float = 20.0 / 3600.0,
    ) -> None:
        super().__init__()

        self.fixation_legnth = fixation_length
        self.roi_size = roi_size
        self.fs = f_samp_hz
        self.ppd = pixels_per_degree
        self.D = diffusion_constant

        # Load images
        root = "upenn/cd02A"
        self.files = sorted(glob(f"data/{root}/*.npy"))
        print("Loading {} images from {} ...".format(len(self.files), root))
        self.images = []
        for file in tqdm(self.files, desc="Loading images", ncols=100):
            image = np.load(file, mmap_mode="r")
            self.images.append(image)

    def __len__(self) -> int:
        return 1000000

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_idx = np.random.randint(len(self.images))
        img = self.images[image_idx]
        h, w = img.shape

        # Generate eye trace and make sure it's within frame
        while True:
            start_idx = np.array(
                [
                    np.random.randint(w - self.roi_size),
                    np.random.randint(h - self.roi_size),
                ]
            )

            eye_deg = utils.brownian_eye_trace(self.D, self.fs, self.fixation_legnth).T
            roi = np.astype(eye_deg * self.ppd, np.int64) + start_idx[None, :]

            if np.all((np.max(roi, axis=0) + self.roi_size) < (w, h)) and np.all(
                np.min(roi, axis=0) >= 0
            ):
                break

        cropped = torch.from_numpy(
            img[
                roi[:, 1].min() : roi[:, 1].max() + self.roi_size,
                roi[:, 0].min() : roi[:, 0].max() + self.roi_size,
            ].copy()
        )

        # Pad to fixed size for batching
        cropped = F.pad(
            cropped,
            (
                0,
                self.roi_size * 8 - cropped.shape[1],
                0,
                self.roi_size * 8 - cropped.shape[0],
            ),
            mode="constant",
            value=0.0,
        )

        roi -= roi.min(axis=0)
        seq = np.zeros([roi.shape[0], self.roi_size, self.roi_size], np.float32)
        for t in range(seq.shape[0]):
            seq[t, :, :] = utils.crop_image(cropped, self.roi_size, roi[t, :])

        return torch.from_numpy(seq), cropped, torch.from_numpy(roi)


class WhitenedDataset(Dataset):
    def __init__(
        self,
        filename: str = "data/cd02A_patches.npy",
        sampling_frequency: float = 360.0,
        pixels_per_degree: float = 180.0,
        roi_size: int = 32,
        fixation_samples: int = 256,
        diffusion_constant: float = 15.0 / 3600.0,
        device: str = "cpu",
    ):
        self.filename = filename
        self.fs = sampling_frequency
        self.ppd = pixels_per_degree
        self.roi_size = roi_size
        self.samples = fixation_samples
        self.d = diffusion_constant
        self.device = device

        # Extract length
        self.data = np.load(filename, mmap_mode="r", allow_pickle=True)
        self.length = self.data.shape[0]

    def __len__(self) -> int:
        return self.length

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        patches = self.data[:, idx].copy()

        original = torch.FloatTensor(patches[0])
        whitened = torch.FloatTensor(patches[1])

        h, w = original.shape

        # Generate drift within bounds of image
        while True:
            x_start = torch.randint(
                0,
                w - self.roi_size,
                [
                    1,
                ],
            )
            y_start = torch.randint(
                0,
                h - self.roi_size,
                [
                    1,
                ],
            )
            gaze_pos = (
                torch.round(self.brownian_eye_trace() * self.ppd)
                + torch.tensor([x_start, y_start])[:, None]
            )
            gaze_pos = gaze_pos.int()

            # Within bounds
            if (
                torch.all(
                    torch.max(gaze_pos + self.roi_size, dim=1).values
                    <= torch.Tensor([w, h])
                )
                and torch.min(gaze_pos) >= 0
            ):
                break

        # Extract ROIs and combine as sequence
        input = torch.stack([self.crop(original, idx) for idx in gaze_pos.T]).to(
            self.device
        )
        target = torch.stack([self.crop(whitened, idx) for idx in gaze_pos.T]).to(
            self.device
        )

        # Find keyframe
        target_idx = self.find_keyframe_index(gaze_pos).to(self.device)
        # target = self.crop(original, gaze_pos[:, target_idx])

        return input, target, target_idx, gaze_pos.to(self.device)

    def brownian_eye_trace(self) -> torch.Tensor:
        """ """
        K = math.sqrt(2.0 * self.d / self.fs)
        eye_trace = K * torch.randn((2, self.samples))
        eye_trace = torch.concat([torch.tensor([0.0, 0.0])[:, None], eye_trace], dim=1)
        eye_trace = torch.cumsum(eye_trace, dim=1)
        return eye_trace[:, :-1]

    def crop(self, img: torch.Tensor, top_left: torch.Tensor | Tuple[int, int]):
        return img[
            top_left[1] : top_left[1] + self.roi_size,
            top_left[0] : top_left[0] + self.roi_size,
        ]

    @staticmethod
    def find_keyframe_index(gaze_coords: torch.Tensor) -> torch.Tensor():
        mean_pos = gaze_coords.float().mean(dim=1, keepdim=True)
        distance = torch.square(mean_pos - gaze_coords).sum(dim=0)
        return torch.min(distance, dim=0).indices

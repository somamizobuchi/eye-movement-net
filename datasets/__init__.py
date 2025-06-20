import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple

from .video_dataset import VideoDataset
from .filtered_video_dataset import FilteredVideoDataset
from datasets import Dataset, np, Tuple, torch



class FilteredVideoDataset(Dataset):
    """
    A dataset class for loading and processing video data.

    Attributes:
        n_spatial (int): Spatial size of the crops.
        n_temporal (int): Temporal length of the crops.
    """

    def __init__(
        self,
        filename_input: str,
        filename_filtered: str,
        n_spatial: int = 32,
        n_temporal: int = 96,
    ):
        """Spatial size of the crops."""
        self.n_spatial = n_spatial
        """Temporal length of the crops."""
        self.n_temporal = n_temporal

        # Extract length
        """The loaded pink noise video data."""
        self.data_in = np.load(filename_input, mmap_mode="r", allow_pickle=True)
        self.data_out = np.load(filename_filtered, mmap_mode="r", allow_pickle=True)

        if self.data_in.shape[0] != self.data_out.shape[0]:
            raise ValueError

        """The number of videos in the dataset."""
        self.length = self.data_in.shape[0]

    def __len__(self) -> int:
        return 10

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        """Randomly select a video index."""
        fi = torch.randint(self.data_in.shape[0], (1,))

        """Extract the crop and convert to float32."""
        coords = torch.randint(self.data_in.shape[2] - self.n_spatial, (2,))
        ti = torch.randint(self.data_in.shape[1] - self.n_temporal, (1,))
        input = (
            self.data_in[
                fi,
                ti : ti + self.n_temporal,
                coords[1] : coords[1] + self.n_spatial,
                coords[0] : coords[0] + self.n_spatial,
            ]
            .copy()
            .astype(np.float32)
        )
        output = (
            self.data_out[
                fi,
                ti : ti + self.n_temporal,
                coords[1] : coords[1] + self.n_spatial,
                coords[0] : coords[0] + self.n_spatial,
            ]
            .copy()
            .astype(np.float32)
        )

        """Normalize the crop."""
        input = (input - input.mean()) / input.std()
        output = (output - output.mean()) / output.std()

        return (torch.from_numpy(input), torch.from_numpy(output))

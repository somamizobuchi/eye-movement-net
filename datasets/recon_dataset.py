from datasets import Dataset, np, Tuple, torch


class ReconDataset(Dataset):
    """
    A dataset class for loading and processing reconstruction data.
    """

    def __init__(
        self,
        img_size: int = 256,
        roi_size: int = 32,
        total_samples: int = 128,
        pad_start: int = 32,
        diffusion_coefficient: float = 20 / 3600,
        sampling_frequency: int = 360,
        pixels_per_degree: int = 240,
    ):
        """Initialize the dataset with parameters."""
        self.img_size = img_size
        self.roi_size = roi_size
        self.total_samples = total_samples
        self.pad_start = pad_start
        self.diffusion_coefficient = diffusion_coefficient
        self.sampling_frequency = sampling_frequency
        self.pixels_per_degree = pixels_per_degree

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return 1_000_000

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        """Generate a sample consisting of a video frame and eye trace."""
        # Generate pink noise gray image
        img = self.pink_noise_gray_image(self.img_size)

        # Generate eye trace
        eye_trace = self.generate_eye_trace()

        # Generate video frames and target image based on the eye trace
        video_frames = np.zeros(
            (self.total_samples, self.roi_size, self.roi_size), dtype=np.float32
        )
        target = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        for i in range(self.total_samples):
            x, y = eye_trace[:, i]
            video_frames[i] = img[y : y + self.roi_size, x : x + self.roi_size]
            if i >= self.pad_start:
                target[y : y + self.roi_size, x : x + self.roi_size] += 1.0 / (
                    self.total_samples - self.pad_start
                )

        target *= img  # Scale target by the original image
        return (torch.from_numpy(video_frames), torch.from_numpy(target), eye_trace)

    def generate_eye_trace(self) -> np.ndarray:
        start_point = np.random.randint(
            0, self.img_size - self.roi_size, size=(2, 1)
        )  # Random starting point
        while True:
            d1 = self.generate_brownian_motion(
                self.diffusion_coefficient, self.sampling_frequency, self.pad_start
            )
            d1 = np.round(d1 * self.pixels_per_degree).astype(int) + start_point
            if np.all(d1 >= 0) and np.all(d1 < self.img_size - self.roi_size):
                break

        while True:
            end_point = np.random.randint(
                0, self.img_size - self.roi_size, size=(2, 1)
            )  # Random end point
            # Generate saccades
            amp = np.linalg.norm(end_point - d1[:, -1]) / self.pixels_per_degree
            theta = np.atan2(
                end_point[1] - d1[1, -1], end_point[0] - d1[0, -1]
            )  # Angle in radians
            theta = np.rad2deg(theta)
            _, sx, sy, _ = self.generate_saccade(amp, theta, self.sampling_frequency)
            s = np.vstack((sx, sy))
            s = np.round(s * self.pixels_per_degree).astype(int) + d1[:, -1:]
            if np.all(s >= 0) and np.all(s < self.img_size - self.roi_size):
                break

        while True:
            d2 = self.generate_brownian_motion(
                self.diffusion_coefficient,
                self.sampling_frequency,
                self.total_samples - self.pad_start - s.shape[1],
            )
            d2 = np.round(d2 * self.pixels_per_degree).astype(int) + s[:, -1:]
            if np.all(d2 >= 0) and np.all(d2 < self.img_size - self.roi_size):
                break

        # Combine drift and saccade
        return np.concatenate((d1, s, d2), axis=1)

    @staticmethod
    def generate_brownian_motion(D: float, fs: int, length: int) -> np.ndarray:
        """
        Generates a 2D Brownian motion trace.

        Args:
            D (float): Diffusion coefficient.
            fs (int): Sampling frequency.
            length (int): Length of the trace (number of samples).

        Returns:
            np.ndarray: A 2D array of shape (2, length) representing the x and y
                        coordinates of the Brownian motion over time.
        """
        K = np.sqrt(2.0 * D / fs)
        eye_trace = K * np.random.randn(2, length - 1)
        eye_trace = np.concatenate([np.zeros((2, 1)), eye_trace], axis=1)
        eye_trace = np.cumsum(eye_trace, axis=1)
        return eye_trace

    @staticmethod
    def generate_saccade(amplitude_deg, direction_deg, sampling_frequency_hz):
        """
        Generate a 2D saccade trajectory based on amplitude, direction, and sampling frequency.

        Parameters:
        - amplitude_deg: float, amplitude of saccade in degrees
        - direction_deg: float, direction of saccade in degrees (0 = right, 90 = up)
        - sampling_frequency_hz: float, number of samples per second

        Returns:
        - time: np.ndarray, time vector in seconds
        - x: np.ndarray, horizontal position over time in degrees
        - y: np.ndarray, vertical position over time in degrees
        - velocity_profile: np.ndarray, velocity over time in deg/s
        """
        direction_rad = np.deg2rad(direction_deg)

        # Saccade main sequence estimates
        peak_velocity = 20 * np.sqrt(amplitude_deg) + 50  # deg/s
        duration_ms = 2.2 * amplitude_deg + 21  # ms
        duration_sec = duration_ms / 1000

        samples = int(duration_sec * sampling_frequency_hz)
        time = np.linspace(0, duration_sec, samples)

        # Normalized time (0 to 1)
        t_norm = time / duration_sec

        # Generate velocity profile (bell-shaped using beta-like function)
        beta_profile = 30 * t_norm**2 * (1 - t_norm) ** 2
        beta_profile /= np.max(beta_profile)

        velocity_profile = peak_velocity * beta_profile
        position_profile = np.cumsum(velocity_profile) / sampling_frequency_hz

        # Project to 2D
        x = position_profile * np.cos(direction_rad)
        y = position_profile * np.sin(direction_rad)

        return time, x, y, velocity_profile

    @staticmethod
    def reconstruct_static_image(
        pos: np.ndarray, video: torch.Tensor, img_size: int
    ) -> torch.Tensor:
        if pos.shape[1] != video.shape[0]:
            raise ValueError("Position array length must match video frame count.")

        roi_size = video.shape[1]
        img = torch.zeros([img_size, img_size], dtype=torch.float32)
        for i in range(pos.shape[1]):
            img[
                pos[1, i] : pos[1, i] + roi_size, pos[0, i] : pos[0, i] + roi_size
            ] += video[i]

        return img

    @staticmethod
    def pink_noise_gray_image(size: int, alpha: float = 1.0) -> np.ndarray:
        """
        Generates a pink noise (1/f) grayscale image of a given size.

        This function creates a pink noise image by generating white noise in the
        frequency domain and then applying a 1/f^alpha filter. The result is then
        transformed back to the spatial domain to produce the final image.

        Args:
            size (int): The size of the square image (width and height).
            alpha (float): The power-law exponent for the frequency filter.
                        alpha=1 corresponds to pink noise, alpha=0 to white noise,
                        and alpha=2 to Brownian noise.

        Returns:
            np.ndarray: A 2D numpy array representing the pink noise grayscale image.
        """
        k = np.fft.fftfreq(size)
        k[0] = 1.0
        kr = np.sqrt(k[:, None] ** 2 + k[None, :] ** 2)
        H = 1.0 / (kr**alpha)
        H[0, 0] = 0.0
        Im = H * np.exp(1j * 2 * np.pi * np.random.randn(size, size))

        return np.real(np.fft.ifft2(Im))

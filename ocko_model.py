import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# -------------------------------
# 1. Generate Pink Noise (same logic)
# -------------------------------
N = 50000
NX = 24  # size in space x
NY = 24  # size in space y
NT = 10  # size in time
ALPHA_X = 2
ALPHA_T = 2

# If you already have the pink noise generated in a .npy file,
# you can just load it. For demonstration, let's show generation:

"""
# Example of direct generation (commented out if already using your .npy)
X = np.zeros((N, NX, NY, NT, 1))

for i in range(N):
    if i % 1000 == 0:
        print("Generating sample", i)
    # Frequencies in x,y
    fX = np.fft.fftfreq(NX, d=1)[:NX]
    fX[-1] = abs(fX[-1])
    gX, gY = np.meshgrid(fX, fX)
    descXY = (gX**2 + gY**2)**(-ALPHA_X/2.0)  # shape in XY
    descXY[0, :] = descXY[1, :]
    descXY[:, 0] = descXY[:, 1]
    descXY[0, 0] = descXY[1, 1]
    
    # Frequencies in t
    fT = np.fft.fftfreq(NT, d=1)[:(NT//2+1)]
    fT[-1] = abs(fT[-1])
    descT = np.abs((fT/10)**(-ALPHA_T))/100.
    descT[0] = descT[1]
    
    # Combine
    descXYT = descXY[:, :, None]*descT[None, None, :]
    
    # White noise
    wn = np.random.randn(NX, NY, NT)
    s = np.fft.rfftn(wn)
    fft_sim = s * descXYT
    vid = np.fft.irfftn(fft_sim, s=wn.shape)
    X[i, :, :, :, 0] = vid

np.save('pinknoise23D.npy', X)
"""

# X = np.load("data/ocko_videos.npy")  # shape (N, NX, NY, NT, 1)
# X = X[:, :NX, :NY, :NT, :]  # ensure shapes are correct

# split_frame = int(0.8 * N)
# X_train = X[:split_frame]
# X_test = X[split_frame:]

# # Convert to torch tensors
# X_train_torch = torch.from_numpy(X_train).float()
# X_test_torch = torch.from_numpy(X_test).float()

# # -------------------------------
# # 2. Create PyTorch Datasets / Loaders
# # -------------------------------
# from torch.utils.data import TensorDataset, DataLoader

# batch_size = 16
# train_ds = TensorDataset(X_train_torch, X_train_torch)  # Input=Target
# test_ds = TensorDataset(X_test_torch, X_test_torch)

# train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

# -------------------------------
# 3. Model Definition
# -------------------------------
# We want something like:
#  - 3D Pad (19,19,7,2) => but in PyTorch, 3D pad is (Dleft, Dright, Htop, Hbottom, Wleft, Wright).
#    We'll interpret your pad: (x: 19 left, 19 right, y: 19 left,19 right, t: 7 left,2 right)
#    Actually it's likely (Z, Y, X) => We'll assume order is (D, H, W).
#
#  - Conv3D with output channels = NTYPE
#  - ReLU
#  - Flatten
#  - A "Dense" => Nx * Ny * Nt * NTYPE -> Nx * Ny * Nt * NTYPE (untrainable W)
#  - Reshape
#  - Gaussian noise
#  - Another 3D pad
#  - Another Conv3D => 1 output channel
#  - Final reshape
#
#  - We also have L2 reg on the first conv, and a custom L1 on the "Dense" layer.
#    We'll handle L1 inside the training loop.
#
#  - We also have that special mask W for the "Dense" layer.


class FixedLinear(nn.Module):
    """
    A linear layer with a fixed weight (no gradient).
    weight shape: [out_features, in_features].
    No bias.
    """

    def __init__(self, in_features, out_features, W_matrix):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Register as buffer so it’s not a parameter:
        self.register_buffer("W", W_matrix)  # shape [out_features, in_features]

    def forward(self, x):
        # x: [batch_size, in_features]
        # W: [out_features, in_features]
        # so output = x.mm(W.T) or (W*x).sum(...) depends on dimension.
        # But PyTorch standard is: out = x @ W^T if W is [out_features, in_features].
        return F.linear(x, self.W, bias=None)


class GaussianNoise(nn.Module):
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma

    def forward(self, x):
        if self.training and self.sigma > 0.0:
            noise = torch.randn_like(x) * self.sigma
            return x + noise
        return x


# We'll define the main model. We’ll keep the conv kernel sizes, strides, etc.
# For simplicity, use ConstantPad3d for zero padding in each conv stage.


class PinkNet(nn.Module):
    def __init__(
        self,
        NTYPE,
        Nneurons,
        KS=20,
        KT=10,
        lambda_l2=50000.0,
        lambda_l1=2.5e6,  # or 2.5e6 * sqrt(10) if that was in your code
        log_noise=-1.0,
    ):
        super().__init__()

        self.NX = NX
        self.NY = NY
        self.NT = NT
        self.KS = KS
        self.KT = KT
        self.NTYPE = NTYPE
        self.Nneurons = Nneurons

        # First 3D padding: (7,2) in time, (19,19) in x, (19,19) in y
        # In PyTorch Conv3d uses NCDHW => (N, C, D, H, W).
        # But you have data shape (N, NX, NY, NT, 1). We'll rearrange dimension ordering inside forward.
        # For the pad we do (left, right, top, bottom, front, back).
        # Let’s interpret:
        #   x => "width" => 19 left, 19 right
        #   y => "height" => 19 left, 19 right
        #   t => "depth" => 7 left, 2 right
        # So pad = (19, 19, 19, 19, 7, 2).
        self.pad1 = nn.ConstantPad3d((19, 19, 19, 19, 7, 2), 0.0)

        # First Conv3D
        # We have in_channels=1, out_channels=NTYPE, kernel_size=(KS, KS, KT).
        # dilation = (2, 2, 1)
        # There's no direct "kernel_regularizer" in PyTorch,
        # we’ll handle L2 via the optimizer or manual.
        # We also set bias=False to replicate Keras code.

        self.conv1 = nn.Conv3d(
            in_channels=1,
            out_channels=NTYPE,
            kernel_size=(KT, KS, KS),
            stride=(1, 1, 1),
            padding="valid",  # we already pad externally
            dilation=(1, 2, 2),
            bias=False,
        )

        self.relu = nn.ReLU()

        # Next: Flatten => a single dimension.
        # The shape after conv1:
        #    out shape = (N, NTYPE, D_out, H_out, W_out)
        # We will flatten in forward().

        # The custom "Dense" layer with dimension Nx*Ny*Nt*NTYPE => same size
        # We'll construct W of shape [output_dim, input_dim].
        # input_dim = (NX*NY*NT*NTYPE).

        in_dim = NX * NY * NT * NTYPE
        out_dim = in_dim

        # Build the mask W:
        #   We'll replicate your logic of “REGULAR LATTICE of neurons”.
        #   Then store it in a buffer for FixedLinear.

        W_np = np.zeros((out_dim, in_dim), dtype=np.float32)

        # Fill according to your logic
        pointer = 0
        idx_count = NX * NY
        # The shape is flatten in (t, x, y, channel?), but from your code:
        #  index = typ + NTYPE*t + NTYPE*NT*(NX*y + x).
        # We'll replicate that indexing for the diagonal entries = 1
        # that exist in Keras code.

        # Summation of Nneurons
        total_neurons = sum(Nneurons)

        for typ in range(NTYPE):
            # Let's define the "density" the same way:
            density = np.sqrt((NX * NY) / float(Nneurons[typ]))
            # We'll keep track of how many we actually set to 1 for this type
            totalcount = 0

            # Lattice pass
            ycount = typ
            for y in range(NY):
                # see your logic for stepping ycount by 1 each row
                # check if ycount%density is < or >= something...
                # The original code was a bit ad-hoc. We'll replicate it as close as possible:
                if ycount % density - density < -1:
                    ycount += 1
                elif ycount % density - density >= -1:
                    ycount += 1
                    xcount = 0
                    for x in range(NX):
                        if xcount % density >= 1:
                            xcount += 1
                        else:
                            xcount += 1
                            totalcount += 1
                            for t in range(NT):
                                idx = typ + NTYPE * t + NTYPE * NT * (NX * y + x)
                                W_np[idx, idx] = 1.0

            # Random fill for mismatch
            # The original code picks random locations in Nx*Ny
            # for the leftover neurons. Then for each leftover we set diagonal=1
            leftover = Nneurons[typ] - totalcount
            if leftover > 0:
                Px = np.random.permutation(NX * NY)
                for x_ in range(leftover):
                    for t in range(NT):
                        idx = typ + NTYPE * t + NTYPE * NT * Px[x_]
                        W_np[idx, idx] = 1.0

        W_torch = torch.from_numpy(W_np)
        self.fixed_linear = FixedLinear(in_dim, out_dim, W_torch)

        # Next: Gaussian noise
        self.gauss_noise = GaussianNoise(10**log_noise)

        # Second padding: (10,9) in x, (10,9) in y, (7,2) in t => (9,10,9,10,7,2)
        self.pad2 = nn.ConstantPad3d(
            (10, 9, 10, 9, 7, 2), 0.0
        )  # Adjust if your code changes

        # Then final Conv3D => 1 output channel
        self.conv2 = nn.Conv3d(
            in_channels=NTYPE,
            out_channels=1,
            kernel_size=(KT, KS, KS),
            stride=(1, 1, 1),
            padding="valid",
            dilation=(1, 1, 1),
            bias=False,
        )

    def forward(self, x):
        """
        x shape: (N, NX, NY, NT, 1) in your Keras code.
        PyTorch conv3d expects (N, C, D, H, W).
        We'll interpret:
          C = 1
          D = NT (time)
          H = NY
          W = NX
        But you have them as (N, X, Y, T, C).
        We'll permute dims: from (N, X, Y, T, 1) => (N, 1, T, Y, X).
        """
        # re-arrange:
        x = x.permute(0, 4, 3, 2, 1)  # (N, 1, T, Y, X)

        # pad1
        x = self.pad1(x)  # zero-pad

        # conv1
        x = self.conv1(x)  # shape => (N, NTYPE, D_out, H_out, W_out)

        # ReLU
        x = self.relu(x)

        # Flatten
        # we can do: x = x.view(x.size(0), -1)
        x = x.view(x.size(0), -1)

        # Fixed linear
        x = self.fixed_linear(
            x
        )  # shape => (N, in_dim) => (N, out_dim), out_dim = in_dim

        # Reshape back to (N, NTYPE, D, H, W)
        # out_dim = NX*NY*NT*NTYPE => we want (N, NTYPE, NT, NY, NX)
        # So we do:
        x = x.view(-1, self.NTYPE, self.NT, self.NY, self.NX)

        # Gaussian noise
        x = self.gauss_noise(x)

        # Second pad
        x = self.pad2(x)

        # conv2 => shape (N, 1, D2, H2, W2)
        x = self.conv2(x)

        # Final reshape => match original shape => (N, NX, NY, NT, 1)
        # But internally we have (N, 1, T, Y, X). We want (N, X, Y, T, 1).
        x = x.permute(0, 4, 3, 2, 1)  # => (N, X, Y, T, 1)

        return x


# -------------------------------
# 4. Instantiate / Train
# -------------------------------
# Example hyperparameters
# NTYPE = 4  # or however many you want
# Nneurons = [36, 144, 36, 144]  # from your example
# model = PinkNet(
#     NTYPE=NTYPE,
#     Nneurons=Nneurons,
#     KS=20,
#     KT=10,
#     lambda_l2=50000.0,
#     lambda_l1=2.5e6,  # example
#     log_noise=-1.0,
# )

# # We'll place the model on GPU if available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# # Define an optimizer, e.g. Adam
# # We'll incorporate L2 weight decay (for conv1, conv2).
# # For the fixed_linear we don't want to update its weight,
# # so that doesn't matter for weight decay. We'll exclude it.
# params_with_decay = []
# for name, param in model.named_parameters():
#     # The fixed_linear has no params (they're buffers).
#     if param.requires_grad:
#         params_with_decay.append(param)

# optimizer = optim.Adam(params_with_decay, lr=1e-3, weight_decay=0.0)  # adjust as needed

# # We'll define a typical MSE loss.
# criterion = nn.MSELoss()


# If we want an L1 penalty specifically on the "fixed_linear" output or something else,
# we can add it manually in the training loop. But remember it's a fixed weight,
# so no gradient. Possibly you wanted L1 on conv1 or conv2's weights?
# We'll show a typical approach for L1 on conv2's weights as an example:
def l1_penalty_on_conv2(model, factor):
    l1_loss = 0.0
    for name, param in model.conv2.named_parameters():
        if "weight" in name:
            l1_loss += torch.sum(torch.abs(param))
    return factor * l1_loss


# num_epochs = 10

# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#     for batch_idx, (data_in, data_out) in enumerate(train_loader):
#         data_in = data_in.to(device)
#         data_out = data_out.to(device)

#         optimizer.zero_grad()

#         pred = model(data_in)

#         loss = criterion(pred, data_out)

#         # If we want a custom L1 on conv2 or conv1:
#         #   e.g. loss += 0.0001 * l1_penalty_on_conv2(model, factor=1.0)
#         # Adjust factor to match your Keras code’s scale
#         # We'll skip here or set factor=some_value
#         # e.g.:
#         # loss += l1_penalty_on_conv2(model, factor=1e-7)

#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()

#     train_loss = running_loss / len(train_loader)

#     # Evaluate on test set
#     model.eval()
#     test_loss = 0.0
#     with torch.no_grad():
#         for batch_idx, (data_in, data_out) in enumerate(test_loader):
#             data_in = data_in.to(device)
#             data_out = data_out.to(device)
#             pred = model(data_in)
#             loss = criterion(pred, data_out)
#             test_loss += loss.item()

#     test_loss /= len(test_loader)

#     print(
#         f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.5f}, Test Loss: {test_loss:.5f}"
#     )

# If you need to extract the conv1 filters or do an FFT:
# weights_conv1 = model.conv1.weight.data.cpu().numpy()
# Then do your analysis as needed.

# Similarly, you can compute the layer outputs by just forwarding data through the model
# and looking at intermediate activations. In PyTorch, you can do forward hooks or just
# break up the forward pass.

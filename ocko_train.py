import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
from datetime import datetime
from ocko_model import PinkNet, l1_penalty_on_conv2
import torchvision


def create_pink_noise_dataloader(batch_size=16):
    """Create dataloaders for pink noise dataset"""
    NX = 24
    NY = 24
    NT = 10

    # Load the data
    X = np.load("data/ocko_videos.npy")
    X = X[:, :NX, :NY, :NT, :]  # ensure shapes are correct

    # Split into train and test
    split_frame = int(0.8 * X.shape[0])
    X_train = X[:split_frame]
    X_test = X[split_frame:]

    # Convert to torch tensors
    X_train_torch = torch.from_numpy(X_train).float()
    X_test_torch = torch.from_numpy(X_test).float()

    # Create datasets
    train_ds = TensorDataset(X_train_torch, X_train_torch)
    test_ds = TensorDataset(X_test_torch, X_test_torch)

    # Create dataloaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def log_conv_kernels(writer, model, step):
    """Log convolution kernel visualizations to tensorboard"""
    # Log Conv1 kernels
    conv1_weights = model.conv1.weight.data.cpu()
    # Reshape for visualization if needed
    B, C, D, H, W = conv1_weights.shape
    conv1_grid = torchvision.utils.make_grid(
        conv1_weights.view(B * C, 1, D * H, W), normalize=True, nrow=4
    )
    writer.add_image("Conv1/Kernels", conv1_grid, step)

    # Log Conv2 kernels
    conv2_weights = model.conv2.weight.data.cpu()
    B, C, D, H, W = conv2_weights.shape
    conv2_grid = torchvision.utils.make_grid(
        conv2_weights.view(B * C, 1, D * H, W), normalize=True, nrow=2
    )
    writer.add_image("Conv2/Kernels", conv2_grid, step)


def train_model(
    model,
    train_loader,
    test_loader,
    num_epochs=10,
    learning_rate=1e-3,
    device="cuda",
    log_dir="runs",
    save_dir="checkpoints",
):
    """
    Train the PinkNet model with full logging and checkpointing
    """
    # Create directories if they don't exist
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    # Initialize tensorboard writer
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(os.path.join(log_dir, current_time))

    # Setup optimizer
    params_with_decay = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params_with_decay, lr=learning_rate, weight_decay=0.0)
    criterion = nn.MSELoss()

    # Training loop
    best_test_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0

        # Training progress bar
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch_idx, (data_in, data_out) in enumerate(train_pbar):
            data_in = data_in.to(device)
            data_out = data_out.to(device)

            optimizer.zero_grad()
            pred = model(data_in)
            loss = criterion(pred, data_out)

            # Add L1 penalty if needed
            l1_loss = l1_penalty_on_conv2(model, factor=1e-7)
            total_loss = loss + l1_loss

            total_loss.backward()
            optimizer.step()

            # Update progress bar
            epoch_train_loss += loss.item()
            train_pbar.set_postfix(
                {"loss": f"{loss.item():.5f}", "l1_penalty": f"{l1_loss.item():.5f}"}
            )

            # Log batch metrics
            global_step = epoch * len(train_loader) + batch_idx

            # Log convolution kernels
            if global_step % 5 == 0:
                writer.add_scalar("Loss/train_batch", loss.item(), global_step)
                writer.add_scalar("L1_Penalty/train_batch", l1_loss.item(), global_step)
                log_conv_kernels(writer, model, global_step)

        avg_train_loss = epoch_train_loss / len(train_loader)

        # Evaluation phase
        model.eval()
        epoch_test_loss = 0.0

        test_pbar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Test]")
        with torch.no_grad():
            for data_in, data_out in test_pbar:
                data_in = data_in.to(device)
                data_out = data_out.to(device)

                pred = model(data_in)
                loss = criterion(pred, data_out)
                epoch_test_loss += loss.item()

                test_pbar.set_postfix({"loss": f"{loss.item():.5f}"})

        avg_test_loss = epoch_test_loss / len(test_loader)

        # Log epoch metrics
        writer.add_scalar("Loss/train_epoch", avg_train_loss, epoch)
        writer.add_scalar("Loss/test_epoch", avg_test_loss, epoch)

        # Save model if it's the best so far
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            checkpoint_path = os.path.join(save_dir, f"pinknet_best.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": avg_train_loss,
                    "test_loss": avg_test_loss,
                },
                checkpoint_path,
            )

        # Save periodic checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(save_dir, f"pinknet_epoch_{epoch+1}.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": avg_train_loss,
                    "test_loss": avg_test_loss,
                },
                checkpoint_path,
            )

    writer.close()
    return model


# Example usage:
if __name__ == "__main__":
    # Create dataloaders
    train_loader, test_loader = create_pink_noise_dataloader(batch_size=16)

    # Initialize model
    NTYPE = 4
    Nneurons = [36, 144, 36, 144]
    model = PinkNet(
        NTYPE=NTYPE,
        Nneurons=Nneurons,
        KS=20,
        KT=10,
        lambda_l2=50000.0,
        lambda_l1=2.5e6,
        log_noise=-1.0,
    )

    # Move to device
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    print(device)

    model.to(device)

    # Train model
    model = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=3,
        learning_rate=1e-3,
        device=device,
        log_dir="runs/pinknet",
        save_dir="checkpoints/pinknet",
    )

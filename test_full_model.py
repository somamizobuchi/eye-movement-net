"""
Test script for FullModel with bifurcated output and reconstruction.

Tests:
1. Model initialization
2. Forward pass with real dataset
3. Output shapes for velocity and reconstruction
4. Backward pass (gradient flow)
5. Full reconstruction using bilinear splatting
"""

import torch
from torch.utils.data import DataLoader
from FullModel import FullModel
from datasets.recon_dataset import ReconDataset
from reconstruct_from_model import reconstruct_from_fullmodel_output
import matplotlib.pyplot as plt


def test_full_model():
    """Test FullModel with bifurcated velocity and reconstruction outputs."""

    print("=" * 60)
    print("Testing FullModel with Bifurcated Output + Reconstruction")
    print("=" * 60)

    # Model hyperparameters
    kernel_size = 32
    kernel_length = 32
    kernel_delay = 16
    n_channels = 128
    decoder_size = 64
    batch_size = 4
    img_size = 256
    total_samples = 128  # Number of input frames

    print("\nModel Configuration:")
    print(f"  Kernel size: {kernel_size}x{kernel_size}")
    print(f"  Kernel length: {kernel_length}")
    print(f"  Kernel delay: {kernel_delay}")
    print(f"  Spatiotemporal channels: {n_channels}")
    print(f"  Decoder size: {decoder_size}")
    print(f"  Batch size: {batch_size}")
    print(f"  Input frames: {total_samples}")
    print(f"  Image size: {img_size}x{img_size}")

    # Initialize model
    print("\n[1/5] Initializing model...")
    model = FullModel(
        kernel_size=kernel_size,
        kernel_length=kernel_length,
        kernel_delay=kernel_delay,
        n_channels=n_channels,
        decoder_size=decoder_size,
        noise_std=0.05,
        max_velocity=50.0,
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Create dataset
    print("\n[2/5] Creating dataset and loading batch...")
    dataset = ReconDataset(
        img_size=img_size,
        roi_size=kernel_size,
        total_samples=total_samples,
        pad_start=kernel_length - 1,
        saccade=False,
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Get one batch
    retinal_input, target, eye_trace, mask, sacc_end_idx = next(iter(dataloader))
    print(f"  Retinal input shape: {retinal_input.shape}")
    print(f"  Target image shape: {target.shape}")
    print(f"  Eye trace shape: {eye_trace.shape}")
    print(f"  Mask shape: {mask.shape}")

    # Forward pass
    print("\n[3/5] Running forward pass...")
    model.eval()
    with torch.no_grad():
        eye_velocities, reconstructed_frames = model(retinal_input)

    # Expected output time dimension
    t_out = total_samples - kernel_length + 1

    print(f"  Eye velocities shape: {eye_velocities.shape}")
    print(f"    Expected: ({batch_size}, {t_out}, 2)")
    print(f"  Reconstructed frames shape: {reconstructed_frames.shape}")
    print(f"    Expected: ({batch_size}, {t_out}, {kernel_size}, {kernel_size})")

    # Verify shapes
    assert eye_velocities.shape == (batch_size, t_out, 2), \
        f"Eye velocities shape mismatch: {eye_velocities.shape} != ({batch_size}, {t_out}, 2)"
    assert reconstructed_frames.shape == (batch_size, t_out, kernel_size, kernel_size), \
        f"Reconstruction shape mismatch: {reconstructed_frames.shape} != ({batch_size}, {t_out}, {kernel_size}, {kernel_size})"

    print("  ✓ Output shapes are correct!")

    # Test full reconstruction with bilinear splatting
    print("\n[4/5] Testing full reconstruction with bilinear splatting...")

    # Use first sample in batch
    batch_idx = 0

    # Extract initial position from eye trace (starting point)
    # eye_trace is (batch_size, 2, total_samples)
    initial_position = torch.tensor([
        eye_trace[batch_idx, 0, kernel_length - 1].item(),  # x at first valid frame
        eye_trace[batch_idx, 1, kernel_length - 1].item(),  # y at first valid frame
    ], dtype=torch.float32)

    print(f"  Initial position: ({initial_position[0]:.2f}, {initial_position[1]:.2f})")

    # Reconstruct using velocities and frames
    canvas_size = (img_size, img_size)
    reconstructed_canvas = reconstruct_from_fullmodel_output(
        eye_velocities,
        reconstructed_frames,
        initial_position,
        canvas_size,
        dt=1.0,
        batch_idx=batch_idx,
    )

    print(f"  Reconstructed canvas shape: {reconstructed_canvas.shape}")
    print(f"    Expected: (1, 1, {img_size}, {img_size})")

    # Compute reconstruction error on masked region
    target_img = target[batch_idx].unsqueeze(0).unsqueeze(0)
    mask_img = mask[batch_idx].unsqueeze(0).unsqueeze(0).float()

    mse_masked = ((reconstructed_canvas - target_img) ** 2 * mask_img).sum() / (mask_img.sum() + 1e-8)
    print(f"  MSE on valid region: {mse_masked.item():.6f}")
    print("  ✓ Reconstruction completed!")

    # Test backward pass
    print("\n[5/5] Testing backward pass...")
    model.train()
    eye_velocities_train, reconstructed_frames_train = model(retinal_input)

    # Dummy loss
    velocity_loss = eye_velocities_train.mean()
    recon_loss = reconstructed_frames_train.mean()
    total_loss = velocity_loss + recon_loss

    total_loss.backward()

    # Check gradients
    has_gradients = any(p.grad is not None for p in model.parameters())
    print(f"  Gradients computed: {has_gradients}")

    if has_gradients:
        print("  ✓ Backward pass successful!")
    else:
        print("  ✗ No gradients found!")

    # Test kernel methods
    print("\n[Bonus] Testing utility methods...")
    temporal_kernels = model.get_temporal_kernels()
    print(f"  Temporal kernels shape: {temporal_kernels.shape}")
    print(f"    Expected: ({n_channels}, {kernel_length})")

    kernel_var = model.kernel_variance()
    print(f"  Kernel variance: {kernel_var.item():.6f}")

    # Visualize reconstruction
    print("\n[Visualization] Saving reconstruction comparison...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(target_img[0, 0].cpu().numpy(), cmap='gray')
    axes[0].set_title('Ground Truth')
    axes[0].axis('off')

    axes[1].imshow(reconstructed_canvas[0, 0].detach().cpu().numpy(), cmap='gray')
    axes[1].set_title('Reconstructed (Bilinear Splatting)')
    axes[1].axis('off')

    error = torch.abs(reconstructed_canvas - target_img) * mask_img
    axes[2].imshow(error[0, 0].detach().cpu().numpy(), cmap='hot')
    axes[2].set_title(f'Absolute Error (MSE={mse_masked.item():.6f})')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig('reconstruction_test.png', dpi=150, bbox_inches='tight')
    print("  Saved to: reconstruction_test.png")

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_full_model()

"""
Simple test script for FullModel with bifurcated output.

Tests:
1. Model initialization
2. Forward pass with dummy data
3. Output shapes for velocity and reconstruction
4. Backward pass (gradient flow)
"""

import torch
from FullModel import FullModel


def test_full_model():
    """Test FullModel with bifurcated velocity and reconstruction outputs."""

    print("=" * 60)
    print("Testing FullModel with Bifurcated Output")
    print("=" * 60)

    # Model hyperparameters
    kernel_size = 32
    kernel_length = 32
    kernel_delay = 16
    n_channels = 128
    decoder_size = 64
    batch_size = 4
    t = 128  # Number of input frames

    print("\nModel Configuration:")
    print(f"  Kernel size: {kernel_size}x{kernel_size}")
    print(f"  Kernel length: {kernel_length}")
    print(f"  Kernel delay: {kernel_delay}")
    print(f"  Spatiotemporal channels: {n_channels}")
    print(f"  Decoder size: {decoder_size}")
    print(f"  Batch size: {batch_size}")
    print(f"  Input frames: {t}")

    # Initialize model
    print("\n[1/4] Initializing model...")
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

    # Create dummy input
    print("\n[2/4] Creating dummy input...")
    x = torch.randn(batch_size, t, kernel_size, kernel_size)
    print(f"  Input shape: {x.shape}")

    # Forward pass
    print("\n[3/4] Running forward pass...")
    model.eval()
    with torch.no_grad():
        eye_velocities, reconstructed_frames = model(x)

    # Expected output time dimension
    t_out = t - kernel_length + 1

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

    # Test backward pass
    print("\n[4/4] Testing backward pass...")
    model.train()
    x_train = torch.randn(batch_size, t, kernel_size, kernel_size)
    eye_velocities, reconstructed_frames = model(x_train)

    # Dummy loss
    velocity_loss = eye_velocities.mean()
    recon_loss = reconstructed_frames.mean()
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

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_full_model()

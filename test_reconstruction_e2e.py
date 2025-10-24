"""
End-to-end test for bilinear splatting reconstruction.

This test simulates the full reconstruction pipeline:
1. Start with a known test image
2. Extract ROI crops from known positions
3. Use bilinear_splat_training_batch to reconstruct the image
4. Compare reconstruction with original image
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from bilinear_splat import bilinear_splat_training_batch


def create_test_image(size=64):
    """
    Create a test image with recognizable patterns for visual verification.

    Returns a checkerboard pattern with gradients.
    """
    # Create coordinate grids
    y, x = torch.meshgrid(
        torch.linspace(0, 1, size),
        torch.linspace(0, 1, size),
        indexing='ij'
    )

    # Create a pattern combining multiple features
    # 1. Checkerboard pattern
    checkerboard = ((torch.floor(x * 4) + torch.floor(y * 4)) % 2) * 0.3

    # 2. Radial gradient from center
    center_x, center_y = 0.5, 0.5
    dist = torch.sqrt((x - center_x)**2 + (y - center_y)**2)
    radial = 0.3 * (1 - dist / dist.max())

    # 3. Horizontal gradient
    horizontal = 0.4 * x

    # Combine patterns
    image = checkerboard + radial + horizontal

    # Normalize to [0, 1]
    image = (image - image.min()) / (image.max() - image.min())

    return image


def extract_roi_crops(image, positions, roi_size=24):
    """
    Extract ROI crops from the image at given positions.

    Args:
        image: Tensor (H, W) - source image
        positions: Tensor (T, 2) - (x, y) positions for each crop
        roi_size: int - size of ROI crops

    Returns:
        crops: Tensor (T, roi_size, roi_size) - extracted crops
    """
    H, W = image.shape
    T = positions.shape[0]
    crops = torch.zeros(T, roi_size, roi_size)

    for t in range(T):
        x, y = positions[t]
        x_int, y_int = int(x), int(y)

        # Extract crop (with bounds checking)
        x_end = min(x_int + roi_size, W)
        y_end = min(y_int + roi_size, H)

        crop_h = y_end - y_int
        crop_w = x_end - x_int

        crops[t, :crop_h, :crop_w] = image[y_int:y_end, x_int:x_end]

    return crops


def generate_scan_path(canvas_size, roi_size, num_steps, pattern='spiral'):
    """
    Generate a scan path across the image.

    Args:
        canvas_size: tuple (H, W)
        roi_size: int
        num_steps: int - number of positions
        pattern: str - 'spiral', 'raster', or 'random'

    Returns:
        positions: Tensor (num_steps, 2) - (x, y) positions
    """
    H, W = canvas_size
    max_x = W - roi_size
    max_y = H - roi_size

    if pattern == 'spiral':
        # Spiral from center outward
        center_x, center_y = max_x / 2, max_y / 2
        positions = []

        for i in range(num_steps):
            angle = i * 2 * np.pi / 8  # 8 positions per revolution
            radius = (i / num_steps) * min(center_x, center_y)

            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)

            # Clamp to valid range
            x = np.clip(x, 0, max_x)
            y = np.clip(y, 0, max_y)

            positions.append([x, y])

        return torch.tensor(positions, dtype=torch.float32)

    elif pattern == 'raster':
        # Raster scan (left to right, top to bottom)
        positions = []
        step_x = max_x / (np.sqrt(num_steps) - 1) if num_steps > 1 else 0
        step_y = max_y / (np.sqrt(num_steps) - 1) if num_steps > 1 else 0

        for i in range(int(np.sqrt(num_steps))):
            for j in range(int(np.sqrt(num_steps))):
                x = j * step_x
                y = i * step_y
                positions.append([x, y])
                if len(positions) >= num_steps:
                    break
            if len(positions) >= num_steps:
                break

        return torch.tensor(positions[:num_steps], dtype=torch.float32)

    elif pattern == 'random':
        # Random positions (with seed for reproducibility)
        torch.manual_seed(42)
        positions = torch.rand(num_steps, 2)
        positions[:, 0] *= max_x
        positions[:, 1] *= max_y
        return positions

    else:
        raise ValueError(f"Unknown pattern: {pattern}")


def compute_reconstruction_metrics(original, reconstructed, mask=None):
    """
    Compute metrics comparing reconstruction to original.

    Args:
        original: Tensor (H, W)
        reconstructed: Tensor (H, W)
        mask: Optional Tensor (H, W) - valid region mask

    Returns:
        dict with metrics
    """
    if mask is not None:
        valid_pixels = mask.bool()
        orig_masked = original[valid_pixels]
        recon_masked = reconstructed[valid_pixels]
    else:
        orig_masked = original.flatten()
        recon_masked = reconstructed.flatten()

    # Mean Squared Error
    mse = ((orig_masked - recon_masked) ** 2).mean().item()

    # Peak Signal-to-Noise Ratio
    if mse > 0:
        psnr = 10 * np.log10(1.0 / mse)
    else:
        psnr = float('inf')

    # Mean Absolute Error
    mae = (orig_masked - recon_masked).abs().mean().item()

    # Correlation coefficient
    if len(orig_masked) > 1:
        corr = torch.corrcoef(torch.stack([orig_masked, recon_masked]))[0, 1].item()
    else:
        corr = 1.0

    return {
        'mse': mse,
        'psnr': psnr,
        'mae': mae,
        'correlation': corr
    }


def visualize_reconstruction(original, reconstructed, positions, roi_size,
                            mask=None, save_path='reconstruction_test.png'):
    """
    Create visualization of the reconstruction process.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Original image
    axes[0, 0].imshow(original.numpy(), cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # Original with scan path overlay
    axes[0, 1].imshow(original.numpy(), cmap='gray', vmin=0, vmax=1)
    # Draw scan path
    pos_np = positions.numpy()
    axes[0, 1].plot(pos_np[:, 0] + roi_size/2, pos_np[:, 1] + roi_size/2,
                    'r-', linewidth=1, alpha=0.5)
    axes[0, 1].scatter(pos_np[:, 0] + roi_size/2, pos_np[:, 1] + roi_size/2,
                      c=range(len(pos_np)), cmap='hot', s=20, alpha=0.7)
    # Draw ROI boxes for first few positions
    for i in range(min(5, len(pos_np))):
        x, y = pos_np[i]
        rect = plt.Rectangle((x, y), roi_size, roi_size,
                            fill=False, edgecolor='cyan', linewidth=1, alpha=0.5)
        axes[0, 1].add_patch(rect)
    axes[0, 1].set_title(f'Scan Path ({len(positions)} positions)')
    axes[0, 1].axis('off')

    # Reconstructed image
    axes[0, 2].imshow(reconstructed.numpy(), cmap='gray', vmin=0, vmax=1)
    axes[0, 2].set_title('Reconstructed Image')
    axes[0, 2].axis('off')

    # Absolute error
    error = torch.abs(original - reconstructed)
    if mask is not None:
        error = error * mask
    im = axes[1, 0].imshow(error.numpy(), cmap='hot', vmin=0, vmax=0.5)
    axes[1, 0].set_title('Absolute Error')
    axes[1, 0].axis('off')
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046)

    # Mask (if provided)
    if mask is not None:
        axes[1, 1].imshow(mask.numpy(), cmap='gray')
        axes[1, 1].set_title('Valid Region Mask')
        axes[1, 1].axis('off')
    else:
        axes[1, 1].text(0.5, 0.5, 'No Mask', ha='center', va='center')
        axes[1, 1].axis('off')

    # Histogram comparison
    axes[1, 2].hist(original.flatten().numpy(), bins=50, alpha=0.5,
                   label='Original', color='blue', density=True)
    axes[1, 2].hist(reconstructed.flatten().numpy(), bins=50, alpha=0.5,
                   label='Reconstructed', color='red', density=True)
    axes[1, 2].set_xlabel('Pixel Value')
    axes[1, 2].set_ylabel('Density')
    axes[1, 2].set_title('Pixel Value Distribution')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {save_path}")
    plt.close()


def test_e2e_reconstruction(pattern='spiral', num_positions=20, batch_size=4,
                           visualize=True):
    """
    End-to-end test of the reconstruction pipeline.

    Args:
        pattern: str - scan pattern ('spiral', 'raster', or 'random')
        num_positions: int - number of ROI positions
        batch_size: int - batch size for testing
        visualize: bool - whether to create visualization
    """
    print(f"\n{'='*70}")
    print(f"End-to-End Reconstruction Test: {pattern.upper()} pattern")
    print(f"{'='*70}")

    # Setup
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    canvas_size = (64, 64)
    roi_size = 24
    Hc, Wc = canvas_size

    print(f"Canvas size: {canvas_size}")
    print(f"ROI size: {roi_size}x{roi_size}")
    print(f"Number of positions: {num_positions}")
    print(f"Batch size: {batch_size}")

    # 1. Create test image
    print("\n[1/5] Creating test image...")
    original_image = create_test_image(size=Hc)
    print(f"  Image shape: {original_image.shape}")
    print(f"  Image range: [{original_image.min():.3f}, {original_image.max():.3f}]")

    # 2. Generate scan path
    print(f"\n[2/5] Generating {pattern} scan path...")
    positions = generate_scan_path(canvas_size, roi_size, num_positions, pattern=pattern)
    print(f"  Positions shape: {positions.shape}")
    print(f"  Position range: x=[{positions[:,0].min():.1f}, {positions[:,0].max():.1f}], "
          f"y=[{positions[:,1].min():.1f}, {positions[:,1].max():.1f}]")

    # 3. Extract ROI crops
    print("\n[3/5] Extracting ROI crops...")
    roi_crops = extract_roi_crops(original_image, positions, roi_size)
    print(f"  Crops shape: {roi_crops.shape}")
    print(f"  Crops range: [{roi_crops.min():.3f}, {roi_crops.max():.3f}]")

    # 4. Create batch data
    print(f"\n[4/5] Creating batched reconstruction data...")
    # Replicate to create batch (each batch item is the same for this test)
    I_batch = roi_crops.unsqueeze(0).repeat(batch_size, 1, 1, 1).to(device)  # (B, T, H, W)
    positions_batch = positions.unsqueeze(0).repeat(batch_size, 1, 1).to(device)  # (B, T, 2)

    print(f"  Batch images shape: {I_batch.shape}")
    print(f"  Batch positions shape: {positions_batch.shape}")

    # 5. Reconstruct using bilinear splatting
    print("\n[5/5] Reconstructing image using bilinear_splat_training_batch...")
    try:
        reconstructed_batch = bilinear_splat_training_batch(
            I_batch,
            positions_batch,
            canvas_size
        )  # (B, 1, Hc, Wc)

        print(f"  ✓ Reconstruction successful")
        print(f"  Output shape: {reconstructed_batch.shape}")

        # Extract first batch item for analysis
        reconstructed = reconstructed_batch[0, 0].cpu()

        print(f"  Reconstructed range: [{reconstructed.min():.3f}, {reconstructed.max():.3f}]")

        # Create mask of covered regions
        mask = torch.zeros(canvas_size)
        for pos in positions:
            x, y = int(pos[0]), int(pos[1])
            x_end = min(x + roi_size, Wc)
            y_end = min(y + roi_size, Hc)
            mask[y:y_end, x:x_end] = 1.0

        coverage = mask.sum() / (Hc * Wc)
        print(f"  Coverage: {coverage*100:.1f}% of canvas")

        # Compute metrics
        print("\n" + "="*70)
        print("RECONSTRUCTION METRICS")
        print("="*70)

        metrics = compute_reconstruction_metrics(original_image, reconstructed, mask)
        print(f"  MSE:         {metrics['mse']:.6f}")
        print(f"  PSNR:        {metrics['psnr']:.2f} dB")
        print(f"  MAE:         {metrics['mae']:.6f}")
        print(f"  Correlation: {metrics['correlation']:.4f}")

        # Quality assessment
        print("\n" + "="*70)
        print("QUALITY ASSESSMENT")
        print("="*70)

        if metrics['psnr'] > 30:
            quality = "EXCELLENT"
        elif metrics['psnr'] > 25:
            quality = "GOOD"
        elif metrics['psnr'] > 20:
            quality = "FAIR"
        else:
            quality = "POOR"

        print(f"  Quality: {quality}")
        print(f"  Status:  {'✅ PASS' if metrics['psnr'] > 20 else '❌ FAIL'}")

        # Visualize
        if visualize:
            print("\n" + "="*70)
            print("VISUALIZATION")
            print("="*70)
            visualize_reconstruction(
                original_image,
                reconstructed,
                positions,
                roi_size,
                mask,
                save_path=f'reconstruction_test_{pattern}.png'
            )

        print("\n" + "="*70)
        print("✅ END-TO-END TEST PASSED")
        print("="*70 + "\n")

        return True, metrics

    except Exception as e:
        print(f"\n❌ Reconstruction failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def run_all_tests():
    """Run tests with different scan patterns."""
    print("\n" + "="*70)
    print("RUNNING COMPREHENSIVE END-TO-END RECONSTRUCTION TESTS")
    print("="*70)

    patterns = ['spiral', 'raster', 'random']
    results = {}

    for pattern in patterns:
        success, metrics = test_e2e_reconstruction(
            pattern=pattern,
            num_positions=25,  # More positions for better coverage
            batch_size=4,
            visualize=True
        )
        results[pattern] = (success, metrics)

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    for pattern, (success, metrics) in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        if metrics:
            print(f"{pattern.upper():8s}: {status} | PSNR: {metrics['psnr']:6.2f} dB | "
                  f"Corr: {metrics['correlation']:.4f}")
        else:
            print(f"{pattern.upper():8s}: {status}")

    all_passed = all(success for success, _ in results.values())

    print("\n" + "="*70)
    if all_passed:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("="*70 + "\n")

    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)

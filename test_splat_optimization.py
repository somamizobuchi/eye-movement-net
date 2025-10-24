"""
Quick test to verify the optimized bilinear_splat_training_batch works correctly.
"""

import torch
from bilinear_splat import bilinear_splat_training_batch

def test_optimized_splat():
    """Test the optimized splatting function."""
    print("Testing optimized bilinear_splat_training_batch...")

    # Test parameters (matching training dimensions)
    B = 8  # batch size
    T = 9  # timesteps
    H, W = 24, 24  # ROI size
    Hc, Wc = 64, 64  # canvas size

    # Create test inputs
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    I = torch.randn(B, T, H, W, device=device, requires_grad=True)
    offsets = torch.randn(B, T, 2, device=device, requires_grad=True) * 10 + 20  # Random offsets around center
    canvas_size = (Hc, Wc)

    print(f"Input shape: {I.shape}")
    print(f"Offsets shape: {offsets.shape}")
    print(f"Canvas size: {canvas_size}")

    # Run the optimized function
    try:
        result = bilinear_splat_training_batch(I, offsets, canvas_size)
        print(f"✓ Function executed successfully")
        print(f"Output shape: {result.shape}")
        print(f"Expected shape: ({B}, 1, {Hc}, {Wc})")

        # Check output properties
        assert result.shape == (B, 1, Hc, Wc), f"Wrong shape: {result.shape}"
        print(f"✓ Shape is correct")

        assert not torch.isnan(result).any(), "Output contains NaN"
        print(f"✓ No NaN values")

        assert not torch.isinf(result).any(), "Output contains Inf"
        print(f"✓ No Inf values")

        print(f"Output min: {result.min().item():.4f}")
        print(f"Output max: {result.max().item():.4f}")
        print(f"Output mean: {result.mean().item():.4f}")

        # Check differentiability
        result.sum().backward()
        print(f"✓ Backward pass successful (differentiable)")

        print("\n✅ All tests passed!")
        return True

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_optimized_splat()
    exit(0 if success else 1)

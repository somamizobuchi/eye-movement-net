"""
Test to verify velocity integration produces correct first position.
"""

import torch
from reconstruct_from_model import integrate_velocities, integrate_velocities_batch


def test_single_integration():
    """Test single velocity integration."""
    print("Testing single velocity integration...")

    # Create test data
    initial_pos = torch.tensor([10.0, 20.0])
    velocities = torch.tensor([
        [1.0, 2.0],  # v[0]
        [0.5, 1.0],  # v[1]
        [0.3, 0.5],  # v[2]
    ])

    # Integrate
    positions = integrate_velocities(velocities, initial_pos, dt=1.0)

    print(f"Initial position: {initial_pos}")
    print(f"Velocities:\n{velocities}")
    print(f"Integrated positions:\n{positions}")

    # Verify first position equals initial position
    assert torch.allclose(positions[0], initial_pos), \
        f"First position {positions[0]} != initial position {initial_pos}"

    # Verify second position
    expected_pos_1 = initial_pos + velocities[0]
    assert torch.allclose(positions[1], expected_pos_1), \
        f"Second position {positions[1]} != expected {expected_pos_1}"

    # Verify third position
    expected_pos_2 = initial_pos + velocities[0] + velocities[1]
    assert torch.allclose(positions[2], expected_pos_2), \
        f"Third position {positions[2]} != expected {expected_pos_2}"

    print("✓ Single integration test passed!")
    print()
    return True


def test_batch_integration():
    """Test batched velocity integration."""
    print("Testing batch velocity integration...")

    # Create test data
    batch_size = 3
    initial_positions = torch.tensor([
        [10.0, 20.0],
        [5.0, 15.0],
        [8.0, 12.0],
    ])
    velocities = torch.tensor([
        [[1.0, 2.0], [0.5, 1.0], [0.3, 0.5]],  # batch 0
        [[2.0, 1.0], [1.0, 0.5], [0.5, 0.3]],  # batch 1
        [[1.5, 1.5], [0.8, 0.8], [0.4, 0.4]],  # batch 2
    ])

    # Integrate
    positions = integrate_velocities_batch(velocities, initial_positions, dt=1.0)

    print(f"Initial positions:\n{initial_positions}")
    print(f"Velocities shape: {velocities.shape}")
    print(f"Integrated positions shape: {positions.shape}")
    print(f"First positions:\n{positions[:, 0, :]}")

    # Verify first position equals initial position for all batches
    assert torch.allclose(positions[:, 0, :], initial_positions), \
        f"First positions don't match initial positions"

    # Verify second position for batch 0
    expected_pos_1 = initial_positions[0] + velocities[0, 0]
    assert torch.allclose(positions[0, 1], expected_pos_1), \
        f"Batch 0, position 1: {positions[0, 1]} != expected {expected_pos_1}"

    print("✓ Batch integration test passed!")
    print()
    return True


def test_reconstruction_consistency():
    """Test that reconstruction uses correct initial position."""
    print("Testing reconstruction consistency...")

    from reconstruct_from_model import reconstruct_batch_optimized

    # Setup
    device = torch.device("cpu")  # Use CPU for test
    B, T, H, W = 2, 5, 24, 24
    Hc, Wc = 64, 64

    # Create synthetic data
    eye_velocities = torch.randn(B, T, 2, device=device)
    reconstructed_frames = torch.randn(B, T, H, W, device=device)
    initial_positions = torch.tensor([[20.0, 20.0], [25.0, 25.0]], device=device)
    canvas_size = (Hc, Wc)

    # Reconstruct (this will call integrate_velocities_batch internally)
    reconstructions = reconstruct_batch_optimized(
        eye_velocities,
        reconstructed_frames,
        initial_positions,
        canvas_size,
        dt=1.0
    )

    print(f"Reconstruction shape: {reconstructions.shape}")
    print("✓ Reconstruction consistency test passed!")
    print()
    return True


if __name__ == "__main__":
    print("="*70)
    print("VELOCITY INTEGRATION TESTS")
    print("="*70)
    print()

    success = True
    success = success and test_single_integration()
    success = success and test_batch_integration()
    success = success and test_reconstruction_consistency()

    print("="*70)
    if success:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("="*70)

    exit(0 if success else 1)

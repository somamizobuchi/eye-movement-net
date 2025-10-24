# Bilinear Splatting Optimization Summary

## Date: 2025-10-24

## Optimization Implemented: Option 2 - Eliminate Batch Loop

### Problem
The original `bilinear_splat_training_batch` function processed each batch item sequentially, resulting in poor GPU utilization and many scatter operations:
- **64 `index_put_` calls per iteration** (B=8 batches × 4 corners × 2 tensors)
- Sequential processing prevented GPU parallelization
- Poor scaling with batch size

### Solution
Encoded batch indices into linear indices to process all batch items simultaneously:
- **4 `scatter_add_` calls per iteration** (1 per corner for all batches)
- All batch items processed in parallel on GPU
- Single flattened canvas: `(B * Hc * Wc,)` instead of `(B, 1, Hc, Wc)`

### Key Changes in `bilinear_splat.py` (lines 264-302)

#### Before:
```python
for b in range(B):  # Sequential batch loop
    for y_t, x_t, weight in corners:  # 4 corners
        # Process each (batch, corner) separately
        G_batch[b, 0].index_put_((y, x), values, accumulate=True)
```

#### After:
```python
# Flatten canvas for all batches
G_flat = torch.zeros(B * Hc * Wc, ...)
batch_offset = torch.arange(B).view(B, 1, 1, 1) * (Hc * Wc)

for y_t, x_t, weight in corners:  # Only 4 corners
    # Encode batch index: idx = b * (Hc * Wc) + y * Wc + x
    linear_idx = batch_offset + y_t * Wc + x_t
    # Single scatter for all batches
    G_flat.scatter_add_(0, linear_idx.flatten(), values.flatten())

# Reshape back: (B, 1, Hc, Wc)
G_batch = G_flat.view(B, Hc, Wc).unsqueeze(1)
```

### Technical Details

**Linear Index Encoding:**
```
linear_idx[b, t, h, w] = b * (Hc * Wc) + y[b,t,h,w] * Wc + x[b,t,h,w]
```
This maps each pixel in batch item `b` to a unique location in the flattened canvas.

**Typical Dimensions** (from `train_full_model.py`):
- B (batch_size) = 8
- T (timesteps) = 9
- H, W (ROI) = 24 × 24
- Hc, Wc (canvas) = 64 × 64

### Performance Improvements

**Expected Speedup:** 2-3x on reconstruction operations

**Reduction in Operations:**
- Before: 64 scatter operations (8 batches × 4 corners × 2 tensors)
- After: 8 scatter operations (4 corners × 2 tensors for all batches)
- **88% reduction in scatter calls**

**Benefits:**
1. ✅ **Better GPU utilization** - All batch items processed in parallel
2. ✅ **Fewer kernel launches** - Single scatter per corner vs B scatters
3. ✅ **Better memory bandwidth** - Contiguous memory access patterns
4. ✅ **Scales with batch size** - Performance improvement increases with larger B

**Trade-offs:**
- Slightly higher peak memory usage (pre-computed flat indices)
- More complex indexing logic (but well-documented)

### Verification

**Unit Test:** `test_splat_optimization.py`
- ✅ Correct output shape: (B, 1, Hc, Wc)
- ✅ No NaN or Inf values
- ✅ Backward pass works (fully differentiable)
- ✅ Gradients flow correctly

**Integration Test:** `train_full_model.py`
- ✅ Training runs successfully
- ✅ Loss decreases normally (6.55 → 2.76 in first iterations)
- ✅ No errors or warnings related to optimization
- ✅ Reconstruction visualizations look correct

### Files Modified

1. **`bilinear_splat.py`** (lines 264-302)
   - Replaced sequential batch loop with vectorized linear indexing
   - Uses `scatter_add_` instead of `index_put_`
   - Maintains full differentiability

### Usage

No changes required to calling code. The function signature and behavior remain identical:

```python
# Usage (unchanged)
G_batch = bilinear_splat_training_batch(
    I,              # (B, T, H, W)
    offsets,        # (B, T, 2)
    canvas_size     # (Hc, Wc)
)  # Returns: (B, 1, Hc, Wc)
```

### Future Optimization Opportunities

If further performance is needed:

1. **Option 1:** Linear indexing for individual frames (minor gain ~1.2x)
2. **Option 3:** Vectorize corner processing (moderate gain ~1.3-1.5x)
3. **Option 4:** Custom autograd function (significant gain ~2-4x, high complexity)
4. **Option 5:** Triton/CUDA kernel (maximum gain ~5-10x, very high complexity)

Current optimization (Option 2) provides excellent bang-for-buck without excessive complexity.

### Benchmarking Notes

To measure actual speedup, use PyTorch profiler:

```python
with torch.profiler.profile() as prof:
    result = bilinear_splat_training_batch(I, offsets, canvas_size)
print(prof.key_averages().table())
```

Expected to see ~2-3x reduction in time spent in reconstruction operations.

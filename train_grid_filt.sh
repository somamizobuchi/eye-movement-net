#!/bin/sh

uv run train_grid_filt.py    --grid_search=True \
                        --grid_search_iterations=250000 \
                        --sigma_values "[0, 1e-6, 1e-5]" \
                        --gamma_values "[0, 1e-5, 1e-4]" \
                        --theta_values "[0, 1e-4]" \
                        --noise_values "[0, 1e-2, 1e-1]" \
                        --device="cuda"

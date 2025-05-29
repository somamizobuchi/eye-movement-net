#!/bin/sh

uv run train_grid.py    --grid_search=True \
                        --grid_search_iterations=300000 \
                        --sigma_values "[0, 1e-5, 1e-4]" \
                        --gamma_values "[0, 1e-5, 1e-4]" \
                        --theta_values "[0, 1e-3, 1e-2]" \
                        --device="cuda"

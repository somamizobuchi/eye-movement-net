#!/bin/sh

uv run train_grid.py    --grid_search=True \
                        --grid_search_iterations=250000 \
                        --gamma_values "[1e-4, 1e-3, 1e-2]" \
                        --sigma_values "[1e-4, 1e-3, 1e-2]" \
                        --theta_values "[1e-2, 1e-1, 1e0]" \
                        --device="cuda"

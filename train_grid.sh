#!/bin/sh

uv run train_grid.py --grid_search=True --grid_search_iterations=100000 --gamma_values "[1e-3]" --sigma_values "[1e-4, 1e-3, 1e-2]" --theta_values "[1e-2, 1e-1, 1]" --device="cpu"

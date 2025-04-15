#!/bin/sh

uv run train_grid.py --grid_search=True --grid_search_iterations=500000 --alpha_values "[0.01]" --delta_values "[0.1, 1.0]" --gamma_values "[0.01]" --beta_values "[0.01]" --device="cpu"

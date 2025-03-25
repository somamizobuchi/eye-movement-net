#!/bin/sh

uv run train_grid.py --grid_search=True --grid_search_iterations=250000 --alpha_values "[0, 0.01, 0.1, 1]" --delta_values "[0, 0.01, 0.1, 1]" --gamma_values "[0, 0.01, 0.1, 1]" --beta_values "[0, 0.01, 0.1, 1]" --device="cuda"

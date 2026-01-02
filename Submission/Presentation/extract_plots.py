#!/usr/bin/env python3
"""
Script to extract plots and results from Team1-AC.ipynb notebook
and save them to the Presentation folder for use in beamer.
"""

import json
import base64
import os
from pathlib import Path

# Load the notebook
notebook_path = "../Team1-AC.ipynb"
with open(notebook_path, 'r') as f:
    nb = json.load(f)

# Create output directory for plots
plots_dir = Path("plots")
plots_dir.mkdir(exist_ok=True)

# Extract plots from cells
plot_count = 0
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and 'outputs' in cell:
        for output in cell['outputs']:
            # Check for matplotlib/image output
            if 'data' in output and 'image/png' in output['data']:
                plot_count += 1
                # Decode base64 image
                img_data = output['data']['image/png']
                img_bytes = base64.b64decode(img_data)
                
                # Save to file
                img_path = plots_dir / f"plot_{plot_count:02d}.png"
                with open(img_path, 'wb') as f:
                    f.write(img_bytes)
                
                print(f"Extracted plot {plot_count}: {img_path}")

print(f"\nTotal plots extracted: {plot_count}")

# Extract text results (looking for specific patterns)
print("\n--- Searching for numerical results ---")
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and 'outputs' in cell:
        for output in cell['outputs']:
            if 'text' in output:
                text = ''.join(output['text']) if isinstance(output['text'], list) else output['text']
                # Look for key results
                if any(keyword in text.lower() for keyword in ['accuracy', 'mse', 'optimal', 'best', 'μ', 'sigma', 'bandwidth']):
                    print(f"\n--- Cell {i} ---")
                    print(text[:500])  # First 500 chars

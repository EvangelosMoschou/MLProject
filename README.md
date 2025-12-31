# ML Project: The Epsilon Protocol (Part D)

Part D implements the state-of-the-art "Epsilon Protocol" for tabular classification, aiming for the theoretical limit of accuracy.

## Architecture: The Quantum Ensemble

The solution (`PartD/solution_quantum.py`) integrates five advanced paradigms:

1.  **True TabR (PyTorch)**: A retrieval-augmented neural network that uses Multi-Head Attention to attend to similar training samples.
2.  **Generative Classifier (DAE)**: A non-discriminative approach that models the density $P(x|y)$ of each class using Denoising Autoencoders. Inference is based on Energy Minimization.
3.  **TabM (Mamba)**: Adapts the Mamba State Space Model for tabular data.
4.  **KAN (Kolmogorov-Arnold Networks)**: Replaces fixed activation functions with learnable splines.
5.  **HyperTabPFN**: A Transformer pre-trained on millions of datasets to perform in-context learning.

## Optimization: SAM
We employ **SAM (Sharpness-Aware Minimization)** to flatten the loss landscape, improving generalization on unseen test data.

## Usage

To run the full Epsilon Protocol:

```bash
python3 PartD/solution_quantum.py
```

## Directory Structure
- `PartD/src/`: Core components
    - `tabr.py`: True TabR implementation.
    - `generative.py`: Generative DAE Classifier.
    - `sam.py`: SAM Optimizer.
    - `dae_model.py`: Legacy DAE (kept for reference).
- `PartD/solution_quantum.py`: Main execution script.
# Walkthrough: The Epsilon Protocol

The "Epsilon Protocol" represents the final evolutionary stage of our classification system (Part D). It moves beyond the approximations of "Theta" and "Omega" to implement the theoretically optimal architectures.

## Core Advancements

### 1. True TabR (Attention-Based Retrieval)
We replaced the CatBoost proxy with a fully differentiable **PyTorch TabR** architecture in `PartD/src/tabr.py`.
- **Mechanism**: For every sample $x$, we retrieve $K$ neighbors. A Multi-Head Cross-Attention mechanism attends to these neighbors (Query=$x$, Key/Value=$Neighbors$) to augment the feature space before the final Gated MLP prediction.
- **Why**: This allows the model to "look up" similar past cases and learn from their labels/features dynamically.

### 2. Generative Classification (Bayesian DAE)
We implemented a **Generative Classifier** in `PartD/src/generative.py` based on class-specific Denoising Autoencoders.
- **Training**: We train $C$ separate DAEs, one for each class density $P(x|y=c)$.
- **Inference**: To classify $x$, we measure the "Energy" (Reconstruction Error) of $x$ under each DAE. The class that reconstructs $x$ best (lowest energy) implies the highest likelihood $P(x|y)$.
- **Inference Trick**: For uncertain ("Silver") samples, we perform gradient descent on $x$ to minimize the energy of each class model.

### 3. SAM Optimization
We integrated **SAM (Sharpness-Aware Minimization)** into the training loops via `PartD/src/sam.py`. SAM minimizes loss value *and* loss sharpness, leading to better generalization.

## The Ensemble (The Quantum State)
The final prediction in `PartD/solution_quantum.py` is a consensus of:
1.  **TabM (Mamba)**: Sequence modeling on tabular features.
2.  **KAN (Kolmogorov-Arnold)**: Learnable activation functions.
3.  **True TabR**: Attention-based retrieval.
4.  **Generative DAE**: Energy-based classification.
5.  **HyperTabPFN**: Transformer-based prior-data fitted network.

## Execution
Run the Epsilon Protocol:
```bash
python3 PartD/solution_quantum.py
```

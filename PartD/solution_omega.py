"""
Omega-Singularity Protocol Entry Point.

This script triggers the final build execution residing in `PartD/sigma_omega`.
The architecture includes:
- Trinity Feature Engine (RankGauss, Stability, Manifold, DAE)
- Tabular Diffusion Augmentation
- Model Zoo: ThetaTabM (BatchEnsemble), TrueTabR, KAN, XGB-DART, Cat-Langevin
- Inference: Adversarial Reweighting, TTT (Entropy+Consistency), LID Scaling, NNLS Stacking.
"""

from sigma_omega.main import main

if __name__ == '__main__':
    main()

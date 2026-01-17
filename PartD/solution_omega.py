"""
Omega-Singularity Protocol Entry Point.

This script triggers the final build execution residing in `PartD/sigma_omega`.
The architecture includes:
- Trinity Feature Engine (RankGauss, Stability, Manifold, DAE)
- Tabular Diffusion Augmentation
- Model Zoo: ThetaTabM (BatchEnsemble), TrueTabR, KAN, XGB-DART, Cat-Langevin
- Inference: Adversarial Reweighting, TTT (Entropy+Consistency), LID Scaling, NNLS Stacking.
"""

import sys
import os

print(f"DEBUG: Starting script {sys.argv[0]} from {os.getcwd()}", flush=True)

try:
    print("DEBUG: Importing sigma_omega.main...", flush=True)
    from sigma_omega.main import main
    print("DEBUG: Import successful.", flush=True)
except Exception as e:
    import traceback
    print(f"CRASH during import: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)

if __name__ == '__main__':
    print("DEBUG: Entering __name__ == __main__ block", flush=True)
    try:
        main()
        print("DEBUG: main() returned successfully", flush=True)
    except Exception as e:
        import traceback
        print(f"CRASH during main(): {e}", flush=True)
        traceback.print_exc()

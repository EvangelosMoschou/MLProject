"""Thin wrapper entrypoint.

The full implementation was refactored into `PartD/sigma_omega/`.
The original monolithic file was moved to `PartD/legacy/solution_quantum_monolith.py`.
"""

from sigma_omega.main import main


if __name__ == '__main__':
    main()

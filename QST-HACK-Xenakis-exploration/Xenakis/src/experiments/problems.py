# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 09:18:09 2026

@author: Elian PC
"""

import numpy as np
from pennylane import qchem

def build_problem(molecule: str = "h2"):

    molecule = molecule.lower()

    if molecule == "h2o":
        # Water example (active space)
        symbols = ["H", "O", "H"]
        coordinates = np.array([-0.0399, -0.0038, 0.0,
                                1.5780,  0.8540, 0.0,
                                2.7909, -0.5159, 0.0])
        H, n_qubits = qchem.molecular_hamiltonian(
            symbols, coordinates, active_electrons=4, active_orbitals=4
        )
        hf_state = qchem.hf_state(4, 8)
        return H, n_qubits, hf_state

    # Default: H2
    symbols = ["H", "H"]
    coordinates = np.array([0.0, 0.0, -0.6614,
                            0.0, 0.0,  0.6614])

    H, n_qubits = qchem.molecular_hamiltonian(symbols, coordinates)
    hf_state = qchem.hf_state(2, 4)
    return H, n_qubits, hf_state

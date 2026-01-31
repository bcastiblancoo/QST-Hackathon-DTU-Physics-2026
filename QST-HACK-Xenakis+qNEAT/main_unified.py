import numpy as np
from pennylane import qchem
import pennylane as qml

# Import Adapters
from adapters.xenakis_adapter import XenakisOptimizer
from adapters.qneat_adapter import QNeatOptimizer


# --- CONFIGURATION ---
METHOD = "QNEAT"       # Choose "QNEAT" or "XENAKIS"
POPULATION = 50        # CHANGED FROM 10 TO 50
GENERATIONS = 9       # Increase this to give it time to improve
N_GATES = 6

# --- PHYSICS SETUP ---
print("Setting up H2 Molecule...")
symbols = ["H", "H"]
coordinates = np.array([0.0, 0.0 , -0.6614, 0.0, 0.0, 0.6614])
H, n_qubits = qchem.molecular_hamiltonian(symbols, coordinates)
initial_state = qchem.hf_state(2, 4)

# --- FACTORY ---
if METHOD == "XENAKIS":
    optimizer = XenakisOptimizer(n_qubits, N_GATES, H, initial_state)
elif METHOD == "QNEAT":
    optimizer = QNeatOptimizer(n_qubits, N_GATES, H, initial_state)

# --- RUN ---
print(f"Running with {METHOD}...")

if METHOD == "QNEAT":
    optimizer = QNeatOptimizer(n_qubits, N_GATES, H, initial_state)
    final_energy = optimizer.run(POPULATION, GENERATIONS)
    
    print("\n" + "="*30)
    print(f"RESULTS FOR {METHOD}")
    print("="*30)
    print(f"(Target for H2 is approx: -1.137 Ha)")
    print("-" * 30)

elif METHOD == "XENAKIS":
    optimizer = XenakisOptimizer(n_qubits, N_GATES, H, initial_state)
    final_genome_str = optimizer.run(POPULATION, GENERATIONS)
    print("\n" + "="*30)
    print(f"RESULTS FOR {METHOD}")
    print("="*30)
    print(f"Final Genome String: {final_genome_str}")

print("\nDone.")
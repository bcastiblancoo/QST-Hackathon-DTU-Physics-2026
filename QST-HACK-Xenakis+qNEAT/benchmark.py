import matplotlib.pyplot as plt
import numpy as np
from pennylane import qchem

# Import Adapters
from adapters.xenakis_adapter import XenakisOptimizer
from adapters.qneat_adapter import QNeatOptimizer

# --- SETTINGS ---
POPULATION = 20
GENERATIONS = 10
N_GATES = 6 

# --- PHYSICS ---
print("Initializing Molecule...")
symbols = ["H", "H"]
coordinates = np.array([0.0, 0.0 , -0.6614, 0.0, 0.0, 0.6614])
H, n_qubits = qchem.molecular_hamiltonian(symbols, coordinates)
initial_state = qchem.hf_state(2, 4)

# --- RUN XENAKIS ---
print("\n" + "="*40)
print("Running XENAKIS (Standard GA)...")
print("="*40)
opt_xenakis = XenakisOptimizer(n_qubits, N_GATES, H, initial_state)
# Note: Adapters now return (result, history)
_, xenakis_history = opt_xenakis.run(POPULATION, GENERATIONS)

# --- RUN QNEAT ---
print("\n" + "="*40)
print("Running QNEAT (Neuroevolution)...")
print("="*40)
opt_qneat = QNeatOptimizer(n_qubits, N_GATES, H, initial_state)
# QNEAT adapter returns (circuit, energy, history)
_, _, qneat_history = opt_qneat.run(POPULATION, GENERATIONS)

# --- PLOT RESULTS ---
print("\nPlotting results...")
plt.figure(figsize=(10, 6))
plt.plot(xenakis_history, label='Xenakis (GA)', marker='o', linestyle='--')
plt.plot(qneat_history, label='QNEAT', marker='s', linestyle='-')

# Add target line for H2
plt.axhline(y=-1.113, color='r', linestyle=':', label='Target H2 Energy (-1.113)')

plt.title('Convergence Comparison: Xenakis vs QNEAT')
plt.xlabel('Generation')
plt.ylabel('Ground State Energy (Hartree)')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig("becnhmark.png", dpi=300)
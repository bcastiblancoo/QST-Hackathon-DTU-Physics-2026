from abc import ABC, abstractmethod

class QuantumOptimizer(ABC):
    def __init__(self, n_qubits, n_gates, hamiltonian, initial_state):
        self.n_qubits = n_qubits
        self.n_gates = n_gates
        self.hamiltonian = hamiltonian
        self.initial_state = initial_state

    @abstractmethod
    def run(self, population_size, generations):
        pass
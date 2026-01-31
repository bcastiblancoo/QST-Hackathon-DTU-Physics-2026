from optimizer import QuantumOptimizer
import genetic_algorithm_runtime as ga_script

class XenakisOptimizer(QuantumOptimizer):
    def run(self, population_size, generations):
        str_len = self.n_gates * (6 + 5)
        # Capture the history return value
        final_genome, history = ga_script.ga(
            population_size, generations, 1000, str_len,
            self.n_qubits, self.n_gates, self.hamiltonian, self.initial_state
        )
        return final_genome, history
from optimizer import QuantumOptimizer
import qNEAT.qNEAT as qNEAT
import qNEAT.gate as g 
import pennylane as qml
from vqe import ground_state_energy_VQE
import numpy as np

class QNeatOptimizer(QuantumOptimizer):
    def run(self, population_size, generations):
        self.qneat = qNEAT.QNEAT(population_size, self.n_qubits)
        history = [] 

        # Boost mutation
        for genome in self.qneat.population:
            genome.prob_add_gate_mutation = 0.5 
            genome.prob_weight_mutation = 0.8

        for gen in range(generations):
            # Fitness
            for genome in self.qneat.population:
                self.custom_fitness(genome)
            
            # Record Best
            best = sorted(self.qneat.population, key=lambda g: g._fitness if g._fitness else -1000, reverse=True)[0]
            current_energy = -best._fitness
            history.append(current_energy) 
            print(f"Gen {gen}: {current_energy:.6f} Ha")

            # Evolution
            self.qneat.run_generation("local_sim")
            
            # Reset settings
            for genome in self.qneat.population:
                genome.get_fitness = self.custom_fitness
                genome.prob_add_gate_mutation = 0.5 

        # Final return: Circuit, Energy, History
        best = sorted(self.qneat.population, key=lambda g: g._fitness if g._fitness else -1000, reverse=True)[0]
        qiskit_circuit, _ = best.get_circuit(self.n_qubits)
        opt_params = self.extract_params(best)
        if opt_params:
            bound_circuit = qiskit_circuit.assign_parameters(opt_params)
        else:
            bound_circuit = qiskit_circuit
        
        return qml.from_qiskit(bound_circuit), -best._fitness, history

    def extract_params(self, genome):
        param_values = []
        for layer_ind in sorted(genome.layers.keys()):
            layer = genome.layers[layer_ind]
            for gatetype in g.GateType:
                if gatetype.name in layer.gates:
                    for gate in layer.gates[gatetype.name]:
                        param_values.extend(gate.parameters)
        return param_values

    def update_genome_params(self, genome, optimized_params):
        if isinstance(optimized_params, np.ndarray): optimized_params = optimized_params.tolist()
        if not isinstance(optimized_params, list): return
        counter = 0
        for layer_ind in sorted(genome.layers.keys()):
            layer = genome.layers[layer_ind]
            for gatetype in g.GateType:
                if gatetype.name in layer.gates:
                    for gate in layer.gates[gatetype.name]:
                        n_gate_params = len(gate.parameters)
                        if n_gate_params > 0:
                            gate.parameters = np.array(optimized_params[counter : counter + n_gate_params])
                            counter += n_gate_params

    def custom_fitness(self, genome, n_qubits=None, backend=None):
        if not genome._update_fitness and genome._fitness is not None:
            return genome._fitness

        qiskit_circuit, _ = genome.get_circuit(self.n_qubits)
        start_params = self.extract_params(genome)
        
        def pennylane_circuit(params):
            if len(params) == 0: qml.from_qiskit(qiskit_circuit)()
            else: qml.from_qiskit(qiskit_circuit)(*params)
        
        try:
            energy, optimized_theta = ground_state_energy_VQE(
                self.hamiltonian, self.initial_state, 
                pennylane_circuit, len(start_params),
                initial_params=start_params
            )
            self.update_genome_params(genome, optimized_theta)
            genome._fitness = -energy
        except Exception:
            genome._fitness = -1000.0

        genome._update_fitness = False
        return genome._fitness
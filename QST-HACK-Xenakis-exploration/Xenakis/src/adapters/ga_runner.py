# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 09:26:27 2026

@author: Elian PC
"""

import numpy as np
from Xenakis.genetic_algorithm import ga
from Xenakis.circuit_generation import genome_to_circuit
from Xenakis.vqe import ground_state_energy_VQE

def run_ga(
    H,
    n_qubits: int,
    hf_state,
    population: int = 10,
    generations: int = 5,
    n_gates: int = 6,
    threshold: float = 1e9,
):

    str_len = n_gates * (6 + 5)  # matches Xenakis gene length convention

    def fitness_fn(agents):
        for agent in agents:
            circuit, n_params, complexity = genome_to_circuit(
                agent.string, n_qubits=n_qubits, n_gates=n_gates, initial_state=hf_state
            )

            if isinstance(n_params, int):
                n_params = np.zeros(1) if n_params == 0 else np.array([n_params])
            elif isinstance(n_params, list):
                n_params = np.array(n_params)

            energy = ground_state_energy_VQE(H, hf_state, circuit, n_params, total_complexity=complexity)
            agent.energy = energy
            agent.fitness = -energy - 1e-4 * complexity
        return agents

    best_genome = ga(population, generations, threshold, str_len, n_qubits, n_gates, H, hf_state)

    # Reconstruct final circuit and stats for best genome
    circuit, n_params, complexity = genome_to_circuit(
        best_genome, n_qubits=n_qubits, n_gates=n_gates, initial_state=hf_state
    )

    if isinstance(n_params, int):
        n_params = np.zeros(1) if n_params == 0 else np.array([n_params])
    elif isinstance(n_params, list):
        n_params = np.array(n_params)

    energy = ground_state_energy_VQE(H, hf_state, circuit, n_params, total_complexity=complexity)
    fitness = -energy - 1e-4 * complexity

    print("\n[GA] Best genome:", best_genome)
    print("Fitness:", fitness, "Energy:", energy)

    return best_genome, energy, fitness

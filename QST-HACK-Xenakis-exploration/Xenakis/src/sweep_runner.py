# -*- coding: utf-8 -*-

import itertools
import pandas as pd
from experiments.problems import build_problem
from adapters.ga_runner import run_ga

# Define parameter ranges
pop_sizes = [10, 20, 30]
gens_list = [5, 10, 15]
n_gates_list = [4, 6, 8]

# Load the molecule problem
H, n_qubits, hf_state = build_problem("h2")

# Store results for analysis
results = []

# Run experiments
for pop, gens, n_gates in itertools.product(pop_sizes, gens_list, n_gates_list):
    print(f"Running: pop={pop}, gens={gens}, n_gates={n_gates}")
    
    best_genome, energy, fitness = run_ga(H, n_qubits, hf_state, population=pop, generations=gens, n_gates=n_gates)

    results.append({
        "population": pop,
        "generations": gens,
        "n_gates": n_gates,
        "energy": energy,
        "fitness": fitness,
    })

# Save results to file
df = pd.DataFrame(results)
df.to_csv("sweep_results.csv", index=False)
print("\nFinished parameter sweep.\nResults saved to sweep_results.csv")


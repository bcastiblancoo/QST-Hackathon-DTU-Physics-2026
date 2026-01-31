import pennylane as qml
from pennylane import numpy as np

def ground_state_energy_VQE(H, initial_state, circuit, n_params, initial_params=None):

    n_qubits = len(initial_state)
    dev = qml.device("default.qubit", wires=n_qubits)
    
    @qml.qnode(dev)
    def cost_fn(param):
        qml.BasisState(initial_state, wires=range(n_qubits))
        circuit(param)
        return qml.expval(H)

    # Optimization Setup
    opt = qml.AdamOptimizer(stepsize=0.1)
    
    # --- FIX 1: USE MEMORY IF AVAILABLE ---
    if initial_params is not None and len(initial_params) == n_params:
        # Warm start: Start where the parent left off
        theta = np.array(initial_params, requires_grad=True)
    else:
        # Fallback: Random start
        theta = np.random.normal(0, 0.1, n_params, requires_grad=True)
        # Partial memory transfer if mutation changed size
        if initial_params is not None:
            min_len = min(len(initial_params), n_params)
            if min_len > 0:
                theta[:min_len] = initial_params[:min_len]
    
    if n_params == 0:
        return cost_fn([]), []

    max_iterations = 80
    conv_tol = 1e-06
    energy = 1000.0
    
    for n in range(max_iterations):
        prev_energy = energy
        theta, energy = opt.step_and_cost(cost_fn, theta)
        if np.abs(energy - prev_energy) <= conv_tol:
            break
            
    # RETURN TUPLE: (Energy, Optimized Angles)
    return energy, theta
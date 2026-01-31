import random
import numpy as np
import pennylane as qml
from vqe import ground_state_energy_VQE

class Agent:
    def __init__(self, length):
        self.string = ''.join(str(random.randint(0,1)) for _ in range(length))
        self.fitness = -1
        self.energy = 0

def string_to_gate(string, n_qubits):
    n_params = 0
    complexity = 0 
    gate_string = string[:6]
    qubit_string = string[6:]
    qubit_seed = int(qubit_string, 2) % (2**32 - 1)
    qubits = np.random.RandomState(seed=qubit_seed).permutation(n_qubits)
    
    if gate_string == '000000':
        def gate(theta):
            qml.RX(theta, wires=qubits[0])
        n_params = 1
        complexity = 2
    elif gate_string == '000001':
        def gate(theta):
            qml.RY(theta, wires=qubits[0])
        n_params = 1
        complexity = 2
    elif gate_string == '000010':
        def gate(theta):
            qml.RZ(theta, wires=qubits[0])
        n_params = 1
        complexity = 2
    elif gate_string == '000100':
        def gate():
            qml.CY(wires=qubits[:2])
        complexity = 2
    elif gate_string == '000101':
        def gate():
            qml.CZ(wires=qubits[:2])
        complexity = 2
    elif gate_string == '000110':
        def gate(theta):
            qml.CRX(theta, wires=qubits[:2])
        n_params = 1
        complexity = 3
    elif gate_string == '000111':
        def gate(theta):
            qml.CRY(theta, wires=qubits[:2])
        n_params = 1
        complexity = 3
    elif gate_string == '001000':
        def gate(theta):
            qml.CRZ(theta, wires=qubits[:2])
        n_params = 1
        complexity = 3
    elif gate_string == '001001':
        def gate():
            qml.PauliX(wires=qubits[0])
    elif gate_string == '001010':
        def gate():
            qml.PauliY(wires=qubits[0])
    elif gate_string == '001011':
        def gate():
            qml.PauliZ(wires=qubits[0])
    elif gate_string == '001100':
        def gate(theta):
            qml.PhaseShift(theta, wires=qubits[0])
        n_params = 1
        complexity = 1
    elif gate_string == '001101':
        def gate():
            qml.QubitCarry(wires=qubits[:4])
        complexity = 3
    elif gate_string == '001111':
        def gate():
            qml.QubitSum(wires=qubits[:3])
        complexity = 3
    elif gate_string == '010000':
        def gate(theta, phi, omega):
            qml.Rot(theta, phi, omega, wires=qubits[0])
        n_params = 3
        complexity = 2
    elif gate_string == '010001':
        def gate():
            qml.S(wires=qubits[0])
    elif gate_string == '010010':
        def gate():
            qml.SQISW(wires=qubits[:2])
        complexity = 2
    elif gate_string == '010011':
        def gate():
            qml.SWAP(wires=qubits[:2])
        complexity = 1
    elif gate_string == '010100':
        def gate():
            qml.SX(wires=qubits[0])
        complexity = 1
    elif gate_string == '010101':
        def gate(theta):
            qml.SingleExcitation(theta, wires=qubits[:2])
        n_params = 1
        complexity = 3
    elif gate_string == '010110':
        def gate(theta):
            qml.SingleExcitationPlus(theta, wires=qubits[:2])
        n_params = 1
        complexity = 3
    elif gate_string == '010111':
        def gate(theta):
            qml.SingleExcitationMinus(theta, wires=qubits[:2])
        n_params = 1
        complexity = 3
    elif gate_string == '011000':
        def gate():
            qml.Toffoli(wires=qubits[:3])
        complexity = 2
    elif gate_string == '011001':
        def gate(theta):
            qml.U1(theta, wires=qubits[0])
        n_params = 1
        complexity = 1
    elif gate_string == '011010':
        def gate(theta, phi):
            qml.U2(theta, phi, wires=qubits[0])
        n_params = 2
        complexity = 1
    elif gate_string == '011011':
        def gate(theta, phi, delta):
            qml.U3(theta, phi, delta, wires=qubits[0])
        n_params = 3
        complexity = 2
    elif gate_string == '011101':
        def gate(theta):
            qml.DoubleExcitation(theta, wires=qubits[:4])
        n_params = 1
        complexity = 4
    elif gate_string == '011110':
        def gate(theta):
            qml.DoubleExcitationPlus(theta, wires=qubits[:4])
        n_params = 1
        complexity = 4
    elif gate_string == '011111':
        def gate(theta):
            qml.DoubleExcitationMinus(theta, wires=qubits[:4])
        n_params = 1
        complexity = 4
    else:
        gate = None
    return gate, n_params, complexity

def genome_to_circuit(genome, n_qubits, n_gates):
    gates = []
    n_params = []
    total_complexity = 0
    gene_length = len(genome) // n_gates
    for i in range(n_gates):
        gene = genome[i * gene_length : (i+1) * gene_length]
        gate, n, complexity = string_to_gate(gene, n_qubits)
        if gate is not None:
            gates.append(gate)
            n_params.append(n)
            total_complexity += complexity

    def circuit(params):
        param_counter = 0
        for gate, n in zip(gates, n_params):
            if n == 0: gate()
            elif n == 1: 
                gate(params[param_counter])
            param_counter += n

    return circuit, sum(n_params), total_complexity

def ga(population, generations, threshold, str_len, n_qubits, n_gates, H, initial_state):
    agents = [Agent(str_len) for _ in range(population)]
    history = [] 

    for generation in range(generations):
        # Fitness
        for agent in agents:
            circuit, n_params, complexity = genome_to_circuit(agent.string, n_qubits, n_gates)
            try:
                energy, _ = ground_state_energy_VQE(H, initial_state, circuit, n_params)
                agent.fitness = -energy - (complexity * 0.001)
                agent.energy = energy
            except:
                agent.fitness = -1000
                agent.energy = 0
                
        agents = sorted(agents, key=lambda x: x.fitness, reverse=True)
        best_energy = agents[0].energy
        history.append(best_energy)
        print(f"Gen {generation}: {best_energy:.6f} Ha")
        
        # Selection
        survivors = agents[:max(1, int(0.2 * len(agents)))]
        offspring = []
        while len(offspring) + len(survivors) < population:
            p1, p2 = random.choice(survivors), random.choice(survivors)
            child = Agent(str_len)
            split = random.randint(0, str_len)
            child.string = p1.string[:split] + p2.string[split:]
            for i in range(len(child.string)):
                if random.random() < 0.05:
                    child.string = child.string[:i] + ('1' if child.string[i]=='0' else '0') + child.string[i+1:]
            offspring.append(child)
            
        agents = survivors + offspring

    return agents[0].string, history 

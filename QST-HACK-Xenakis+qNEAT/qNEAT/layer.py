class Layer:
    def __init__(self, ind):
        self.gates = {} 
        self.ind = ind

    def add_gate(self, gate):
        if gate.gatetype.name not in self.gates:
            self.gates[gate.gatetype.name] = []
        
        # Prevent duplicate qubit usage in same layer
        for g in self.gates[gate.gatetype.name]:
            if g.qubit == gate.qubit: return False
        
        self.gates[gate.gatetype.name].append(gate)
        return True

    def add_to_circuit(self, circuit, n_params):
        for key in self.gates:
            for gate in self.gates[key]:
                circuit, n_params = gate.add_to_circuit(circuit, n_params)
        circuit.barrier()
        return circuit, n_params

    def get_gates_generator(self):
        for key in self.gates:
            for gate in self.gates[key]:
                yield gate

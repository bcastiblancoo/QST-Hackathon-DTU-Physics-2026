import copy
from . import helper as h
from . import gate as g
from . import layer as l
from qiskit import QuantumCircuit, QuantumRegister
import numpy as np

class Genome(object):
    def __init__(self, global_layer_number):
        self.layers = {}
        self.global_layer_number = global_layer_number
        self._fitness = None
        self._update_fitness = True

        # --- FIX: INITIALIZE MUTATION PARAMETERS ---
        self.prob_add_gate_mutation = 0.1
        self.max_add_gate_tries = 10
        self.prob_weight_mutation = 0.8
        self.prob_weight_perturbation = 0.9
        self.perturbation_amplitude = 0.5

    @classmethod
    def from_layers(cls, global_layer_numer, layers):
        genome = Genome(global_layer_numer)
        genome.layers = layers
        return genome

    def add_gate(self, gate):
        ind = np.random.randint(self.global_layer_number.current() + 1)
        if ind == self.global_layer_number.current():
            self.global_layer_number.next()
        if ind not in self.layers:
            self.layers[ind] = l.Layer(ind)
        
        added = self.layers[ind].add_gate(gate)
        if added: self._update_fitness = True
        return added
    
    def mutate(self, innov_gen, n_qubits):
        # 1. Add Gate Mutation
        if np.random.random() < self.prob_add_gate_mutation:
            for _ in range(self.max_add_gate_tries):
                gate_type = np.random.choice(list(g.GateType))
                new_gate = g.GateGene(innov_gen.next(), gate_type, np.random.randint(n_qubits))
                if self.add_gate(new_gate): 
                    break
        
        # 2. Parameter Mutation (Weight perturbation)
        if np.random.random() < self.prob_weight_mutation:
            for layer in self.layers.values():
                for gate in layer.get_gates_generator():
                    perturbation = np.random.uniform(-1, 1, size=len(gate.parameters)) * self.perturbation_amplitude
                    gate.parameters += perturbation

    def get_circuit(self, n_qubits, n_parameters=0):
        circuit = QuantumCircuit(QuantumRegister(n_qubits))
        for layer_ind in sorted(self.layers.keys()):
            circuit, n_parameters = self.layers[layer_ind].add_to_circuit(circuit, n_parameters)
        return circuit, n_parameters

    @staticmethod
    def compatibility_distance(genome1, genome2):
        # Simplified distance based on disjoint genes
        g1_gates = set(gate.innovation_number for layer in genome1.layers.values() for gate in layer.get_gates_generator())
        g2_gates = set(gate.innovation_number for layer in genome2.layers.values() for gate in layer.get_gates_generator())
        disjoint = len(g1_gates.symmetric_difference(g2_gates))
        return disjoint / max(1, len(g1_gates | g2_gates))

    @classmethod
    def crossover(cls, g1, g2, n_qubits, backend):
        # Simple crossover: take union of layers
        new_layers = copy.deepcopy(g1.layers)
        for idx, layer in g2.layers.items():
            if idx not in new_layers:
                new_layers[idx] = copy.deepcopy(layer)
        return cls.from_layers(g1.global_layer_number, new_layers)
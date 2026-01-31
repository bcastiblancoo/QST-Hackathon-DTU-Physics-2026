import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

class GlobalInnovationNumber(object):
    def __init__(self): self._num = -1
    def next(self): self._num += 1; return self._num

class GlobalLayerNumber(object):
    def __init__(self): self._num = 0
    def next(self): self._num += 1; return self._num
    def current(self): return self._num

class GlobalSpeciesNumber(object):
    def __init__(self): self._num = -1
    def next(self): self._num += 1; return self._num

def energy_from_circuit(circuit, parameters, shots):
    return 0 

def configure_circuit_to_backend(circuit, backend):
    sim = AerSimulator()
    return transpile(circuit, sim), sim

def get_circuit_properties(circuit, backend):
    return 0
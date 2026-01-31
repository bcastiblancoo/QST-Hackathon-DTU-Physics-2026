import copy
import numpy as np
from . import helper as h
from . import genome as gen
from . import gate as g
from . import species as s
import random
from . import logger as log

class QNEAT:
    def __init__(self, population_size:int, n_qubits:int):
        self.global_innovation_number = h.GlobalInnovationNumber()
        self.global_layer_number = h.GlobalLayerNumber()
        self.global_species_number = h.GlobalSpeciesNumber()
        self.n_qubits = n_qubits
        self.compatibility_threshold = 3
        self.prob_mutation_without_crossover = 0.25
        self.percentage_survivors = 0.5
        self.generation = 0
        self.best_fitness = None
        self.species = []
        self.population_size = population_size

        self.population = []
        for _ in range(population_size):
            genome = gen.Genome(self.global_layer_number)
            gate_type = np.random.choice(list(g.GateType))
            qubit = np.random.randint(n_qubits)
            gate = g.GateGene(self.global_innovation_number.next(), gate_type, qubit)
            genome.add_gate(gate)
            self.population.append(genome)

        self.speciate(0)

    def generate_new_population(self, backend):
        # 1. Sort and clean species
        for specie in self.species:
            # Sort genomes by fitness (descending)
            specie.genomes.sort(key=lambda x: x._fitness if x._fitness is not None else -1000, reverse=True)

        new_population = []
        
        # 2. Process each species
        for specie in self.species:
            if not specie.genomes: continue
            
            # ELITISM: Always keep the Champion (unmutated)
            champion = copy.deepcopy(specie.genomes[0])
            new_population.append(champion)
            
            # Survival of the fittest cutoff
            n_survivors = max(1, int(len(specie.genomes) * self.percentage_survivors))
            survivors = specie.genomes[:n_survivors]

            # 3. Create Offspring
            n_offspring = len(specie.genomes) - 1 
            
            for _ in range(n_offspring):
                child = None
                # Crossover
                if len(survivors) > 1 and random.random() > 0.75:
                    p1, p2 = random.sample(survivors, 2)
                    child = gen.Genome.crossover(p1, p2, self.n_qubits, backend)
                else:
                    # Asexual reproduction
                    child = copy.deepcopy(random.choice(survivors))
                
                # FORCE MUTATION
                child.mutate(self.global_innovation_number, self.n_qubits)
                new_population.append(child)
            
            specie.empty()
            
        # 4. Fill remaining slots
        if len(new_population) > 0:
            best_genome = new_population[0] 
        else:
            best_genome = gen.Genome(self.global_layer_number)

        while len(new_population) < self.population_size:
            # Clone a random individual from the new pool to maintain diversity
            parent = random.choice(new_population)
            clone = copy.deepcopy(parent)
            clone.mutate(self.global_innovation_number, self.n_qubits)
            new_population.append(clone)
            

        self.population = new_population[:self.population_size]

    def speciate(self, generation):
        for genome in self.population:
            found = False
            for specie in self.species:
                if gen.Genome.compatibility_distance(genome, specie.representative) < self.compatibility_threshold:
                    specie.add(genome)
                    found = True
                    break
            if not found:
                new_s = s.Species(generation, self.global_species_number.next())
                new_s.update(genome, [genome])
                self.species.append(new_s)
        
        self.species = [s for s in self.species if s.update_representative()]

    def run_generation(self, backend):
        self.generate_new_population(backend)
        self.speciate(self.generation)
        self.generation += 1
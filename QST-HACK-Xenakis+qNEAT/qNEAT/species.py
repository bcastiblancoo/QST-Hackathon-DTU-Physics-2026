class Species:
    def __init__(self, generation, key=None):
        self.genomes = []
        self.representative = None

    def update(self, representative, genomes):
        self.representative = representative
        self.genomes = genomes

    def empty(self): self.genomes = []
    def add(self, genome): self.genomes.append(genome)

    def update_representative(self):
        if not self.genomes: return False
        self.representative = self.genomes[0]
        return True
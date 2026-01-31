# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 11:05:18 2026

@author: Elian PC
"""

import csv
import os
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class GenRecord:
    algo: str
    run_id: str
    molecule: str
    generation: int
    best_energy: float
    best_fitness: float
    best_complexity: float  # genes, gates, params, etc.
    pop_size: int

class ExperimentLogger:
    def __init__(self, outdir="results"):
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)

    def make_run_id(self, algo):
        t = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{algo}_{t}"

    def csv_path(self, run_id):
        return os.path.join(self.outdir, f"{run_id}.csv")

    def init_csv(self, run_id):
        path = self.csv_path(run_id)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=GenRecord.__annotations__.keys())
            writer.writeheader()
        return path

    def log(self, record: GenRecord):
        path = self.csv_path(record.run_id)
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=GenRecord.__annotations__.keys())
            writer.writerow(asdict(record))

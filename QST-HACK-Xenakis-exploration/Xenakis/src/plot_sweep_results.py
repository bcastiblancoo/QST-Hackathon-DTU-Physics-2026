# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the results
df = pd.read_csv("sweep_results.csv")

# Ensure numeric types
df["fitness"] = pd.to_numeric(df["fitness"], errors="coerce")
df["energy"] = pd.to_numeric(df["energy"], errors="coerce")
df["generation"] = pd.to_numeric(df["generation"], errors="coerce")

# Drop rows with missing values
df = df.dropna()

# --- Plot 1: Fitness vs Energy, colored by Generation ---
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df,
    x="energy", y="fitness",
    hue="generation",
    palette="viridis",
    style="is_best",
    s=80
)
plt.title("Fitness vs Energy (colored by Generation)")
plt.xlabel("Energy")
plt.ylabel("Fitness")
plt.grid(True)
plt.legend(title="", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig("fitness_vs_energy.png", dpi=300)
plt.show()

# --- Plot 2: Fitness evolution per generation ---
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x="generation", y="fitness", palette="coolwarm")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.title("Fitness Distribution Across Generations")
plt.grid(True)
plt.tight_layout()
plt.savefig("fitness_per_generation.png", dpi=300)
plt.show()

# --- Plot 3: Energy evolution per generation ---
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x="generation", y="energy", palette="YlGnBu")
plt.xlabel("Generation")
plt.ylabel("Energy")
plt.title("Energy Distribution Across Generations")
plt.grid(True)
plt.tight_layout()
plt.savefig("energy_per_generation.png", dpi=300)
plt.show()

# --- Plot 4: Highlight best genome ---
best_row = df[df["is_best"] == True].sort_values("generation").iloc[-1]
print("\n Best Genome Info:\n", best_row)

df.to_csv("full_ga_results_clean.csv", index=False)

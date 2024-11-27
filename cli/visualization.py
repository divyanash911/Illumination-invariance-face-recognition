import matplotlib.pyplot as plt
import numpy as np

def plot_similarity_scores(scores, labels, save = False):
    plt.bar(labels, scores)
    plt.xlabel("Methods")
    plt.ylabel("Similarity Scores")
    plt.title("Similarity Score Comparison")
    plt.xticks(rotation=45)
    if save:
        plt.savefig("similarity_scores.png")
    plt.show()

def plot_density(alpha_grid, mu_grid, density, save = False):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(mu_grid, alpha_grid)
    ax.plot_surface(X, Y, density, cmap='viridis', edgecolor='none')
    ax.set_title("Density Estimate")
    ax.set_xlabel("μ")
    ax.set_ylabel("α")
    ax.set_zlabel("Density")
    if save:
        plt.savefig("density_estimate.png")
    plt.show()

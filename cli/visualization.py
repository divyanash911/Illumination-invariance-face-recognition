import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math


def plot_density(alpha_grid, mu_grid, density, epoch, save=True):
    """Plot density as a 3D surface."""
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(mu_grid, alpha_grid)
    ax.plot_surface(X, Y, density, cmap='viridis', edgecolor='none')
    ax.set_title(f"Density Estimate after {epoch} Epochs")
    ax.set_xlabel("μ")
    ax.set_ylabel("α")
    ax.set_zlabel("Density")

    if save:
        plt.savefig(f"output/density_{epoch}.png")
    plt.show()


def visualize_similarities(query_image, dataset, filter_function, similarity_function, alpha_grid, mu_grid, density, idx, top_n=25, save=True, output_dir="output"):
    """Visualize the top N matches for the query image with improved spacing between images and text."""
    unprocessed_similarities = []
    for gallery_idx, gallery_image_val in enumerate(dataset):
        if gallery_idx == idx:
            continue

        gallery_image = gallery_image_val["image_data"]
        unprocessed_sim = similarity_function(query_image, gallery_image)
        unprocessed_similarities.append(
            (gallery_idx, unprocessed_sim, gallery_image_val))

    unprocessed_similarities.sort(key=lambda x: x[1], reverse=True)

    mu = abs(unprocessed_similarities[0][1] - unprocessed_similarities[1][1]) if len(unprocessed_similarities) > 1 else 0
    mu_idx = np.argmin(np.abs(mu_grid - mu))
    alpha_probabilities = density[:, mu_idx]
    best_alpha_idx = np.argmax(alpha_probabilities)
    best_alpha = alpha_grid[best_alpha_idx]

    similarities = []
    for _, unprocessed_sim, gallery_image_val in unprocessed_similarities:
        gallery_image = gallery_image_val["image_data"]
        filtered_sim = similarity_function(filter_function(query_image), filter_function(gallery_image))
        combined_similarity = best_alpha * unprocessed_sim + (1 - best_alpha) * filtered_sim
        similarities.append({
            'image': gallery_image,
            'subject_id': gallery_image_val['subject_id'],
            'illumination': gallery_image_val['illumination'],
            'combined_sim': unprocessed_sim,
            'unprocessed_sim': combined_similarity
        })

    similarities.sort(key=lambda x: x['combined_sim'], reverse=True)
    top_similarities = similarities[:top_n]

    # Create the grid for the query image and top matches
    n_cols = 5
    n_rows = math.ceil((top_n + 1) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
    axes = axes.flatten()

    # Display the query image
    axes[0].imshow(query_image, cmap='gray')
    axes[0].set_title('Query Image', fontsize=10, pad=10)
    axes[0].axis('off')

    # Display the top matches with improved spacing
    for i, sim_data in enumerate(top_similarities, start=1):
        axes[i].imshow(sim_data['image'], cmap='gray')
        axes[i].set_title(
            f"ID: {sim_data['subject_id']}\nCombined: {sim_data['combined_sim']:.2f}\nUnprocessed: {sim_data['unprocessed_sim']:.2f}",
            fontsize=8,
            pad=15  # Add padding between the image and text
        )
        axes[i].axis('off')

    # Hide any unused subplots
    for i in range(len(top_similarities) + 1, len(axes)):
        axes[i].axis('off')

    # Adjust layout for spacing using subplots_adjust
    fig.subplots_adjust(wspace=0.5, hspace=1.5)
    
    # Save the plot if required
    if save:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(os.path.join(output_dir, "similarities_with_spacing.png"))
    plt.show()

    """Visualize the top N matches for the query image with improved spacing between images and text."""
    unprocessed_similarities = []
    for gallery_idx, gallery_image_val in enumerate(dataset):
        if gallery_idx == idx:
            continue

        gallery_image = gallery_image_val["image_data"]
        unprocessed_sim = similarity_function(query_image, gallery_image)
        unprocessed_similarities.append(
            (gallery_idx, unprocessed_sim, gallery_image_val))

    unprocessed_similarities.sort(key=lambda x: x[1], reverse=True)

    mu = abs(unprocessed_similarities[0][1] - unprocessed_similarities[1][1]) if len(unprocessed_similarities) > 1 else 0
    mu_idx = np.argmin(np.abs(mu_grid - mu))
    alpha_probabilities = density[:, mu_idx]
    best_alpha_idx = np.argmax(alpha_probabilities)
    best_alpha = alpha_grid[best_alpha_idx]

    similarities = []
    for _, unprocessed_sim, gallery_image_val in unprocessed_similarities:
        gallery_image = gallery_image_val["image_data"]
        filtered_sim = similarity_function(filter_function(query_image), filter_function(gallery_image))
        combined_similarity = best_alpha * unprocessed_sim + (1 - best_alpha) * filtered_sim
        similarities.append({
            'image': gallery_image,
            'subject_id': gallery_image_val['subject_id'],
            'illumination': gallery_image_val['illumination'],
            'combined_sim': unprocessed_sim,
            'unprocessed_sim': combined_similarity
        })

    similarities.sort(key=lambda x: x['combined_sim'], reverse=True)
    top_similarities = similarities[:top_n]

    # Create the grid for the query image and top matches
    n_cols = 5
    n_rows = math.ceil((top_n + 1) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
    axes = axes.flatten()

    # Display the query image
    axes[0].imshow(query_image, cmap='gray')
    axes[0].set_title('Query Image', fontsize=10, pad=10)
    axes[0].axis('off')

    # Display the top matches with improved spacing
    for i, sim_data in enumerate(top_similarities, start=1):
        axes[i].imshow(sim_data['image'], cmap='gray')
        axes[i].set_title(
            f"ID: {sim_data['subject_id']}\nCombined: {sim_data['combined_sim']:.2f}\nUnprocessed: {sim_data['unprocessed_sim']:.2f}",
            fontsize=8,
            pad=10 # Add padding between the image and text
        )
        axes[i].axis('off')

    # Hide any unused subplots
    for i in range(len(top_similarities) + 1, len(axes)):
        axes[i].axis('off')

    # Adjust layout for spacing
    plt.tight_layout(pad=3.0)
    
    # Save the plot if required
    if save:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(os.path.join(output_dir, "similarities_with_spacing.png"))
    plt.show()

    """Visualize the top N matches for the query image with improved spacing."""
    unprocessed_similarities = []
    for gallery_idx, gallery_image_val in enumerate(dataset):
        if gallery_idx == idx:
            continue

        gallery_image = gallery_image_val["image_data"]
        unprocessed_sim = similarity_function(query_image, gallery_image)
        unprocessed_similarities.append(
            (gallery_idx, unprocessed_sim, gallery_image_val))

    unprocessed_similarities.sort(key=lambda x: x[1], reverse=True)

    mu = abs(unprocessed_similarities[0][1] - unprocessed_similarities[1][1]) if len(unprocessed_similarities) > 1 else 0
    mu_idx = np.argmin(np.abs(mu_grid - mu))
    alpha_probabilities = density[:, mu_idx]
    best_alpha_idx = np.argmax(alpha_probabilities)
    best_alpha = alpha_grid[best_alpha_idx]

    similarities = []
    for _, unprocessed_sim, gallery_image_val in unprocessed_similarities:
        gallery_image = gallery_image_val["image_data"]
        filtered_sim = similarity_function(filter_function(query_image), filter_function(gallery_image))
        combined_similarity = best_alpha * unprocessed_sim + (1 - best_alpha) * filtered_sim
        similarities.append({
            'image': gallery_image,
            'subject_id': gallery_image_val['subject_id'],
            'illumination': gallery_image_val['illumination'],
            'combined_sim': unprocessed_sim,
            'unprocessed_sim': combined_similarity
        })

    similarities.sort(key=lambda x: x['combined_sim'], reverse=True)
    top_similarities = similarities[:top_n]

    # Create the grid for the query image and top matches
    n_cols = 5
    n_rows = math.ceil((top_n + 1) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
    axes = axes.flatten()

    # Display the query image
    axes[0].imshow(query_image, cmap='gray')
    axes[0].set_title('Query Image')
    axes[0].axis('off')

    # Display the top matches
    for i, sim_data in enumerate(top_similarities, start=1):
        axes[i].imshow(sim_data['image'], cmap='gray')
        axes[i].set_title(f"ID: {sim_data['subject_id']}\nCombined: {sim_data['combined_sim']:.2f}\nUnprocessed: {sim_data['unprocessed_sim']:.2f}")
        axes[i].axis('off')

    # Hide any unused subplots
    for i in range(len(top_similarities) + 1, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    
    # Save the plot if required
    if save:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(os.path.join(output_dir, "similarities.png"))
    plt.show()
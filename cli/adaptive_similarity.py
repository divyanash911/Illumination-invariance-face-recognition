import numpy as np
from scipy.ndimage import gaussian_filter
import os


def adaptive_similarity(query_image, gallery_images, filter_function, similarity_function):
    """Calculate adaptive similarity using confusion margin."""
    unprocessed_similarities = [similarity_function(
        query_image, gallery) for gallery in gallery_images]
    filtered_similarities = [similarity_function(filter_function(
        query_image), filter_function(gallery)) for gallery in gallery_images]

    top_similarities = sorted(unprocessed_similarities, reverse=True)[:2]
    confusion_margin = top_similarities[0] - top_similarities[1]

    alpha = 1 / (1 + np.exp(-confusion_margin * 10))  # Sigmoid weighting
    combined_similarity = [
        alpha * u + (1 - alpha) * f for u, f in zip(unprocessed_similarities, filtered_similarities)]

    return np.max(combined_similarity)


def estimate_joint_probability_persons(
    training_data, filter_function, similarity_function, alpha_grid_size, mu_grid_size, n_iter=500
):
    """
    Corrected implementation of joint probability density estimation.
    """
    alpha_grid = np.linspace(0, 1, alpha_grid_size)
    mu_grid = np.linspace(0, 1, mu_grid_size)
    density = np.zeros((alpha_grid_size, mu_grid_size))

    for _ in range(n_iter):
        # Step 2: Randomly select a query image
        query_idx = np.random.choice(len(training_data))
        query_image = training_data[query_idx]["image_data"]
        query_id = training_data[query_idx]["subject_id"]
        query_illumination = training_data[query_idx]["illumination"]

        # Compute similarities
        similarities_same_person = []
        similarities_other_persons = []

        for gallery_data in training_data:
            gallery_image = gallery_data["image_data"]
            gallery_id = gallery_data["subject_id"]
            gallery_illumination = gallery_data["illumination"]

            sim_raw = similarity_function(query_image, gallery_image)
            sim_filtered = similarity_function(
                filter_function(query_image), filter_function(gallery_image)
            )

            if gallery_id == query_id and gallery_illumination != query_illumination:
                similarities_same_person.append((sim_raw, sim_filtered))
            elif gallery_id != query_id:
                similarities_other_persons.append((sim_raw, sim_filtered))

        # Step 3: Compute confusion margin (μ)
        # if len(similarities_same_person) >= 2:
        #     top_1_sim = max(sim[0] for sim in similarities_same_person)
        #     top_2_sim = max(sim[0] for sim in similarities_same_person if sim[0] != top_1_sim)
        #     mu = top_1_sim - top_2_sim
        # else:
        #     mu = 0

        # Step 3: Compute confusion margin (μ)
        mu = max(sim[0] for sim in similarities_same_person) - \
            max(sim[0] for sim in similarities_other_persons)
        mu_idx = np.argmin(np.abs(mu_grid - mu))

        # Step 4: Iterate over α grid
        for alpha_idx, alpha in enumerate(alpha_grid):
            # Compute combined similarity scores for query
            similarity_query = [
                alpha * sim[0] + (1 - alpha) * sim[1] for sim in similarities_same_person
            ]
            similarity_other = [
                alpha * sim[0] + (1 - alpha) * sim[1] for sim in similarities_other_persons
            ]

            # Compute numerator and denominator for δ(kΔα)
            numerator = max(similarity_query) if similarity_query else 0
            # Avoid division by zero
            denominator = max(similarity_other) if similarity_other else 1

            delta = numerator / denominator if denominator != 0 else 0

            # Update density
            density[alpha_idx, mu_idx] += delta

    # Step 7: Apply Gaussian smoothing
    smoothing_sigma = 0.05 * max(alpha_grid_size, mu_grid_size)
    density = gaussian_filter(density, sigma=smoothing_sigma)

    # Step 8: Normalize density to unit integral
    density /= np.sum(density)

    return alpha_grid, mu_grid, density

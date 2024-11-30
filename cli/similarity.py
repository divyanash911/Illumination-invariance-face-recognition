from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA
import numpy as np
import face_recognition

def deep_features_similarity(image1, image2):
    image1_embedding = face_recognition.face_encodings(image1)[0]
    image2_embedding = face_recognition.face_encodings(image2)[0]
    return 1 - cosine(image1_embedding, image2_embedding)

def get_best_match_using_face_recognition(query_image, dataset):
    query_embedding = face_recognition.face_encodings(query_image)[0]
    best_match = None
    similarity_score = -np.inf
    
    for gallery_image_val in dataset:
        gallery_image = gallery_image_val["image_data"]
        gallery_embedding = face_recognition.face_encodings(gallery_image)[0]
        similarity = 1 - cosine(query_embedding, gallery_embedding)
        
        if similarity > similarity_score:
            similarity_score = similarity
            best_match = gallery_image_val
    
    return best_match, similarity_score

def cosine_similarity(image1, image2):
    return 1 - cosine(image1.flatten(), image2.flatten())

def mutual_subspace_method(images1, images2, n_components=6):
    pca = PCA(n_components=n_components)
    subspace1 = pca.fit_transform(images1)
    subspace2 = pca.fit_transform(images2)
    
    u, _, vh = np.linalg.svd(subspace1.T @ subspace2)
    correlations = np.diag(u.T @ vh)
    return np.mean(correlations[:3])

def match_faces(query_image, training_data, filter_function, similarity_function, alpha_grid, mu_grid, density, idx):
    """
    Match a query face with the most similar face in the training data using joint probability density,
    ensuring that the query image itself is not considered.
    
    Parameters:
    -----------
    query_image : array-like
        The query image data.
    training_data : list of dicts
        Each dict contains:
            - "image_data" : array-like image data
            - "subject_id" : unique identifier for the subject
            - "illumination" : illumination condition of the image
    filter_function : callable
        Function to filter/process images.
    similarity_function : callable
        Function to compute similarity between two images.
    alpha_grid : array-like
        Grid of (alpha) values.
    mu_grid : array-like
        Grid of (mu) values.
    density : 2D array
        Joint probability density estimated over ((alpha, mu)).
    idx : int
        Index of the query image in the training data.
    
    Returns:
    --------
    best_match : dict
        Information about the best-matching image (from training_data).
    similarity_score : float
        Highest similarity score achieved.
    best_alpha : float
        Optimal (alpha) determined by the density for the given confusion margin.
    """
    best_match = None
    similarity_score = -np.inf  # Initialize with a very low similarity
    best_alpha = None
    
    # Compute similarities and confusion margin (μ)
    similarities = []
    best_unmatched = None 
    similarity_unmatched = -np.inf 
    for gallery_idx, gallery_image_val in enumerate(training_data):
        if gallery_idx == idx:  # Skip the query image itself
            continue
        
        gallery_image = gallery_image_val["image_data"]
        
        # Compute unprocessed and filtered similarities
        unprocessed_sim = similarity_function(query_image, gallery_image)
        
        if unprocessed_sim>similarity_unmatched:
            best_unmatched = gallery_image
        
        filtered_sim = similarity_function(
            filter_function(query_image), 
            filter_function(gallery_image)
        )
        
        similarities.append({
            "gallery_image": gallery_image_val,
            "unprocessed_sim": unprocessed_sim,
            "filtered_sim": filtered_sim
        })
    
    # Sort similarities by unprocessed similarity (descending)
    similarities.sort(key=lambda x: x["unprocessed_sim"], reverse=True)
    
    # Compute confusion margin (μ)
    top_1_sim = similarities[0]["unprocessed_sim"]
    top_2_sim = similarities[1]["unprocessed_sim"]
    mu = top_1_sim - top_2_sim
    mu_idx = np.argmin(np.abs(mu_grid - mu))  # Find the closest \(\mu\) grid point
    
    # Determine \(\alpha\) based on the density
    alpha_probabilities = density[:, mu_idx]
    best_alpha_idx = np.argmax(alpha_probabilities)
    best_alpha = alpha_grid[best_alpha_idx]
    
    # Compute the similarity for the best \(\alpha\)
    for similarity in similarities:
        gallery_image = similarity["gallery_image"]
        unprocessed_sim = similarity["unprocessed_sim"]
        filtered_sim = similarity["filtered_sim"]
        
        combined_similarity = best_alpha * unprocessed_sim + (1 - best_alpha) * filtered_sim
        
        if combined_similarity > similarity_score:
            similarity_score = combined_similarity
            best_match = gallery_image
    
    return best_match, similarity_score, best_alpha, best_unmatched

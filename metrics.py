import os
from itertools import combinations
import cv2
import numpy as np
from scipy.linalg import svd

def preprocess_image(image_path, size=(64, 64)):
    # Load the image in grayscale and resize it
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, size)
    # Flatten the image to a 1D vector
    img_vector = img.flatten()
    return img_vector

def mutual_subspace_method(image_path1, image_path2, n_components=10):
    # Preprocess the images
    img1_vector = preprocess_image(image_path1)
    img2_vector = preprocess_image(image_path2)
    
    # Stack the images into matrices (each column is a flattened image)
    X1 = img1_vector.reshape(-1, 1)
    X2 = img2_vector.reshape(-1, 1)
    
    # Compute the SVD to get principal subspace components
    U1, _, _ = svd(X1, full_matrices=False)
    U2, _, _ = svd(X2, full_matrices=False)
    
    # Select the top `n_components` vectors for each subspace
    U1 = U1[:, :n_components]
    U2 = U2[:, :n_components]
    
    # Compute the cosine of the principal angles between the two subspaces
    # by calculating singular values of U1.T @ U2
    S = svd(U1.T @ U2, compute_uv=False)
    principal_angles = np.arccos(S)
    
    # The similarity measure can be the cosine of the first principal angle
    similarity = np.cos(principal_angles[0])
    
    return similarity

def constrained_mutual_subspace_method(image_path1, image_path2, n_components=10, constraint_dim=5):
    # Preprocess the images
    img1_vector = preprocess_image(image_path1)
    img2_vector = preprocess_image(image_path2)
    
    # Stack the images into matrices (each column is a flattened image)
    X1 = img1_vector.reshape(-1, 1)
    X2 = img2_vector.reshape(-1, 1)
    
    # Compute the SVD to get principal subspace components
    U1, _, _ = svd(X1, full_matrices=False)
    U2, _, _ = svd(X2, full_matrices=False)
    
    # Select the top `n_components` vectors for each subspace
    U1 = U1[:, :n_components]
    U2 = U2[:, :n_components]
    
    # Apply constraint by limiting the dimensions to `constraint_dim`
    U1_constrained = U1[:, :constraint_dim]
    U2_constrained = U2[:, :constraint_dim]
    
    # Compute the cosine of the principal angles between the constrained subspaces
    S = svd(U1_constrained.T @ U2_constrained, compute_uv=False)
    principal_angles = np.arccos(S)
    
    # The similarity measure can be the cosine of the first principal angle
    similarity = np.cos(principal_angles[0])
    
    return similarity


    
def find_image_similarity(image_dir, n_components=10, constraint_dim=5):
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    image_combinations = list(combinations(image_files, 2))

    min_similarity = float('inf')
    max_similarity = float('-inf')

    min_similarity_pair = None
    max_similarity_pair = None

    for image1, image2 in image_combinations:
        image_path1 = os.path.join(image_dir, image1)
        image_path2 = os.path.join(image_dir, image2)

        similarity = constrained_mutual_subspace_method(image_path1, image_path2, n_components, constraint_dim)

        if similarity < min_similarity:
            min_similarity = similarity
            min_similarity_pair = (image1, image2)

        if similarity > max_similarity:
            max_similarity = similarity
            max_similarity_pair = (image1, image2)

    return min_similarity, min_similarity_pair, max_similarity, max_similarity_pair

    



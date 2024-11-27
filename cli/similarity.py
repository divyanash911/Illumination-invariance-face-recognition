from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA
import numpy as np

def cosine_similarity(image1, image2):
    return 1 - cosine(image1.flatten(), image2.flatten())

def mutual_subspace_method(images1, images2, n_components=6):
    pca = PCA(n_components=n_components)
    subspace1 = pca.fit_transform(images1)
    subspace2 = pca.fit_transform(images2)
    
    u, _, vh = np.linalg.svd(subspace1.T @ subspace2)
    correlations = np.diag(u.T @ vh)
    return np.mean(correlations[:3])

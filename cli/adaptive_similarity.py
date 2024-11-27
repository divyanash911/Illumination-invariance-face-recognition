import numpy as np

def adaptive_similarity(query_image, gallery_images, filter_function, similarity_function):
    unprocessed_similarities = [similarity_function(query_image, gallery) for gallery in gallery_images]
    filtered_similarities = [similarity_function(filter_function(query_image), filter_function(gallery)) for gallery in gallery_images]
    
    top_similarities = sorted(unprocessed_similarities, reverse=True)[:2]
    confusion_margin = top_similarities[0] - top_similarities[1]
    
    alpha = 1 / (1 + np.exp(-confusion_margin * 10))  # Sigmoid function
    combined_similarity = [alpha * u + (1 - alpha) * f for u, f in zip(unprocessed_similarities, filtered_similarities)]
    
    return np.max(combined_similarity)

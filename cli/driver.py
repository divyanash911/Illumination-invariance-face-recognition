from filters import gaussian_highpass, self_quotient_image
from similarity import cosine_similarity
from data_utils import load_image, load_dataset, load_paths
from visualization import plot_similarity_scores
import face_recognition
import numpy as np

def deep_learning_similarity(query_image_path, gallery_image_paths):
    """
    Computes similarity scores using deep learning-based face recognition.

    Parameters:
    - query_image_path (str): Path to the query image.
    - gallery_image_paths (list): List of paths to gallery images.

    Returns:
    - scores (list): Similarity scores between the query image and each gallery image.
    """
    try:
        # Load query image as RGB
        query_image = face_recognition.load_image_file(query_image_path)
        query_encodings = face_recognition.face_encodings(query_image)
        if not query_encodings:
            raise ValueError("No face detected in query image.")
        query_encoding = query_encodings[0]

        scores = []
        for gallery_image_path in gallery_image_paths:
            # Load gallery image as RGB
            gallery_image = face_recognition.load_image_file(gallery_image_path)
            gallery_encodings = face_recognition.face_encodings(gallery_image)
            if not gallery_encodings:
                print(f"Warning: No face detected in gallery image {gallery_image_path}. Skipping...")
                scores.append(0)
                continue

            gallery_encoding = gallery_encodings[0]
            # Compute similarity (inverse of face distance)
            score = 1 - face_recognition.face_distance([query_encoding], gallery_encoding)[0]
            scores.append(score)

        return scores
    except Exception as e:
        print(f"Error during deep learning similarity computation: {e}")
        return []

def driver_comparative():
    """
    Driver script for comparative study.
    """
    print("Starting Comparative Study for Face Recognition Methods.")

    # Load query and gallery dataset
    query_image_path = "../query.jpg"
    gallery_path = "../data_new/"

    gallery_images = load_dataset(gallery_path)
    query_image = load_image(query_image_path)

    gallery_image_paths = load_paths(gallery_path)
    
    # Filter-based pipeline
    print("\nComputing similarity using filter-based pipeline...")
    sq_filter_scores = [
        cosine_similarity(self_quotient_image(query_image), self_quotient_image(img))
        for img in gallery_images
    ]
    # print("Filter-based scores:", sq_filter_scores)

    # Deep learning-based face recognition
    print("\nComputing similarity using deep learning-based solution...")
    dl_scores = deep_learning_similarity(query_image_path, gallery_image_paths)
    # print("Deep learning-based scores:", dl_scores)

    # Plot comparison
    labels = [f"Image {i+1}" for i in range(len(gallery_images))]
    plot_similarity_scores([d - s for d, s in zip(dl_scores, sq_filter_scores)], labels, save=True)

if __name__ == "__main__":
    driver_comparative()

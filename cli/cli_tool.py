import os
from filters2 import gaussian_highpass, self_quotient_image, laplacian_of_gaussian, gradient_filters
from similarity import cosine_similarity
from adaptive_similarity import adaptive_similarity
from data_utils import load_image, load_dataset
from visualization import plot_similarity_scores
import face_recognition

# Mapping filters and similarity functions for user selection
FILTERS = {
    "gaussian_highpass": gaussian_highpass,
    "self_quotient": self_quotient_image,
    "laplacian_of_gaussian": laplacian_of_gaussian,
    "gradient_filters": gradient_filters,
}

def display_menu(title, options):
    """
    Displays a menu and returns the user's choice.
    """
    print(f"\n{title}\n" + "-" * len(title))
    for idx, option in enumerate(options, 1):
        print(f"{idx}. {option}")
    try:
        choice = int(input(f"Enter your choice (1-{len(options)}): "))
        if 1 <= choice <= len(options):
            return choice - 1
    except ValueError:
        pass
    print("Invalid choice. Please try again.")
    return display_menu(title, options)

def deep_learning_similarity(query_image, gallery_images):
    """
    Computes similarity scores using deep learning-based face recognition.
    """
    query_encoding = face_recognition.face_encodings(face_recognition.load_image_file(query_image))[0]
    scores = []
    for gallery_image in gallery_images:
        gallery_encoding = face_recognition.face_encodings(face_recognition.load_image_file(gallery_image))[0]
        score = 1 - face_recognition.face_distance([query_encoding], gallery_encoding)[0]
        scores.append(score)
    return scores

def interactive_cli_with_dl():
    """
    Continuous interactive CLI with Deep Learning-based face recognition.
    """
    print("Welcome to the Enhanced Face Recognition CLI Tool!")

    query_image = None
    gallery_images = None

    while True:
        main_menu = ["Load Query Image", "Load Gallery Dataset", "Perform Filter-Based Operations",
                     "Use Deep Learning-Based Face Recognition", "Quit"]
        main_choice = display_menu("Main Menu", main_menu)

        if main_choice == 0:  # Load Query Image
            query_image_path = input("Enter the path to the query image: ")
            query_image = query_image_path

        elif main_choice == 1:  # Load Gallery Dataset
            gallery_path = input("Enter the path to the gallery dataset: ")
            gallery_images = [os.path.join(gallery_path, f) for f in os.listdir(gallery_path) if f.endswith('.jpg')]

        elif main_choice == 2:  # Filter-Based Operations
            if not query_image or not gallery_images:
                print("Please load both query image and gallery dataset first.")
                continue

            filter_names = list(FILTERS.keys())
            filter_choice = display_menu("Select a filter to apply", filter_names)
            filter_function = FILTERS[filter_names[filter_choice]]

            scores = [
                cosine_similarity(filter_function(load_image(query_image)), filter_function(load_image(img)))
                for img in gallery_images
            ]
            print("Filter-based scores:", scores)

            # Plot results
            labels = [f"Image {i+1}" for i in range(len(scores))]
            plot_similarity_scores(scores, labels)

        elif main_choice == 3:  # Deep Learning-Based Face Recognition
            if not query_image or not gallery_images:
                print("Please load both query image and gallery dataset first.")
                continue

            dl_scores = deep_learning_similarity(query_image, gallery_images)
            print("Deep Learning-based scores:", dl_scores)

            # Plot results
            labels = [f"Image {i+1}" for i in range(len(dl_scores))]
            plot_similarity_scores(dl_scores, labels)

        elif main_choice == 4:  # Quit
            print("Thank you for using the CLI Tool. Goodbye!")
            break

if __name__ == "__main__":
    interactive_cli_with_dl()

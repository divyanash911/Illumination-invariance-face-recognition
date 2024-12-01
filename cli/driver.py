import os
import glob
import matplotlib.pyplot as plt
from climage import convert_array
from InquirerPy import prompt

from adaptive_similarity import *
from data_utils import *
from visualization import *
from similarity import *
from filters import *
import face_recognition

import warnings
warnings.filterwarnings("ignore")


def preview_image(image, title="", in_terminal=False):
    """Display image using matplotlib or ASCII art in the terminal."""
    if in_terminal:
        # Convert image to RGB from grayscale if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = cv2.resize(image, (50, 50))
        art = convert_array(image, is_unicode=True)
        print(art)
    else:
        plt.imshow(image, cmap="gray")
        plt.axis("off")
        plt.title(title)
        if title:
            plt.savefig(f"output/{title}.png")
        plt.show()


def validate_file_path(file_path, description="file"):
    """Validate if the given file path exists."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: {description.capitalize()} '{
                                file_path}' not found.")


def load_gallery_images(data_dir, dl=False):
    """Load gallery images from the specified directory."""
    dataset = []
    for file in glob.glob(os.path.join(data_dir, "*.jpg")):
        subject_id = extract_subject_id(file)
        illumination = extract_subject_lighting(file)
        image_data = face_recognition.load_image_file(
            file) if dl else load_image(file)
        dataset.append({
            "subject_id": subject_id,
            "illumination": illumination,
            "image_data": image_data,
            "query_name": os.path.basename(file),
        })
    return dataset


def estimate_or_load_density(dataset, filter_method, similarity_method, args):
    """Estimate or load joint probability density."""
    if not args["inference"]:
        return load_from_files()
    else:
        alpha_grid, mu_grid, density = estimate_joint_probability_persons(
            dataset,
            filter_method,
            similarity_method,
            alpha_grid_size=args["grid_size"],
            mu_grid_size=args["grid_size"],
            n_iter=args["iterations"],
        )
        save_to_files(alpha_grid, mu_grid, density)
        return alpha_grid, mu_grid, density


def get_cli_inputs():
    """Use InquirerPy to collect user inputs interactively."""
    questions = [
        {
            "type": "input",
            "name": "data_dir",
            "message": "Enter the directory path for gallery images:",
            "validate": lambda val: os.path.exists(val) or "Directory not found. Please enter a valid path.",
        },
        {
            "type": "input",
            "name": "query_path",
            "message": "Enter the path for the query image:",
            "validate": lambda val: os.path.exists(val) or "File not found. Please enter a valid path.",
        },
        {
            "type": "list",
            "name": "method",
            "message": "Select the similarity method:",
            "choices": ["cosine", "mutual_subspace"],
            "default": "cosine",
        },
        {
            "type": "list",
            "name": "filter",
            "message": "Select the filtering method:",
            "choices": [
                "self_quotient",
                "gaussian_highpass",
                "laplacian_of_gaussian",
                "homomorphic_filter",
                "edge_map",
                "cannys_edge_detector",
            ],
            "default": "homomorphic_filter",
        },
        {
            "type": "confirm",
            "name": "in_terminal",
            "message": "Enable in-terminal image viewing using ASCII art?",
            "default": False,
        },
        {
            "type": "confirm",
            "name": "inference",
            "message": "Use saved density files for inference?",
            "default": False,
        },
        {
            "type": "number",
            "name": "grid_size",
            "message": "Enter grid size for joint probability density estimation (default: 100):",
            "default": 100,
            "validate": lambda val: val.isdigit() or "Please enter a valid integer.",
        },
        {
            "type": "number",
            "name": "iterations",
            "message": "Enter number of iterations for density estimation (default: 500):",
            "default": 500,
            "validate": lambda val: val.isdigit() or "Please enter a valid integer.",
        },
    ]
    return prompt(questions)


def cli_menu():
    """Generate a user-friendly interactive menu using InquirerPy."""
    choices = [
        {"name": "View query image", "value": "view_query"},
        {"name": "View best match image", "value": "view_best_match"},
        {"name": "View unmatched image", "value": "view_unmatched"},
        {"name": "Use pretrained face recognition model for matching",
            "value": "use_pretrained_model"},
        {"name": "Visualize similarities with dataset",
            "value": "visualize_similarities"},
        {"name": "Exit", "value": "exit"}
    ]
    questions = [
        {
            "type": "list",
            "message": "What would you like to do?",
            "choices": choices,
            "name": "menu_choice",
        }
    ]
    return prompt(questions)["menu_choice"]


def handle_user_choice(choice, query_image, best_match, unmatched, dataset, args, alpha_grid, mu_grid, density, query_index, similarity_method, filter_method):
    """Handle user interaction based on menu choice."""
    if choice == "view_query":
        print("Displaying Query Image...")
        preview_image(query_image, "query_image", args["in_terminal"])
    elif choice == "view_best_match":
        print("Displaying Best Match Image...")
        preview_image(best_match["image_data"],
                      "best_match_image", args["in_terminal"])
    elif choice == "view_unmatched":
        print("Displaying Unfiltered Best Match Image...")
        preview_image(unmatched, "unfiltered_best_match_image",
                      args["in_terminal"])
    elif choice == "use_pretrained_model":
        print("Using Pretrained Face Recognition Model...")
        query_img_dl = face_recognition.load_image_file(args["query_path"])
        dataset_dl = load_gallery_images(args["data_dir"])

        deep_learning_best_match, similarity_score_dl = get_best_match_using_face_recognition(
            query_img_dl, dataset_dl)
        print("\nBest Match Using Pretrained Face Recognition Model:")
        print(f"Subject ID: {deep_learning_best_match['subject_id']}")
        print(f"Illumination: {deep_learning_best_match['illumination']}")
        print(f"Similarity Score: {similarity_score_dl:.4f}")
        preview_image(
            deep_learning_best_match["image_data"], "dl_best_match", args["in_terminal"])
    elif choice == "visualize_similarities":
        try:
            top_n = int(
                input("Enter the number of top matches to visualize (default: 25): ") or 25)
        except ValueError:
            print("Invalid input. Using default value of 25.")
            top_n = 25
        print("Visualizing Similarities...")
        visualize_similarities(
            query_image, dataset, filter_method, similarity_method, alpha_grid, mu_grid, density, query_index, top_n=top_n
        )
    elif choice == "exit":
        print("Exiting program.")
        return False
    return True


def cli_main():
    # Collect interactive inputs
    args = get_cli_inputs()
    args["grid_size"] = int(args["grid_size"])
    args["iterations"] = int(args["iterations"])

    # Validate input paths
    validate_file_path(args["query_path"], "query image")
    validate_file_path(args["data_dir"], "data directory")

    # Load query image
    query_image = load_image(args["query_path"])
    print("Query image loaded successfully.")

    # Display query and sample images
    preview_image(query_image, "query_image", args["in_terminal"])

    # Load gallery dataset
    dataset = load_gallery_images(args["data_dir"])
    print(f"Loaded {len(dataset)} gallery images.")

    print("\nSample Gallery Images:")
    for i, sample in enumerate(dataset[:3]):
        print(f"Sample Image {
              i + 1} - Subject ID: {sample['subject_id']}, Illumination: {sample['illumination']}")
        preview_image(sample["image_data"], in_terminal=args["in_terminal"])

    # Get similarity and filter methods
    similarity_method = mutual_subspace_method if args[
        "method"] == "mutual_subspace" else cosine_similarity
    filter_method = globals()[args["filter"]]

    # Filter query image
    filtered_query_image = filter_method(query_image)

    print("Preview of Filtered Query Image:")
    preview_image(filtered_query_image,
                  "filtered_query_image", args["in_terminal"])

    print("Starting Adaptive Estimation...")
    # Estimate or load joint probability density
    alpha_grid, mu_grid, density = estimate_or_load_density(
        dataset, filter_method, similarity_method, args)

    # Match query image
    query_index = -1  # Placeholder for query index
    print("Matching Query Image...")
    best_match, similarity_score, alpha, unmatched = match_faces(
        query_image, dataset, filter_method, similarity_method, alpha_grid, mu_grid, density, query_index)

    print("\nFace Matching Results:")
    print(f"Best Match Similarity Score: {similarity_score:.4f}")
    print(f"Optimal Alpha: {alpha:.4f}")
    print(f"Best Match Subject: {best_match['subject_id']}, Illumination: {
          best_match['illumination']}")

    # Main menu loop
    while True:
        user_choice = cli_menu()
        if not handle_user_choice(user_choice, query_image, best_match, unmatched, dataset, args, alpha_grid, mu_grid, density, query_index, similarity_method, filter_method):
            break


if __name__ == "__main__":
    cli_main()

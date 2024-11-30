import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt

from adaptive_similarity import *
from data_utils import *
from visualization import *
from similarity import *
from filters import *


def preview_image(image):
    """Display image using matplotlib."""
    plt.imshow(image, cmap="gray")
    plt.axis("off")
    plt.show()


def cli_main():
    parser = argparse.ArgumentParser(description="Face Matching CLI Tool")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Path to the directory containing the gallery images")
    parser.add_argument("--query-path", type=str,
                        required=True, help="Path to the query image")
    parser.add_argument("--method", type=str, default="cosine",
                        choices=["cosine", "mutual_subspace"], help="Similarity method to use")
    parser.add_argument("--filter", type=str, default="homomorphic_filter", choices=[
                        "self_quotient", "gaussian_highpass", "laplacian_of_gaussian", "homomorphic_filter", "edge_map", "cannys_edge_detector"], help="Filtering method to use")
    parser.add_argument("--inference",type=bool,default=True,choices=[True,False],help="Choose whether to run the model or inference")

    args = parser.parse_args()

    similarity_method = mutual_subspace_method if args.method == "mutual_subspace" else cosine_similarity
    filter_method = self_quotient_image if args.filter == "self_quotient" else globals()[
        args.filter]

    # Load query image
    if not os.path.exists(args.query_path):
        print(f"Error: Query image '{args.query_path}' not found.")
        return
    query_image = load_image(args.query_path)
    print("Query image loaded successfully.")
    directory = args.data_dir
    for idx, filename in enumerate(os.listdir(directory)):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            current_image = load_image(filepath)
            # Compare with the query image
            if np.array_equal(query_image, current_image):
                image_idx = idx
                break

    # Display query image
    preview_image(query_image)

    # Load dataset
    dataset = []
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' not found.")
        return
    for file in glob.glob(os.path.join(args.data_dir, "*.jpg")):
        subject_id = extract_subject_id(file)
        illumination = extract_subject_lighting(file)
        image_data = load_image(file)
        dataset.append({
            "subject_id": subject_id,
            "illumination": illumination,
            "image_data": image_data,
            "query_name": os.path.basename(file)
        })
    print(f"Loaded {len(dataset)} gallery images.")

    # Display first 3 gallery images
    for i in range(min(3, len(dataset))):
        print(f"Subject ID: {dataset[i]['subject_id']}")
        print(f"Illumination :{dataset[i]['illumination']}")
        preview_image(dataset[i]['image_data'])

    # Estimate joint probability density 
    
    if args.inference == True:
        alpha_grid, mu_grid, density = load_from_files()
    else:
        alpha_grid, mu_grid, density = estimate_joint_probability_persons(
            dataset,
            self_quotient_image,
            similarity_method,  
            alpha_grid_size=100,
            mu_grid_size=100,
            n_iter=500
        )

    # plot the alpha mu density landscape
    plot_density(alpha_grid, mu_grid, density, epoch=500)

    # Match query image
    query_index = -1  # Set query index for comparison, if needed
    best_match, similarity_score, alpha, unmatched = match_faces(
        query_image,
        dataset,
        filter_method,
        similarity_method,
        alpha_grid,
        mu_grid,
        density,
        query_index
    )

    # Display results
    while True:
        print("\nFace Matching Results:")
        print(f"Best Match Similarity Score: {similarity_score:.4f}")
        print(f"Optimal Alpha: {alpha:.4f}")
        print(f"Best Match Subject: {best_match['subject_id']}, Illumination: {
              best_match['illumination']}")
        print("1. View query image")
        print("2. View best match image")
        print("3. View unmatched image")
        print("4. View best match using pretrained face recognition model")
        print("5. View image similarities with dataset")
        print("6. Exit")
        choice = input("Enter your choice (1-4): ")

        if choice == "1":
            print("Query Image:")
            preview_image(query_image)
        elif choice == "2":
            print("Best Match Image:")
            preview_image(best_match["image_data"])
        elif choice == "3":
            print("Unfiltered Best Match Image:")
            preview_image(unmatched)
        elif choice == "4":
            query_img_dl = face_recognition.load_image_file(args.query_path)
            dataset_dl = []
            for file in glob.glob(os.path.join(args.data_dir, "*.jpg")):
                image_data = face_recognition.load_image_file(file)
                dataset_dl.append({
                    "subject_id": extract_subject_id(file),
                    "illumination": extract_subject_lighting(file),
                    "image_data": image_data,
                    "query_name": os.path.basename(file)
                })

            # get the best match image using face recognition model
            deep_learning_best_match, similarity_score = get_best_match_using_face_recognition(
                query_img_dl, dataset_dl)
            print("Best Match Image using Face Recognition Model:")
            print(f"Subject ID: {deep_learning_best_match['subject_id']}")
            print(f"Illumination: {deep_learning_best_match['illumination']}")
            print(f"Similarity Score: {similarity_score:.4f}")

            preview_image(deep_learning_best_match["image_data"])

        elif choice == "5":
            try:
                top_n = int(
                    input("Enter the number of top matches to visualize: "))
            except ValueError:
                print("Invalid input. Using default value of 25.")
                top_n = 25
                
            print("Visualizing Similarities...")
            visualize_similarities(
                query_image, dataset, filter_method, similarity_method, alpha_grid, mu_grid, density, query_index, top_n=top_n
            )
        elif choice == "6":
            print("Exiting program.")
            break
        else:
            print("Invalid choice. Please select a valid option.")


if __name__ == "__main__":
    cli_main()

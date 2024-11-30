import os
import glob
import cv2
import numpy as np


def load_image(filename):
    """Load image from the file."""
    return cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

def extract_subject_id(filename):
    basename = os.path.basename(filename)  # Get file name without directory
    return int(basename.split('_')[0].split('.')[0].replace("subject","")) if basename.find("_") else int(basename.split('.')[0].replace("subject","")) # Assuming the subject ID is before the first underscore

def extract_subject_lighting(filename):
    basename = os.path.basename(filename)
    if "centerlight" in filename:
        return "c"
    if "leftlight" in filename:
        return "l"
    if "rightlight" in filename:
        return "r"
    else:
        return "n"

def load_dataset(directory):
    """Load dataset with metadata."""
    dataset = []
    for file in glob.glob(os.path.join(directory, "*.jpg")):
        dataset.append({
            "subject_id": extract_subject_id(file),
            "illumination": extract_subject_lighting(file),
            "image_data": load_image(file),
            "query_name": os.path.basename(file)
        })
    return dataset

def save_to_files(alpha_grid, mu_grid, density, output_dir="output"):
    """
    Save alpha grid, mu grid, and density to files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save using numpy's save functionality
    np.save(os.path.join(output_dir, "alpha_grid.npy"), alpha_grid)
    np.save(os.path.join(output_dir, "mu_grid.npy"), mu_grid)
    np.save(os.path.join(output_dir, "density.npy"), density)
    print(f"Data saved in directory: {output_dir}")

def load_from_files(output_dir="output"):
    """
    Load alpha grid, mu grid, and density from files.
    """
    try:
        alpha_grid = np.load(os.path.join(output_dir, "alpha_grid.npy"))
        mu_grid = np.load(os.path.join(output_dir, "mu_grid.npy"))
        density = np.load(os.path.join(output_dir, "density.npy"))
        print(f"Data loaded from directory: {output_dir}")
        return alpha_grid, mu_grid, density
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None, None, None
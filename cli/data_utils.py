import os
import glob
import cv2


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

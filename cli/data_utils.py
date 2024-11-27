import cv2
import os

def load_image(filepath):
    return cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

def load_dataset(folder_path):
    images = []
    for file in os.listdir(folder_path):
        if file.endswith('.jpg'):
            images.append(load_image(os.path.join(folder_path, file)))
    return images

def load_paths(folder_path):
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.jpg')]
import numpy as np
from scipy.ndimage import gaussian_filter, laplace, sobel

def gaussian_highpass(image, sigma=1.5):
    lowpass = gaussian_filter(image, sigma=sigma)
    return image - lowpass

def self_quotient_image(image, sigma=1.5):
    highpass = gaussian_highpass(image, sigma=sigma)
    local_mean = gaussian_filter(highpass, sigma=sigma)
    return np.divide(highpass, local_mean + 1e-5)  # Avoid division by zero

def laplacian_of_gaussian(image, sigma=3.0):
    return laplace(gaussian_filter(image, sigma=sigma))

def gradient_filters(image):
    grad_x = sobel(image, axis=0)
    grad_y = sobel(image, axis=1)
    return grad_x, grad_y

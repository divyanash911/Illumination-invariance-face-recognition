import numpy as np
import cv2

def gaussian_highpass(image, sigma=1.5):
    lowpass = cv2.GaussianBlur(image, (0, 0), sigma)
    return image - lowpass

def self_quotient_image(image, sigma=1.5):
    highpass = gaussian_highpass(image, sigma=sigma)
    local_mean = cv2.GaussianBlur(image, (0, 0), sigma)
    return np.divide(highpass, local_mean + 1e-5)  # Avoid division by zero

def laplacian_of_gaussian(image, sigma=3.0):
    return cv2.Laplacian(image, cv2.CV_64F, ksize=int(2 * round(3 * sigma) + 1))

def gradient_filters(image):
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    return grad_x, grad_y

def homomorphic_filter(image, sigma=1.0, alpha=0.5, beta=1.0):
    rows, cols = image.shape
    center_row, center_col = rows // 2, cols // 2

    img_log = np.log(np.float64(image), dtype=np.float64)
    img_fft = np.fft.fft2(img_log, axes=(0, 1))

    img_fft_shift = np.fft.fftshift(img_fft)

    # Create a Gaussian high-pass filter in the frequency domain (centered)
    rad = 13
    mask = np.zeros_like(image, dtype=np.float64)
    cy = mask.shape[0] // 2
    cx = mask.shape[1] // 2
    cv2.circle(mask, (cx, cy), rad, 1, -1)
    mask = 1 - mask

    # anti-aliasing
    mask = cv2.GaussianBlur(mask, (47, 47), 0)

    # Apply the mask to the image
    img_fft_filtered = np.multiply(img_fft_shift, mask)

    # Inverse FFT
    img_ifft_shift = np.fft.ifftshift(img_fft_filtered)

    img_ifft = np.fft.ifft2(img_ifft_shift, axes=(0, 1))

    img_filtered = np.exp(np.abs(img_ifft), dtype=np.float64)

    img_homomorphic = cv2.normalize(img_filtered, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return img_homomorphic

def edge_map(image):
    grad_x, grad_y = gradient_filters(image)
    return np.sqrt(grad_x ** 2 + grad_y ** 2)

def cannys_edge_detector(image, low_threshold=50, high_threshold=150):
    return cv2.Canny(image, low_threshold, high_threshold)
import cv2
import numpy as np
from skimage import morphology, filters, measure
from scipy import ndimage
import matplotlib.pyplot as plt

def enhanced_vessel_segmentation(image_path):
    """
    Improved classical vessel segmentation combining:
    - Frangi vesselness filter (detects tubular structures)
    - Adaptive thresholding
    - Morphological refinement
    """
    
    # 1. Load and preprocess
    img = cv2.imread(image_path)
    green = img[:, :, 1]  # Green channel has best vessel contrast
    
    # 2. CLAHE for local contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(green)
    
    # 3. FRANGI FILTER (Key Improvement!)
    # Detects vessel-like structures using Hessian matrix
    from skimage.filters import frangi
    vessels = frangi(
        enhanced,
        sigmas=range(1, 8, 2),  # detect vessels of different widths
        black_ridges=False,      # vessels are bright
        alpha=0.5,
        beta=0.5,
        gamma=15
    )
    
    # Normalize to [0, 255]
    vessels = (vessels * 255).astype(np.uint8)
    
    # 4. Adaptive Thresholding (better than global threshold)
    binary = cv2.adaptiveThreshold(
        vessels, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=15,
        C=-2  # adjust sensitivity
    )
    
    # 5. Morphological cleanup
    # Remove small noise
    kernel_small = morphology.disk(2)
    binary = morphology.opening(binary, kernel_small)
    
    # Fill small gaps in vessels
    kernel_close = morphology.disk(3)
    binary = morphology.closing(binary, kernel_close)
    
    # 6. Connected component analysis - remove tiny artifacts
    labeled = measure.label(binary)
    props = measure.regionprops(labeled)
    
    # Keep only significant components
    min_area = 50
    cleaned = np.zeros_like(binary)
    for prop in props:
        if prop.area >= min_area:
            cleaned[labeled == prop.label] = 255
    
    # 7. Skeletonization (optional - gives thin vessel centerlines)
    skeleton = morphology.skeletonize(cleaned > 0).astype(np.uint8) * 255
    
    return {
        'original': green,
        'enhanced': enhanced,
        'frangi': vessels,
        'binary': cleaned,
        'skeleton': skeleton
    }


def matched_filter_segmentation(image_path):
    """
    Alternative: Matched Filter approach
    Uses oriented Gaussian kernels to detect vessels
    """
    img = cv2.imread(image_path)
    green = img[:, :, 1]
    
    # CLAHE preprocessing
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(green)
    
    # Create matched filter kernels at different orientations
    def create_matched_filter(length=9, sigma=2):
        x = np.arange(-length//2, length//2 + 1)
        kernel = np.exp(-x**2 / (2*sigma**2))
        kernel = kernel / kernel.sum()
        return kernel.reshape(-1, 1)
    
    # Apply at 12 orientations
    response = np.zeros_like(enhanced, dtype=float)
    kernel = create_matched_filter()
    
    for angle in range(0, 180, 15):
        # Rotate kernel
        M = cv2.getRotationMatrix2D((kernel.shape[0]//2, kernel.shape[1]//2), angle, 1.0)
        rotated = cv2.warpAffine(kernel, M, kernel.T.shape)
        
        # Convolve
        filtered = cv2.filter2D(enhanced, -1, rotated)
        response = np.maximum(response, filtered)
    
    # Threshold
    response = (response - response.min()) / (response.max() - response.min())
    binary = (response > 0.15).astype(np.uint8) * 255
    
    # Cleanup
    binary = morphology.remove_small_objects(binary > 0, min_size=50)
    binary = morphology.remove_small_holes(binary, area_threshold=50)
    
    return (binary * 255).astype(np.uint8)


# Visualization
def visualize_results(results):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0,0].imshow(results['original'], cmap='gray')
    axes[0,0].set_title('Original (Green Channel)')
    axes[0,0].axis('off')
    
    axes[0,1].imshow(results['enhanced'], cmap='gray')
    axes[0,1].set_title('CLAHE Enhanced')
    axes[0,1].axis('off')
    
    axes[0,2].imshow(results['frangi'], cmap='hot')
    axes[0,2].set_title('Frangi Vesselness')
    axes[0,2].axis('off')
    
    axes[1,0].imshow(results['binary'], cmap='gray')
    axes[1,0].set_title('Binary Segmentation')
    axes[1,0].axis('off')
    
    axes[1,1].imshow(results['skeleton'], cmap='gray')
    axes[1,1].set_title('Skeleton')
    axes[1,1].axis('off')
    
    # Overlay
    overlay = cv2.cvtColor(results['original'], cv2.COLOR_GRAY2RGB)
    overlay[results['binary'] > 0] = [255, 0, 0]
    axes[1,2].imshow(overlay)
    axes[1,2].set_title('Overlay')
    axes[1,2].axis('off')
    
    plt.tight_layout()
    plt.show()


# Example usage:
results = matched_filter_segmentation('image.jpg')
plt.figure(figsize=(15, 6))
plt.imshow(results, cmap='gray')
plt.tight_layout()
plt.show()
import cv2
import numpy as np
from scipy import ndimage
from sklearn.cluster import KMeans

class Segmenter:
    def __init__(self, method='threshold'):
        """
        Initialize the segmenter with the desired method
        
        Args:
            method: Segmentation method ('threshold', 'edge', 'region', 'kmeans')
        """
        self.method = method
    
    def segment(self, image):
        """
        Segment the image using the selected method
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            Segmented image or mask
        """
        if isinstance(image, np.ndarray) and image.ndim == 3 and image.shape[2] == 3:
            # Convert to BGR if it's in RGB format
            if image.dtype == np.float32:
                # Denormalize if the image has been normalized
                image = (image * 255).astype(np.uint8)
        
        # Apply different segmentation methods
        if self.method == 'threshold':
            return self._threshold_segmentation(image)
        elif self.method == 'edge':
            return self._edge_based_segmentation(image)
        elif self.method == 'region':
            return self._region_growing(image)
        elif self.method == 'kmeans':
            return self._kmeans_segmentation(image)
        else:
            raise ValueError(f"Unknown segmentation method: {self.method}")
    
    def _threshold_segmentation(self, image):
        """Custom threshold-based segmentation"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Apply Otsu's thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Apply morphological operations to clean up
        kernel = np.ones((5, 5), np.uint8)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Find sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
        sure_fg = sure_fg.astype(np.uint8)
        
        # Finding unknown region
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Apply watershed algorithm
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        if len(image.shape) == 3:
            markers = cv2.watershed(image, markers)
        else:
            # Convert to BGR for watershed
            colored = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            markers = cv2.watershed(colored, markers)
            
        # Create mask where markers > 1
        mask = np.zeros_like(gray, dtype=np.uint8)
        mask[markers > 1] = 255
        
        return mask
    
    def _edge_based_segmentation(self, image):
        """Custom edge-based segmentation"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Dilate edges to close gaps
        kernel = np.ones((3, 3), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create mask for large contours
        mask = np.zeros_like(gray)
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Filter small contours
                cv2.drawContours(mask, [contour], -1, 255, -1)
        
        # Apply morphological operations to clean up
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
    
    def _region_growing(self, image):
        """Custom region growing segmentation"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Blur the image to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Get initial seeds - we'll use points from a grid
        height, width = blurred.shape
        seeds = []
        step = 20  # Grid spacing
        for i in range(step, height, step):
            for j in range(step, width, step):
                seeds.append((i, j))
        
        # Create output mask
        mask = np.zeros_like(blurred, dtype=np.uint8)
        
        # Define threshold for region growing
        threshold = 10
        
        # Process each seed
        for seed_y, seed_x in seeds:
            if mask[seed_y, seed_x] == 0:  # Skip if already processed
                # Get the seed intensity
                seed_value = blurred[seed_y, seed_x]
                
                # Create a queue for breadth-first search
                queue = [(seed_y, seed_x)]
                region_points = []
                
                while queue:
                    y, x = queue.pop(0)
                    
                    # Check if already processed or out of range
                    if (y < 0 or y >= height or x < 0 or x >= width or 
                        mask[y, x] != 0):
                        continue
                    
                    # Check intensity difference
                    if abs(int(blurred[y, x]) - int(seed_value)) <= threshold:
                        mask[y, x] = 255
                        region_points.append((y, x))
                        
                        # Add neighbors to queue
                        queue.append((y+1, x))
                        queue.append((y-1, x))
                        queue.append((y, x+1))
                        queue.append((y, x-1))
        
        # Apply morphological operations to clean up
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
    
    def _kmeans_segmentation(self, image):
        """Custom K-means segmentation"""
        # Reshape the image to a 2D array of pixels
        if len(image.shape) == 3:
            pixel_values = image.reshape((-1, 3))
        else:
            pixel_values = image.reshape((-1, 1))
            
        # Convert to float32
        pixel_values = np.float32(pixel_values)
        
        # Define criteria and apply kmeans
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        k = 3  # Number of clusters
        _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Convert back to 8-bit values
        centers = np.uint8(centers)
        
        # Flatten the labels array
        labels = labels.flatten()
        
        # Create mask for the most significant cluster
        # We'll find the cluster with the brightest center for simplicity
        if len(image.shape) == 3:
            # For RGB, use mean of center values
            brightest_cluster = np.argmax([np.mean(center) for center in centers])
        else:
            brightest_cluster = np.argmax(centers)
            
        mask = np.zeros(labels.shape, dtype=np.uint8)
        mask[labels == brightest_cluster] = 255
        
        # Reshape back to the original image shape
        if len(image.shape) == 3:
            mask = mask.reshape(image.shape[:2])
        else:
            mask = mask.reshape(image.shape)
            
        # Apply morphological operations to clean up
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
    
    def apply_mask(self, image, mask):
        """
        Apply the segmentation mask to the original image
        
        Args:
            image: Original image
            mask: Binary mask from segmentation
            
        Returns:
            Masked image with only the regions of interest
        """
        # Ensure mask is binary
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # Create 3-channel mask if needed
        if len(image.shape) == 3 and len(binary_mask.shape) == 2:
            binary_mask = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
        
        # Apply mask
        masked_image = cv2.bitwise_and(image, binary_mask)
        
        return masked_image
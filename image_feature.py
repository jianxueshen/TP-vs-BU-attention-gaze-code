import numpy as np
import gzip
import os
from PIL import Image
import cv2
import json
from scipy.ndimage import gaussian_filter
from skimage.feature import local_binary_pattern

import saliencymetr as sm
import matplotlib.pyplot as plt
from line_profiler import LineProfiler
import glob
import itti

def load_data(semantic_path, freeview_fixations_path, search_fixations_path, image_base_path):
    # Function to load JSON data
    def load_json(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)

    # Function to load .npy.gz file
    def load_npy_gz(file_path):
        with gzip.open(file_path, 'rb') as f:
            return np.load(f)

    # Function to get data by base name
    def get_data_by_name(data_list, base_name):
        return [entry for entry in data_list if entry['name'] == f'{base_name}.jpg']


    # Load data
    semantic = load_npy_gz(semantic_path)
    freeview_fixations = load_json(freeview_fixations_path)
    search_fixations = load_json(search_fixations_path)

    # Extract base name and corresponding data
    base_name = os.path.basename(semantic_path).replace('.npy.gz', '')
    freeview_data = get_data_by_name(freeview_fixations, base_name)
    search_data = get_data_by_name(search_fixations, base_name)

    # Extract task and image path
    task = freeview_data[0]['task'] if freeview_data and 'task' in freeview_data[0] else None
    image_path = os.path.join(image_base_path, f'{task}', f'{base_name}.jpg')
    image = Image.open(image_path)
    image = np.array(image) 

    # Print data info
    #print("fv data num:", len(freeview_data))
    #print("search data num:", len(search_data))
    #print("Segmentation Map size:", semantic.shape)
    #print("Image shape:", image.shape)

    return semantic, freeview_data, search_data, image, base_name


def generate_heatmap(image, sample, sigma=10):
    """
    Generate a heatmap based on fixation points and durations.
    Parameters:
        image (numpy.ndarray): The image array.
        sample (dict): A dictionary containing fixation data with keys 'X', 'Y', and 'T'.
        sigma (int): The standard deviation for Gaussian smoothing.
    Returns:
        numpy.ndarray: The generated heatmap.
    """
    # Create an empty heatmap with the same dimensions as the image
    heatmap = np.zeros((image.shape[0], image.shape[1]))

    # Populate the heatmap with fixation durations (T values) at corresponding (X, Y) positions
    for x, y, t in zip(sample['X'], sample['Y'], sample['T']):
        if 0 <= int(y) < heatmap.shape[0] and 0 <= int(x) < heatmap.shape[1]:
            heatmap[int(y), int(x)] += t

    # Normalize the heatmap to range [0, 1]
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-6)

    # Apply Gaussian filter to smooth the heatmap
    heatmap = gaussian_filter(heatmap, sigma=sigma)
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    #print("Heatmap shape:", heatmap.shape)
    return heatmap

# Function to draw heatmap
def draw_heatmap(ax, image, heatmap, title):
    # Display the image with alpha blending
    ax.imshow(image, alpha=0.5)

    # Overlay the heatmap with pseudo-color
    ax.imshow(heatmap, cmap='jet', alpha=0.6, extent=[0, image.shape[1], image.shape[0], 0])

    # Set title and remove axis
    ax.set_title(title)
    ax.axis('off')

# Function to draw fixations
def draw_fixations(ax, fixation_data, color, label):
    """
    Draw fixation points and trajectories on an image.
    Parameters:
        ax (matplotlib.axes.Axes): The axes to draw on.
        fixation_data (list): A list of dictionaries containing fixation data with keys 'X', 'Y', and 'T'.
        color (str): The color to use for the fixation points and trajectories.
        label (str): The label for the fixation data.
    """
    # Extract X, Y, and T values
    x_coords = fixation_data['X']
    y_coords = fixation_data['Y']
    durations = fixation_data['T']
    # Normalize durations for circle sizes
    max_duration = max(durations) if durations else 1
    normalized_sizes = [int((t / max_duration) * 20) + 5 for t in durations]
    # Draw circles and annotate with fixation numbers
    for i, (x, y, size) in enumerate(zip(x_coords, y_coords, normalized_sizes)):
        if 0 <= x < ax.get_xlim()[1] and 0 <= y < ax.get_ylim()[0]:
            circle = plt.Circle((x, y), size, color=color, alpha=0.5, fill=True, label=label if i == 0 else None)
            ax.add_patch(circle)
            ax.text(x, y, str(i + 1), color='white', fontsize=8, ha='center', va='center')
    # Draw lines connecting fixation points
    ax.plot(x_coords, y_coords, color=color, linewidth=1, alpha=0.7, label=None)
    # Set axis limits and remove axis
    ax.axis('off')

# Function to extract brightness map
def extract_gray_map(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #print("Grayscale image shape:", grayscale_image.shape)
    return grayscale_image

# Function to extract brightness maps at three levels
def extract_brightness_map(image):
    """
    Extract brightness maps at three levels from an image by converting it to HSV mode.
    Parameters:
        image (numpy.ndarray): Input image in RGB format.
    Returns:
        dict: A dictionary containing brightness maps at low, medium, and high levels.
    """
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # Extract the V channel (brightness)
    brightness = hsv_image[:, :, 2]
    # Normalize the brightness to range [0, 1]
    normalized_brightness = brightness / 255.0
    # Define thresholds for low, medium, and high brightness levels
    low_threshold = 0.33
    high_threshold = 0.66
    # Create masks for each brightness level
    low_brightness = (normalized_brightness <= low_threshold).astype(np.uint8) * brightness
    medium_brightness = ((normalized_brightness > low_threshold) & (normalized_brightness <= high_threshold)).astype(np.uint8) * brightness
    high_brightness = (normalized_brightness > high_threshold).astype(np.uint8) * brightness
    #print("Low brightness map shape:", low_brightness.shape)
    #print("Medium brightness map shape:", medium_brightness.shape)
    #print("High brightness map shape:", high_brightness.shape)
    return {
        'low_brightness': low_brightness,
        'mid_brightness': medium_brightness,
        'high_brightness': high_brightness
    }

# Function to extract color channels
def extract_color_channels(image):
    """
    Extract individual color channels (Red, Green, Blue) from an RGB image.
    Parameters:
        image (numpy.ndarray): Input image in RGB format.
    Returns:
        dict: A dictionary containing red, green, and blue channels.
    """
    red_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    blue_channel = image[:, :, 2]
    #print("Red channel shape:", red_channel.shape)
    #print("Green channel shape:", green_channel.shape)
    #print("Blue channel shape:", blue_channel.shape)
    return {
        'red_channel': red_channel,
        'green_channel': green_channel,
        'blue_channel': blue_channel
    }

# Function to extract dominant color features using k-means clustering in HSV color space
def extract_dominant_color_features(image):
    """
    Extract dominant color features using k-means clustering in HSV color space.
    Parameters:
        image (numpy.ndarray): Input image in RGB format.
    Returns:
        dict: A dictionary containing the main RGB color region, contrast RGB color region, and dominant RGB colors.
    """
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h_channel = hsv_image[:, :, 0]  # Extract the H (hue) channel

    # Reshape the H channel to a 2D array of pixels
    pixels = h_channel.reshape((-1, 1)).astype(np.float32)

    # Define criteria and apply k-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 7  # Number of clusters
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert centers to int8 and reshape labels to match the image
    centers = np.int16(centers)
    labels = labels.flatten()

    # Print the H values of the cluster centers
    #print("Cluster centers (H values):", centers.flatten())

    # Count the number of pixels in each cluster
    cluster_sizes = np.bincount(labels)

    # Find the index of the largest cluster (main hue region)
    main_cluster_index = np.argmax(cluster_sizes)
    main_hue = centers[main_cluster_index]

    # Create masks for each cluster
    masks = []
    for i in range(k):
        mask = (labels == i).reshape(h_channel.shape)
        masks.append(mask)

    # Define the main hue region
    main_hue_region = masks[main_cluster_index]

    # Calculate the absolute difference of other cluster centers to the main hue
    centers_c = np.where(centers > 90, centers - 180, centers)
    main_hue_c = main_hue - 180 if main_hue > 90 else main_hue
    # Calculate the absolute difference of other cluster centers to the main hue
    distances = np.abs(centers_c - main_hue_c) 
    distances = np.where(distances < 30, 0, distances)
    sizes_rate = cluster_sizes.reshape(7,1) / np.sum(cluster_sizes)

    # Use the size of each cluster as a weight for the distances
    weighted_distances = distances * sizes_rate

    # Find the cluster with the largest distance to the main hue (contrast hue region)
    contrast_cluster_index = np.argmax(weighted_distances)
    contrast_hue_region = masks[contrast_cluster_index]
    contrast_hue = centers[contrast_cluster_index]

    # Extract the original image regions corresponding to the main and contrast hues
    main_hue_image_region = cv2.bitwise_and(image, image, mask=main_hue_region.astype(np.uint8))
    contrast_hue_image_region = cv2.bitwise_and(image, image, mask=contrast_hue_region.astype(np.uint8))

    # Calculate the average RGB color for the main and contrast regions
    main_color = cv2.mean(image, mask=main_hue_region.astype(np.uint8))[:3]
    contrast_color = cv2.mean(image, mask=contrast_hue_region.astype(np.uint8))[:3]

    # Create an overlay image to display the k-means clustering results
    overlay_image = np.zeros_like(image)

    # Assign each pixel in the overlay image the average RGB color of its cluster
    for i in range(k):
        cluster_mask = (labels.reshape(h_channel.shape) == i)
        cluster_pixels = image[cluster_mask]
        if len(cluster_pixels) > 0:
            average_color = np.mean(cluster_pixels, axis=0).astype(np.uint8)
            overlay_image[cluster_mask] = average_color

    # Print the dominant RGB colors
    #print("Main color (RGB) and hue :", list(map(int, main_color)), main_hue)
    #print("Contrast color (RGB) and hue:", list(map(int, contrast_color)), contrast_hue)
    #print("main color region shape:", main_hue_image_region.shape)  
    #print("contrast color region shape:", contrast_hue_image_region.shape)
    #print("overlay image shape:", overlay_image.shape)                        

    return {
        'kmeans_result': overlay_image,
        'main_color_region': main_hue_image_region,
        'contrast_color_region': contrast_hue_image_region,
        'main_color': list(map(int, main_color)),
        'contrast_color': list(map(int, contrast_color)),
        'kmeans_result_gray' : cv2.cvtColor(overlay_image, cv2.COLOR_RGB2GRAY),
        'main_color_region_gray': cv2.cvtColor(main_hue_image_region, cv2.COLOR_RGB2GRAY),
        'contrast_color_region_gray': cv2.cvtColor(contrast_hue_image_region, cv2.COLOR_RGB2GRAY)
    }

# Function to extract orientation features
def extract_orientation_features(brightness_map):
    # Compute gradients in both forward and backward directions
    forward_gradient_x = np.gradient(brightness_map, axis=1)
    forward_gradient_y = np.gradient(brightness_map, axis=0)
    backward_gradient_x = -np.gradient(brightness_map[:, ::-1], axis=1)[:, ::-1]
    backward_gradient_y = -np.gradient(brightness_map[::-1, :], axis=0)[::-1, :]
    # Combine forward and backward gradients
    combined_gradient_x = (forward_gradient_x + backward_gradient_x) / 2
    combined_gradient_y = (forward_gradient_y + backward_gradient_y) / 2
    # Normalize gradients to fit in the range [0, 255]
    horizontal_orientation_image = ((combined_gradient_x - np.min(combined_gradient_x)) / 
                                    (np.max(combined_gradient_x) - np.min(combined_gradient_x)) * 255).astype(np.uint8)
    vertical_orientation_image = ((combined_gradient_y - np.min(combined_gradient_y)) / 
                                  (np.max(combined_gradient_y) - np.min(combined_gradient_y)) * 255).astype(np.uint8)
    #print("Horizontal orientation image shape:", horizontal_orientation_image.shape)
    #print("Vertical orientation image shape:", vertical_orientation_image.shape)
    return {
        'horizontal_orientation_image': horizontal_orientation_image,
        'vertical_orientation_image': vertical_orientation_image
    }

# Function to extract texture features using Local Binary Patterns (LBP)
def extract_texture_features(brightness_map, radius=3, n_points=24):
    """
    Extract texture features using Local Binary Patterns (LBP).
    Parameters:
        brightness_map (numpy.ndarray): Grayscale image.
        radius (int): Radius of the circular LBP pattern.
        n_points (int): Number of points in the circular LBP pattern.
    Returns:
        numpy.ndarray: LBP image representing texture features.
    """
    # Compute the LBP image
    lbp_image = local_binary_pattern(brightness_map, n_points, radius, method='uniform')
    lbp_image = ((lbp_image - lbp_image.min()) / (lbp_image.max() - lbp_image.min()) * 255).astype(np.uint8)
    #print("LBP image shape:", lbp_image.shape)
    return lbp_image

# Function to extract all low-level features
def extract_low_level_features(image):
    gray_map = extract_gray_map(image)
    brightness_map = extract_brightness_map(image)
    color_channels = extract_color_channels(image)
    color_regions = extract_dominant_color_features(image)
    orientation_features = extract_orientation_features(gray_map)
    texture_features = extract_texture_features(gray_map)
    return {
        'gray_map': gray_map,
        'brightness_map': brightness_map,
        'color_channels': color_channels,
        'color_regions': color_regions,
        'orientation_features': orientation_features,
        'texture_features': texture_features
    }

# Function to extract shape features
def extract_shape_features(brightness_map):
    """
    Extract shape features from an image using edge detection and contour analysis.
    Parameters:
        image (numpy.ndarray): Input image.
    Returns:
        dict: A dictionary containing edge map and contours.
    """
    # Apply edge detection (Canny)
    edges = cv2.Canny(brightness_map, threshold1=100, threshold2=200)
    #print("Edge map shape:", edges.shape)
    # Find contours from the edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #print("Number of contours found:", len(contours))
    # Create a blank image to draw contours
    contour_image = np.zeros_like(brightness_map)
    cv2.drawContours(contour_image, contours, -1, (255), thickness=1)
    return {
        'edge_map': edges,
        'contours': contour_image,
        'contour_count': len(contours)
    }

# Function to extract spatial relationship features
def extract_spatial_relationship_features(brightness_map):
    """
    Extract spatial relationship features from an image using keypoint detection and descriptors.
    Parameters:
        image (numpy.ndarray): Input image.
    Returns:
        dict: A dictionary containing keypoints and descriptors.
    """
    # Use ORB (Oriented FAST and Rotated BRIEF) to detect keypoints and compute descriptors
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(brightness_map, None)

    # Create a blank grayscale image to represent spatial relationships
    keypoint_image = np.zeros_like(brightness_map)
    for keypoint in keypoints:
        x, y = int(keypoint.pt[0]), int(keypoint.pt[1])
        size = int(keypoint.size)  # Use the size of the keypoint
        angle = keypoint.angle if keypoint.angle != -1 else 0  # Use the angle of the keypoint
        response = keypoint.response  # Use the response of the keypoint
        octave = keypoint.octave  # Use the octave of the keypoint
        class_id = keypoint.class_id if hasattr(keypoint, 'class_id') else -1  # Use the class_id of the keypoint if available

        if 0 <= x < keypoint_image.shape[1] and 0 <= y < keypoint_image.shape[0]:
            # Draw a filled circle to represent the keypoint
            cv2.circle(keypoint_image, (x, y), size // 2, 255, thickness=1)
            # Draw a line to represent the orientation of the keypoint
            end_x = int(x + size * np.cos(np.deg2rad(angle)))
            end_y = int(y - size * np.sin(np.deg2rad(angle)))
            cv2.line(keypoint_image, (x, y), (end_x, end_y), 255, thickness=1)

        # Normalize the blurred image to range [0, 255]
        #keypoint_image = cv2.GaussianBlur(keypoint_image, (5, 5), 0)
        keypoint_image = cv2.normalize(keypoint_image, None, 0, 255, cv2.NORM_MINMAX)

    #print("Number of keypoints detected:", len(keypoints))
    return {
        'keypoints': keypoints,
        'descriptors': descriptors,
        'spatial_image': keypoint_image
    }


# Function to extract mid-level features
def extract_mid_level_features(brightness_map):
    """
    Extract mid-level features including shape and spatial relationship features.
    Parameters:
        brightness_map (numpy.ndarray): Grayscale image.
    Returns:
        dict: A dictionary containing shape and spatial relationship features.
    """
    # Extract shape features
    shape_features = extract_shape_features(brightness_map)
    # Extract spatial relationship features
    spatial_features = extract_spatial_relationship_features(brightness_map)

    return {
        'shape_features': shape_features,
        'spatial_features': spatial_features
    }

def extract_high_level_features(semantic):
    return {
        'semantic': semantic
    }

# Combine all low-level and mid-level features into a single dictionary
def combine_features(low_features, mid_features, high_features):
    """
    Combine low-level, mid-level, and high-level features into a single dictionary.

    Parameters:
        low_features (dict): Dictionary containing low-level features.
        mid_features (dict): Dictionary containing mid-level features.
        high_features (dict): Dictionary containing high-level features.

    Returns:
        dict: Combined dictionary of all features.
    """
    return {
        'gray_map': low_features['gray_map'],
        'low_brightness': low_features['brightness_map']['low_brightness'],
        'mid_brightness': low_features['brightness_map']['mid_brightness'],
        'high_brightness': low_features['brightness_map']['high_brightness'],
        'horizontal_orientation': low_features['orientation_features']['horizontal_orientation_image'],
        'vertical_orientation': low_features['orientation_features']['vertical_orientation_image'],
        'texture_features': low_features['texture_features'],
        'red_channel': low_features['color_channels']['red_channel'],
        'green_channel': low_features['color_channels']['green_channel'],
        'blue_channel': low_features['color_channels']['blue_channel'],
        'color_main_regions': low_features['color_regions']['main_color_region'],
        'color_contrast_regions': low_features['color_regions']['contrast_color_region'],
        'color_kmeans_regions': low_features['color_regions']['kmeans_result'],
        'contours': mid_features['shape_features']['contours'],
        'spatial_features': mid_features['spatial_features']['spatial_image'],
        'semantic_image': high_features['semantic']
    }

def compute_similarity_metrics(feature_image, heatmap, image, to_plot):
    results = {}
    img = {}
    results['KLdiv'], img['KLdiv'] = sm.KLdiv(feature_image, heatmap, to_plot)
    #results['EMD'] = sm.EMD(feature_image, heatmap, to_plot)
    results['NSS'], img['NSS'] = sm.NSS(feature_image, heatmap, to_plot)
    results['CC'], img['CC'] = sm.CC(feature_image, heatmap, to_plot)
    results['SIM'], img['SIM'] = sm.similarity(feature_image, heatmap, to_plot)
    results['InfoGain'], img['InfoGain'] = sm.InfoGain(feature_image, heatmap, to_plot)
    #results['AUV_judd'] = sm.visualize_AUC(feature_image, heatmap)
    return results, img
    
# Function to calculate similarities for a given heatmap and features
def calculate_similarities(heatmap, saliency_maps, image, to_plot):
    similarity_results = {}
    img_results = {}
    for feature_name, feature_image in saliency_maps.items():
        similarity_results[feature_name], img_results[feature_name] = compute_similarity_metrics(feature_image, heatmap, image, to_plot)
    return similarity_results, img_results

def main(file_path, freeview_fixations_path, search_fixations_path, image_base_path):
    semantic, freeview_data, search_data, image, base_name = load_data(file_path, freeview_fixations_path, search_fixations_path, image_base_path)

    low_features = extract_low_level_features(image)
    mid_features = extract_mid_level_features(low_features['gray_map'])
    high_features = extract_high_level_features(semantic)

    img_features = combine_features(low_features, mid_features, high_features)

    saliency_maps = {}
    for fe, fe_image in img_features.items():
        saliency_map = itti.itti(fe_image)
        saliency_maps[fe] = saliency_map

    freeview_heatmap = {}
    for sample in freeview_data:
        freeview_heatmap[f'{sample['subject']}'] = generate_heatmap(image, sample, sigma=10)
        
    search_heatmap = {}
    for sample in search_data:
        search_heatmap[f'{sample['subject']}'] = generate_heatmap(image, sample, sigma=10)
        
    freeview_similarity_results = {}
    for sub, heatmap in freeview_heatmap.items():
        value, img = calculate_similarities(heatmap, saliency_maps, image, to_plot=False)
        freeview_similarity_results[f'{sub}'] = value

    search_similarity_results = {}
    for sub, heatmap in search_heatmap.items():
        value, img = calculate_similarities(heatmap, saliency_maps, image, to_plot=False).items()
        search_similarity_results[f'{sub}'] = value

    # Combine freeview and search similarity results into a single dictionary
    combined_similarity_results = {
        'freeview_similarity_results': freeview_similarity_results,
        'search_similarity_results': search_similarity_results
    }
    # Save the combined results to a JSON file
    output_path = f'/home/oct/COCO_Search18-and-FV/hatdata/result/{base_name}.json'
    with open(output_path, 'w') as f:
        json.dump(combined_similarity_results, f, indent=4)
    print(f"Combined similarity results saved to {output_path}")

'''
if __name__ == "__main__":
    file_folder = '/home/oct/COCO_Search18-and-FV/hatdata/semantic_seq_full/segmentation_maps'
    freeview_fixations_path = '/home/oct/COCO_Search18-and-FV/hatdata/coco_freeview_fixations_512x320.json'
    search_fixations_path = '/home/oct/COCO_Search18-and-FV/hatdata/coco_search_fixations_512x320_on_target_allvalid.json'
    image_base_path = '/home/oct/COCO_Search18-and-FV/hatdata/images'

    #lp = LineProfiler()
    #lp.add_function(main)
    #lp.enable()
    # Get all files in the folder
    file_paths = glob.glob(os.path.join(file_folder, '*.npy.gz'))
    total_num = len(file_paths)
    # Process each file
    num = 1
    for file_path in file_paths:   
        main(file_path, freeview_fixations_path, search_fixations_path, image_base_path)
        print(f"Processed {file_path}")
        print(f"Processed {num}/{total_num}")
        num += 1
        

    #lp.disable()
    #lp.print_stats()
'''
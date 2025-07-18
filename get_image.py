import glob
import numpy as np
import gzip
import os
from PIL import Image
import cv2
import json
from scipy.ndimage import gaussian_filter
from skimage.feature import local_binary_pattern

import matplotlib.pyplot as plt
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


def load_data(semantic_path, freeview_fixations_path, search_fixations_path, image_base_path):
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

    freeview_sample = freeview_data[0]
    search_sample = search_data[0]

    # Print data info
    print("fv data num:", len(freeview_data))
    print("search data num:", len(search_data))
    print("Freeview Data:", freeview_sample)
    print("Search Data:", search_sample)
    print("Segmentation Map size:", semantic.shape)
    print("Image shape:", image.shape)

    return semantic, freeview_sample, search_sample, image

# Function to draw fixation points and trajectories
def draw_fixations(ax, sample, color, label_prefix):
    for i, (x, y, t) in enumerate(zip(sample['X'], sample['Y'], sample['T'])):
        circle = plt.Circle((x, y), radius=t/20, color=color, alpha=0.5)
        ax.add_patch(circle)
        ax.text(x, y, str(i + 1), color='white', fontsize=8, ha='center', va='center')
        if i > 0:
            ax.plot([sample['X'][i - 1], x], [sample['Y'][i - 1], y], color=color, linewidth=1)

# Function to generate heatmap
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

    print("Heatmap shape:", heatmap.shape)
    return heatmap

# Function to draw heatmap
def draw_heatmap(ax, image, heatmap, title):
    # Display the image with alpha blending
    ax.imshow(image, alpha=0.5)

    # Overlay the heatmap with pseudo-color
    ax.imshow(heatmap, cmap='jet', alpha=0.6, extent=[0, image.shape[1], image.shape[0], 0])

    # Set title and remove axis
    #ax.set_title(title)
    ax.axis('off')

file_path = '/home/oct/COCO_Search18-and-FV/hatdata/semantic_seq_full/segmentation_maps/000000066632.npy.gz'
freeview_fixations_path = '/home/oct/COCO_Search18-and-FV/hatdata/coco_freeview_fixations_512x320.json'
search_fixations_path = '/home/oct/COCO_Search18-and-FV/hatdata/coco_search_fixations_512x320_on_target_allvalid.json'
image_base_path = '/home/oct/COCO_Search18-and-FV/hatdata/images'

# Get the first 6 .npy.gz files from the directory
semantic_files = sorted(glob.glob('/home/oct/COCO_Search18-and-FV/hatdata/semantic_seq_full/segmentation_maps/*.npy.gz'))[:6]

# Initialize lists to store the loaded data
loaded_data = []

fig, axes = plt.subplots(3, len(semantic_files), figsize=(18, 9))
# Loop through the files and load the data
for idx, file_path in enumerate(semantic_files):
    semantic, freeview_sample, search_sample, image = load_data(file_path, freeview_fixations_path, search_fixations_path, image_base_path)

    # Plot the original image
    axes[0, idx].imshow(image)
    axes[0, idx].axis('off')

    # Draw fixation points and trajectories
    axes[1, idx].imshow(image)
    draw_fixations(axes[1, idx], freeview_sample, 'green', 'Freeview')
    axes[1, idx].axis('off')

    # Generate and draw heatmaps
    freeview_heatmap = generate_heatmap(image, freeview_sample, sigma=10)
    draw_heatmap(axes[2, idx], image, freeview_heatmap, "Heatmap")
    axes[2, idx].axis('off')

# Adjust layout and display the figure
plt.tight_layout()
plt.show()

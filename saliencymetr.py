import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.stats
import os
import pandas as pd
from skimage import img_as_float
from matplotlib import cm
from sklearn.metrics import roc_curve, auc

def CC(saliency_map1, saliency_map2, to_plot=True):
    """
    Compute the Pearson correlation coefficient between two saliency maps.
    
    Parameters:
        saliency_map1 (numpy.ndarray): First saliency map.
        saliency_map2 (numpy.ndarray): Second saliency map.
        
    Returns:
        float: Pearson's correlation coefficient.
    """
    if len(saliency_map1.shape) == 3:  # Check if the image has 3 channels
        saliency_map1 = cv2.cvtColor(saliency_map1, cv2.COLOR_BGR2GRAY)
    if len(saliency_map2.shape) == 3:  # Check if the image has 3 channels
        saliency_map2 = cv2.cvtColor(saliency_map2, cv2.COLOR_BGR2GRAY)
    # Resize saliency_map1 to match saliency_map2 size
    map1 = cv2.resize(saliency_map1, (saliency_map2.shape[1], saliency_map2.shape[0])).astype(np.float64)
    map2 = saliency_map2.astype(np.float64)
    
    # Normalize both maps
    map1 = (map1 - np.mean(map1)) / np.std(map1)
    map2 = (map2 - np.mean(map2)) / np.std(map2)
    
    # Calculate Pearson correlation coefficient
    score = np.corrcoef(map1.flatten(), map2.flatten())[0, 1]
    res_map_norm = None
    figtitle = 'Correlation'
    if to_plot:
        res_map = (map1 * map2) / np.sqrt(np.sum(map1**2) + np.sum(map2**2))
        res_map_norm = res_map / np.max(res_map)
        
        #plt.figure()
        #plt.imshow(res_map_norm, cmap='viridis')
        #plt.colorbar()
        #plt.title(figtitle, fontsize=14)
        #plt.axis('off')
        #plt.show()

    return score, res_map_norm

# Example usage:
# saliency_map1 = cv2.imread('path_to_saliency_map1.jpg', cv2.IMREAD_GRAYSCALE)
# saliency_map2 = cv2.imread('path_to_saliency_map2.jpg', cv2.IMREAD_GRAYSCALE)
# score = CC(saliency_map1, saliency_map2)
# print(f"Pearson Correlation Coefficient: {score:.4f}")


def similarity(saliency_map1, saliency_map2, to_plot=True):
    """
    Finds the similarity between two saliency maps when viewed as distributions.
    score=1 means the maps are identical.
    score=0 means the maps are completely opposite.
    
    Parameters:
        saliency_map1: numpy.ndarray, first saliency map
        saliency_map2: numpy.ndarray, second saliency map
        to_plot: bool, whether to display the output maps and similarity computation
        
    Returns:
        float: similarity score between the two saliency maps
    """
    if len(saliency_map1.shape) == 3:  # Check if the image has 3 channels
        saliency_map1 = cv2.cvtColor(saliency_map1, cv2.COLOR_BGR2GRAY)
    if len(saliency_map2.shape) == 3:  # Check if the image has 3 channels
        saliency_map2 = cv2.cvtColor(saliency_map2, cv2.COLOR_BGR2GRAY)

    # Resize saliency_map1 to match saliency_map2 size
    map1 = cv2.resize(saliency_map1, (saliency_map2.shape[1], saliency_map2.shape[0]))
    map1 = map1.astype(np.float64)
    map2 = saliency_map2.astype(np.float64)
    
    # Normalize the map values to lie between 0-1 and then sum to 1
    def normalize_map(saliency_map):
        if np.any(saliency_map):  # If map is non-zero
            saliency_map = (saliency_map - np.min(saliency_map)) / (np.max(saliency_map) - np.min(saliency_map))
            saliency_map /= np.sum(saliency_map)
        return saliency_map

    map1 = normalize_map(map1)
    map2 = normalize_map(map2)
    
    # Return NaN if either map is entirely NaN
    if np.isnan(map1).all() or np.isnan(map2).all():
        return np.nan
    
    # Compute histogram intersection
    diff = np.minimum(map1, map2)
    score = np.sum(diff)
    res_map_norm = None
    figtitle = 'Similarity'
    # Plot if requested
    if to_plot:
        res_map_norm = diff / np.max(diff)
        #plt.figure()
        #plt.imshow(res_map_norm, cmap='viridis')
        #plt.colorbar()
        #plt.title(figtitle, fontsize=14)
        #plt.axis('off')
        #plt.show()
    return score, res_map_norm

# Example usage:
# score = similarity(saliency_map1, saliency_map2, to_plot=True)

def KLdiv(saliency_map, fixation_map, to_plot=True):
    """
    Computes the KL-divergence between two saliency maps viewed as distributions.
    
    Parameters:
        saliency_map (numpy.ndarray): The saliency map
        fixation_map (numpy.ndarray): The human fixation map
        
    Returns:
        float: The KL-divergence score
    """
    # Resize saliency_map to match fixation_map size
    if len(saliency_map.shape) == 3:  # Convert to single channel if map1 is three-channel
        saliency_map = cv2.cvtColor(saliency_map, cv2.COLOR_BGR2GRAY)
    map1 = cv2.resize(saliency_map, (fixation_map.shape[1], fixation_map.shape[0])).astype(np.float64)
    map2 = fixation_map.astype(np.float64)
    
    # Normalize map1 and map2 to sum to 1
    if np.any(map1):
        map1 /= np.sum(map1)
    if np.any(map2):
        map2 /= np.sum(map2)
    
    # Compute KL-divergence with epsilon to prevent log(0)
    eps = np.finfo(np.float64).eps
    kl_map = map2 * np.log(eps + map2 / (map1 + eps))
    kl_div = np.sum(kl_map)

    res_map_norm = None
    
    figtitle = 'KL Divergence'
    if to_plot:
        res_map_norm = kl_map / np.max(kl_map)

        #plt.figure()
        #plt.imshow(res_map_norm, cmap='Reds')
        #plt.colorbar()
        #plt.title(figtitle, fontsize=14)
        #plt.axis('off')
        #plt.show()

    return kl_div, res_map_norm

# Example usage:
# saliency_map = cv2.imread('path_to_saliency_map.jpg', cv2.IMREAD_GRAYSCALE)
# fixation_map = cv2.imread('path_to_fixation_map.jpg', cv2.IMREAD_GRAYSCALE) > 128  # Threshold to binary
# kl_score = KLdiv(saliency_map, fixation_map)
# print(f"KL-divergence Score: {kl_score:.4f}")

def EMD(saliency_map, fixation_map, to_plot=True, downsize=16):
    """
    Compute the Earth Mover's Distance (EMD) between two maps.
    
    Parameters:
        saliency_map (numpy.ndarray): The saliency map.
        fixation_map (numpy.ndarray): The human fixation map (binary matrix).
        to_plot (bool): Whether to plot the results.
        downsize (int): Factor to downsize images for efficiency.
        
    Returns:
        float: EMD score.
    """
    if len(fixation_map.shape) == 3:  # Check if the image has 3 channels
        fixation_map = cv2.cvtColor(fixation_map, cv2.COLOR_BGR2GRAY)

    if len(saliency_map.shape) == 3:  # Check if the image has 3 channels
        saliency_map = cv2.cvtColor(saliency_map, cv2.COLOR_BGR2GRAY)
    
    # Resize and normalize maps
    im1 = cv2.resize(fixation_map, (fixation_map.shape[1] // downsize, fixation_map.shape[0] // downsize)).astype(np.float64)
    im2 = cv2.resize(saliency_map, im1.shape[::-1]).astype(np.float64)
    
    # Normalize maps so their sum is 1 (but scale similarly)
    im1 = cv2.normalize(im1, None, 0, 1, cv2.NORM_MINMAX)
    im2 = cv2.normalize(im2, None, 0, 1, cv2.NORM_MINMAX)

    im1_sum = np.sum(im1)
    im2_sum = np.sum(im2)

    #print(im1_sum, im2_sum)
    #print(im1.shape, im2.shape)

    if im1_sum == 0 or im2_sum == 0:
        return None  # Return None if one of the maps has zero sum

    # Normalize maps so their sum is 1
    im1 /= im1_sum
    im2 /= im2_sum

    # Convert 2D matrices to 1D point distributions
    h, w = im1.shape
    map1_points = np.array([[i % w, i // w, im1.flat[i]] for i in range(h * w)], dtype=np.float32)
    map2_points = np.array([[i % w, i // w, im2.flat[i]] for i in range(h * w)], dtype=np.float32)

    # Compute EMD
    score, _, _ = cv2.EMD(map1_points, map2_points, cv2.DIST_L2)
    #print('emd_score : ',score)

    flow_map = None

    figtitle = 'Earth Mover’s Distance'
    # Optionally, plot results
    if to_plot:
        flow_map = np.abs(im1 - im2) 

        #plt.figure()
        #plt.imshow(flow_map, cmap='inferno')
        #plt.colorbar()
        #plt.title(figtitle, fontsize=14)
        #plt.axis('off')
        #plt.show()
    
    return score, flow_map

# Example usage:
# saliency_map = cv2.imread('path_to_saliency_map.jpg', cv2.IMREAD_GRAYSCALE)
# fixation_map = cv2.imread('path_to_fixation_map.jpg', cv2.IMREAD_GRAYSCALE)
# score = EMD(saliency_map, fixation_map, to_plot=True)
# print(f"EMD Score: {score:.4f}")


def gaussian(img, fc):
    """
    Gaussian low pass filter (with circular boundary conditions).
    
    Parameters:
        img (numpy.ndarray): Input image
        fc (float): Cutoff frequency (-6dB)
        
    Returns:
        BF (numpy.ndarray): Blurred image
        gf (numpy.ndarray): Gaussian filter
    """
    sn, sm, c = img.shape
    n = max(sn, sm)
    n += n % 2
    n = 2 ** int(np.ceil(np.log2(n)))
    
    # Frequencies:
    fx, fy = np.meshgrid(np.arange(n), np.arange(n))
    fx = fx - n / 2
    fy = fy - n / 2
    
    # Convert cutoff frequency into Gaussian width:
    s = fc / np.sqrt(np.log(2))
    
    # Compute transfer function of Gaussian filter:
    gf = np.exp(-(fx**2 + fy**2) / (s**2))
    gf = np.fft.fftshift(gf)
    
    # Convolve (in Fourier domain) each color band:
    BF = np.zeros((n, n, c))
    for i in range(c):
        img_fft = np.fft.fft2(img[:, :, i], (n, n))
        BF[:, :, i] = np.real(np.fft.ifft2(img_fft * gf))
    
    # Crop output to have the same size as the input
    BF = BF[:sn, :sm, :]
    
    # If no output is expected, show a plot of the Gaussian filter
    if __name__ == "__main__":
        plt.figure()
        plt.plot(fx[int(n / 2), :], gf[int(n / 2), :])
        plt.grid(True)
        plt.plot([fc, fc], [0, 1], 'r')
        plt.xlabel('Cycles per image')
        plt.ylabel('Amplitude transfer function')
        plt.show()
    
    return BF, gf

# Example usage:
# Load an image using OpenCV and convert it to grayscale for testing
# img = cv2.imread('path/to/image.jpg')
# BF, gf = gaussian(img, 6)


def run_antonio_gaussian(img, sigma):
    """
    Given a desired sigma blur value, this computes the cutoff frequency
    required for the Gaussian low pass filter in the Fourier domain.
    
    Parameters:
        img (numpy.ndarray): Input image
        sigma (float): Desired standard deviation for Gaussian blur
        
    Returns:
        BF (numpy.ndarray): Blurred image
        gf (numpy.ndarray): Gaussian filter
    """
    sn, sm, c = img.shape
    n = max(sn, sm)
    
    # Compute cutoff frequency based on sigma
    fc = n * np.sqrt(np.log(2) / (2 * (np.pi**2) * (sigma**2)))
    
    # Call the Gaussian function with computed cutoff frequency
    BF, gf = gaussian(img, fc)
    return BF, gf

# Example usage:
# Load an image
# img = cv2.imread('path/to/image.jpg')  # Use a valid path to your image
# BF, gf = run_antonio_gaussian(img, sigma=2.0)  # Replace with your desired sigma

def NSS(saliency_map, fixation_map, to_plot=True):
    """
    Computes the Normalized Scanpath Saliency (NSS) between a saliency map and a fixation map.
    
    Parameters:
        saliency_map (numpy.ndarray): The saliency map
        fixation_map (numpy.ndarray): The human fixation map (binary matrix)
        
    Returns:
        float: The NSS score
    """
    if saliency_map.shape == 3:
        saliency_map = cv2.cvtColor(saliency_map, cv2.COLOR_BGR2GRAY)
    if fixation_map.shape == 3:
        fixation_map = cv2.cvtColor(fixation_map, cv2.COLOR_BGR2GRAY)

    # Resize saliency map to match fixation map size
    map_resized = cv2.resize(saliency_map, (fixation_map.shape[1], fixation_map.shape[0]))
    map_resized = map_resized.astype(np.float64)
    
    # Normalize the saliency map
    mean_val = np.mean(map_resized)
    std_val = np.std(map_resized)
    if std_val == 0:
        std_val = 1  
    map_normalized = (map_resized - mean_val) / std_val
    # Mean value at fixation locations
    score = np.mean(map_normalized[fixation_map.astype(bool)])
    res_map_norm = None

    figtitle = 'Normalized Scanpath Saliency(NSS)'
    if to_plot:
        
        map1 = fixation_map.astype(np.float64)
        res_map = np.zeros_like(fixation_map)
        res_map[map1 > 0] = map_normalized[map1 > 0]

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        res_map = cv2.dilate(res_map, kernel)

        if np.max(res_map) > 0:
            res_map_norm = res_map / np.max(res_map)
        else:
            res_map_norm = res_map
        
        #plt.figure()
        #plt.imshow(res_map_norm, cmap='viridis')
        #plt.colorbar()
        #plt.title(figtitle, fontsize=14)
        #plt.axis('off')
        #plt.show()

    return score, res_map_norm

# Example usage:
# saliency_map = cv2.imread('path_to_saliency_map.jpg', cv2.IMREAD_GRAYSCALE)
# fixation_map = cv2.imread('path_to_fixation_map.jpg', cv2.IMREAD_GRAYSCALE) > 128  # Threshold to binary
# score = NSS(saliency_map, fixation_map)
# print(f"NSS Score: {score:.4f}")

def InfoGain(saliency_map, fixation_map, to_plot=True, baseline_map=None):
    """
    Computes the Information Gain (IG) of the saliency map over a baseline map.
    
    Parameters:
        saliency_map (numpy.ndarray): The saliency map (low-level feature map).
        fixation_map (numpy.ndarray): The human fixation map (binary map of fixations).
        baseline_map (numpy.ndarray): The baseline map (e.g., center prior). Defaults to a Gaussian map.
        to_plot (bool): Whether to plot the resulting IG map.
        
    Returns:
        float: The Information Gain score.
    """
    # Resize saliency and fixation maps to match dimensions
    saliency_map = cv2.resize(saliency_map, (fixation_map.shape[1], fixation_map.shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.float64)
    fixation_map = fixation_map.astype(np.float64)

    # Normalize saliency map to sum to 1
    saliency_map = (saliency_map - np.min(saliency_map)) / (np.max(saliency_map) - np.min(saliency_map) + np.finfo(np.float64).eps)
    saliency_map /= np.sum(saliency_map) + np.finfo(np.float64).eps

    # Normalize fixation map to sum to 1
    fixation_map = fixation_map / np.sum(fixation_map) + np.finfo(np.float64).eps

    # Generate a Gaussian baseline map if none is provided
    if baseline_map is None:
        y, x = np.meshgrid(np.arange(fixation_map.shape[1]), np.arange(fixation_map.shape[0]))
        center_x, center_y = fixation_map.shape[1] // 2, fixation_map.shape[0] // 2
        sigma_x, sigma_y = fixation_map.shape[1] / 6, fixation_map.shape[0] / 6  # Adjust sigma as needed
        gaussian_map = np.exp(-(((x - center_x) ** 2) / (2 * sigma_x ** 2) + ((y - center_y) ** 2) / (2 * sigma_y ** 2)))
        baseline_map = gaussian_map

    # Resize and normalize baseline map to sum to 1
    baseline_map = cv2.resize(baseline_map, (fixation_map.shape[1], fixation_map.shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.float64)
    baseline_map = (baseline_map - np.min(baseline_map)) / (np.max(baseline_map) - np.min(baseline_map) + np.finfo(np.float64).eps)
    baseline_map /= np.sum(baseline_map) + np.finfo(np.float64).eps

    # Compute Information Gain
    log_saliency = np.log2(saliency_map + np.finfo(float).eps)
    log_baseline = np.log2(baseline_map + np.finfo(float).eps)
    info_gain = np.sum(fixation_map * (log_saliency - log_baseline)) / np.sum(fixation_map)
    ig_map = None
    # Compute the IG map for visualization

    # Plot the IG map if requested
    if to_plot:
        ig_map = fixation_map * (log_saliency - log_baseline)
        ig_map = (ig_map - np.min(ig_map)) / (np.max(ig_map) - np.min(ig_map) + np.finfo(float).eps)

        #plt.figure()
        #plt.imshow(ig_map, cmap='viridis')
        #plt.colorbar()
        #plt.title('Information Gain', fontsize=14)
        #plt.axis('off')
        #plt.show()

    return info_gain, ig_map

# Example usage:
# saliency_map = cv2.imread('path_to_saliency_map.jpg', cv2.IMREAD_GRAYSCALE)
# fixation_map = cv2.imread('path_to_fixation_map.jpg', cv2.IMREAD_GRAYSCALE) > 128  # Threshold to binary
# baseline_map = cv2.imread('path_to_baseline_map.jpg', cv2.IMREAD_GRAYSCALE)
# info_gain_score = InfoGain(saliency_map, fixation_map, baseline_map)
# print(f"Information Gain Score: {info_gain_score:.4f}")

def AUC_Judd(saliency_map, fixation_map, jitter=True):
    """
    Computes the Area Under the ROC Curve (AUC) for a given saliency map and fixation map using Judd's method.

    Parameters:
        saliency_map (numpy.ndarray): Saliency map of the image.
        fixation_map (numpy.ndarray): Ground truth binary fixation map.
        jitter (bool): Whether to apply jitter to avoid ties (default: True).
        to_plot (bool): Whether to plot the ROC curve (default: False).

    Returns:
        score (float): AUC score.
        tp (numpy.ndarray): True positive rates.
        fp (numpy.ndarray): False positive rates.
        all_threshes (numpy.ndarray): Threshold values used.
    """
    if saliency_map.shape == 3:
        saliency_map = cv2.cvtColor(saliency_map, cv2.COLOR_BGR2GRAY)
    if fixation_map.shape == 3:
        fixation_map = cv2.cvtColor(fixation_map, cv2.COLOR_BGR2GRAY)

    # Check if fixation map contains any fixations
    if not np.any(fixation_map):
        print('No fixations in fixation_map.')
        return np.nan, None, None, None

    # Resize saliency map to match fixation map
    saliency_map = cv2.resize(saliency_map, (fixation_map.shape[1], fixation_map.shape[0]))
    saliency_map = saliency_map.astype(np.float64)
    # Jitter to avoid ties
    if jitter:
        saliency_map += np.random.rand(*saliency_map.shape) / 1e7

    # Normalize saliency map
    saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())

    # Flatten maps
    S = saliency_map.flatten()
    F = fixation_map.flatten()

    Sth = S[F > 0]  # Saliency values at fixation locations
    Nfixations = len(Sth)
    Npixels = len(S)

    # Sort saliency values at fixations in descending order
    all_threshes = np.sort(Sth)[::-1]
    tp = np.zeros(Nfixations + 2)
    fp = np.zeros(Nfixations + 2)
    tp[-1], fp[-1] = 1, 1

    for i, thresh in enumerate(all_threshes, start=1):
        aboveth = np.sum(S >= thresh)
        tp[i] = i / Nfixations
        fp[i] = (aboveth - i) / (Npixels - Nfixations)

    # Calculate AUC using the trapezoidal rule
    score = np.trapz(tp, fp)
    all_threshes = np.concatenate(([1], all_threshes, [0]))

    return score, tp, fp, all_threshes

# Example usage:
# saliency_map = cv2.imread('path_to_saliency_map.jpg', cv2.IMREAD_GRAYSCALE)
# fixation_map = cv2.imread('path_to_fixation_map.jpg', cv2.IMREAD_GRAYSCALE)
# score, tp, fp, all_threshes = AUC_Judd(saliency_map, fixation_map, jitter=True, to_plot=True)
# print(f"AUC Judd Score: {score:.4f}")


def AUC_Borji(saliency_map, fixation_map, Nsplits=100, step_size=0.1, to_plot=False):
    """
    Computes the Area Under the ROC Curve (AUC) for a given saliency map and fixation map using Borji's method.

    Parameters:
        saliency_map (numpy.ndarray): Saliency map of the image.
        fixation_map (numpy.ndarray): Ground truth binary fixation map.
        Nsplits (int): Number of random splits (default: 100).
        step_size (float): Step size for sweeping through saliency map (default: 0.1).
        to_plot (bool): Whether to plot the ROC curve (default: False).

    Returns:
        score (float): AUC score.
        tp (numpy.ndarray): True positive rates.
        fp (numpy.ndarray): False positive rates.
    """
    if saliency_map.shape == 3:
        saliency_map = cv2.cvtColor(saliency_map, cv2.COLOR_BGR2GRAY)
    if fixation_map.shape == 3:
        fixation_map = cv2.cvtColor(fixation_map, cv2.COLOR_BGR2GRAY)

    score = np.nan

    # Check if there are fixations to predict
    if np.sum(fixation_map) <= 1:
        print('No fixationMap')
        return score, None, None

    # Resize saliency map to match fixation map size
    if saliency_map.shape != fixation_map.shape:
        saliency_map = cv2.resize(saliency_map, (fixation_map.shape[1], fixation_map.shape[0]))

    # Normalize saliency map
    saliency_map = (saliency_map - np.min(saliency_map)) / (np.max(saliency_map) - np.min(saliency_map))

    if np.all(np.isnan(saliency_map)):
        print('NaN saliencyMap')
        return score, None, None

    S = saliency_map.flatten()
    F = fixation_map.flatten()

    Sth = S[F > 0]  # Saliency map values at fixation locations
    Nfixations = len(Sth)
    Npixels = len(S)

    # Sample Nsplits values from random locations
    randfix = np.random.choice(S, size=(Nfixations, Nsplits), replace=True)

    auc = np.empty(Nsplits)
    for s in range(Nsplits):
        curfix = randfix[:, s]

        # Sweep through thresholds
        all_threshes = np.flip(np.arange(0, np.max(np.concatenate((Sth, curfix))), step=step_size))
        tp = np.zeros(len(all_threshes) + 2)
        fp = np.zeros(len(all_threshes) + 2)
        tp[0], tp[-1] = 0, 1
        fp[0], fp[-1] = 0, 1

        for i, thresh in enumerate(all_threshes):
            tp[i + 1] = np.sum(Sth >= thresh) / Nfixations
            fp[i + 1] = np.sum(curfix >= thresh) / Nfixations

        auc[s] = np.trapz(tp, fp)  # Calculate AUC using trapezoidal rule

    score = np.mean(auc)  # Mean AUC across splits

    # Plotting the results if required
    if to_plot:
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(saliency_map, cmap='gray')
        plt.title('Saliency Map with Fixations')
        fix_y, fix_x = np.where(fixation_map > 0)
        plt.scatter(fix_x, fix_y, color='red', s=10)

        plt.subplot(1, 2, 2)
        plt.plot(fp, tp, '.b-')
        plt.title(f'ROC Curve (AUC: {score:.4f})')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.show()

    return score, tp, fp

# Example usage:
# saliency_map = cv2.imread('path_to_saliency_map.jpg', cv2.IMREAD_GRAYSCALE)
# fixation_map = cv2.imread('path_to_fixation_map.jpg', cv2.IMREAD_GRAYSCALE)
# score, tp, fp = AUC_Borji(saliency_map, fixation_map, Nsplits=100, step_size=0.1, to_plot=True)
# print(f"AUC Borji Score: {score:.4f}")


def AUC_shuffled(saliency_map, fixation_map, other_map, Nsplits=100, step_size=0.1, to_plot=False):
    """
    Computes the Area Under the ROC Curve (AUC) for a given saliency map and fixation map using the shuffled method.

    Parameters:
        saliency_map (numpy.ndarray): Saliency map of the image.
        fixation_map (numpy.ndarray): Ground truth binary fixation map.
        other_map (numpy.ndarray): A binary fixation map from other images, used for random sampling.
        Nsplits (int): Number of random splits (default: 100).
        step_size (float): Step size for sweeping through saliency map (default: 0.1).
        to_plot (bool): Whether to plot the ROC curve (default: False).

    Returns:
        score (float): AUC score.
        tp (numpy.ndarray): True positive rates.
        fp (numpy.ndarray): False positive rates.
    """
    score = np.nan

    # Check if there are fixations to predict
    if np.sum(fixation_map) == 0:
        print('No fixationMap')
        return score, None, None

    # Resize saliency map to match fixation map size
    if saliency_map.shape != fixation_map.shape:
        saliency_map = cv2.resize(saliency_map, (fixation_map.shape[1], fixation_map.shape[0]))

    # Normalize saliency map
    saliency_map = (saliency_map - np.min(saliency_map)) / (np.max(saliency_map) - np.min(saliency_map))

    if np.all(np.isnan(saliency_map)):
        print('NaN saliencyMap')
        return score, None, None

    S = saliency_map.flatten()
    F = fixation_map.flatten()
    Oth = other_map.flatten()

    Sth = S[F > 0]  # Saliency map values at fixation locations
    Nfixations = len(Sth)

    # For each fixation, sample Nsplits values from the saliency map at random locations from other images
    ind = np.where(Oth > 0)[0]  # Find fixation locations on other images
    Nfixations_oth = min(Nfixations, len(ind))
    randfix = np.empty((Nfixations_oth, Nsplits))

    for i in range(Nsplits):
        randind = np.random.choice(ind, size=len(ind), replace=False)  # Randomize choice of fixation locations
        randfix[:, i] = S[randind[:Nfixations_oth]]  # Saliency map values at random fixation locations

    # Calculate AUC for each random split (set of random locations)
    auc = np.empty(Nsplits)
    for s in range(Nsplits):
        curfix = randfix[:, s]

        all_threshes = np.flip(np.arange(0, np.max(np.concatenate((Sth, curfix))), step=step_size))
        tp = np.zeros(len(all_threshes) + 2)
        fp = np.zeros(len(all_threshes) + 2)
        tp[0], tp[-1] = 0, 1
        fp[0], fp[-1] = 0, 1

        for i, thresh in enumerate(all_threshes):
            tp[i + 1] = np.sum(Sth >= thresh) / Nfixations
            fp[i + 1] = np.sum(curfix >= thresh) / Nfixations_oth

        auc[s] = np.trapz(tp, fp)  # Calculate AUC using trapezoidal rule

    score = np.mean(auc)  # Mean AUC across splits

    # Plotting the results if required
    if to_plot:
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(saliency_map, cmap='gray')
        plt.title('Saliency Map with Fixations')
        fix_y, fix_x = np.where(fixation_map > 0)
        plt.scatter(fix_x, fix_y, color='red', s=10)

        plt.subplot(1, 2, 2)
        plt.plot(fp, tp, '.b-')
        plt.title(f'ROC Curve (AUC: {score:.4f})')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.show()

    return score, tp, fp

# Example usage:
# saliency_map = cv2.imread('path_to_saliency_map.jpg', cv2.IMREAD_GRAYSCALE)
# fixation_map = cv2.imread('path_to_fixation_map.jpg', cv2.IMREAD_GRAYSCALE)
# other_map = cv2.imread('path_to_other_map.jpg', cv2.IMREAD_GRAYSCALE)
# score, tp, fp = AUC_shuffled(saliency_map, fixation_map, other_map, Nsplits=100, step_size=0.1, to_plot=True)
# print(f"AUC Shuffled Score: {score:.4f}")

def saliency_compare(sal1,sal2,fix1,fix2,to_plot):

    score1, tp1, fp1, all_threshes1 = AUC_Judd(sal1,fix2,to_plot = os.path.join(to_plot,"AUD_Judd1.png"))
    score2, tp2, fp2, all_threshes2 = AUC_Judd(sal2,fix1,to_plot = os.path.join(to_plot,"AUD_Judd2.png"))

    cc = CC(sal1,sal2)
    sim = similarity(sal1,sal2,to_plot = os.path.join(to_plot,"similarity.png"))

    emd1 = EMD(sal1,fix2,to_plot = os.path.join(to_plot,"EMD1.png"))
    emd2 = EMD(sal2,fix1,to_plot = os.path.join(to_plot,"END2.png"))

    kl1 = KLdiv(sal1,fix2)
    kl2 = KLdiv(sal2,fix1)

    nss1 = NSS(sal1,fix2)
    nss2 = NSS(sal2,fix1)

    ig1 = InfoGain(sal1,fix2,sal2)
    ig2 = InfoGain(sal2,fix1,sal1)

    metrics = ['AUC Judd 1', 'AUC Judd 2', 'CC', 'Similarity', 'EMD 1', 'EMD 2', 
           'KL Divergence 1', 'KL Divergence 2', 'NSS 1', 'NSS 2', 
           'InfoGain 1', 'InfoGain 2']

    values = [score1, score2, cc, sim, emd1, emd2, kl1, kl2, nss1, nss2, ig1, ig2]
    # Create a DataFrame
    df = pd.DataFrame([values], columns=metrics)

    # Specify the file path to save the CSV
    csv_file = os.path.join(to_plot, 'saliency_results.csv')

    # Save the DataFrame to a CSV file
    df.to_csv(csv_file, index=False, header=True)

def normalize_map(sal_map):
    return (sal_map - np.min(sal_map)) / (np.max(sal_map) - np.min(sal_map))
def make_level_sets(heatmap, thresholds, colormap):
    colored_map = np.zeros((*heatmap.shape, 3))
    for i, thresh in enumerate(thresholds):
        mask = heatmap >= thresh
        for c in range(3):
            colored_map[:, :, c] += mask * colormap[i, c]
    return np.clip(colored_map, 0, 1)
def scatter_plot_heatmap(x, y, color_map):
    if len(x) == 0 or len(y) == 0:
        return  # Skip plotting if there are no points
    plt.scatter(x, y, c=color_map, edgecolors='black', s=20)

def visualize_AUC(salMap, fixations):
    npoints = 10  # number of points to sample on ROC curve

    # Prepare the color map for correctly detected and missed fixations
    colmap = np.flipud(cm.jet(np.linspace(0, 1, npoints)))  # Create the color map
    G = np.linspace(0.5, 1, 20)
    tpmap = np.hstack((np.zeros((len(G), 1)), G[:, np.newaxis], np.zeros((len(G), 1))))  # green
    fpmap = np.hstack((G[:, np.newaxis], np.zeros((len(G), 1)), np.zeros((len(G), 1))))  # red

    # Compute AUC-Judd
    heatmap = salMap.astype(np.float64)  # Normalize heatmap
    score, tp, fp, allthreshes = AUC_Judd(heatmap, fixations)

    N = int(np.ceil(len(allthreshes) / npoints))
    allthreshes_samp = allthreshes[::N]  # Sample threshold values

    # Normalize the heatmap
    heatmap_norm = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))

    # Generate the saliency map with level sets
    salMap_col = make_level_sets(heatmap, allthreshes_samp, colmap)
    plt.figure(figsize=(12, 6))

    # Plot the saliency map with level sets
    plt.subplot(1, 2, 1)
    plt.imshow(salMap_col)
    plt.axis('off')

    # Plot the ROC curve
    tp1 = tp[::N]
    fp1 = fp[::N]
    plt.subplot(1, 2, 2)
    plt.plot(fp, tp, 'b')
    for ii in range(npoints):
        plt.plot(fp1[ii], tp1[ii], '.', color=colmap[ii], markersize=20)
    plt.title(f'AUC: {score:.2f}', fontsize=14)
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.axis('square')

    # Plot the level sets
    plt.figure('saliency map level sets')
    nplot = int(np.floor(npoints / 2))
    for ii in range(nplot):
        temp = heatmap_norm >= allthreshes_samp[2 * ii]  # Plot every other level set
        temp2 = np.zeros((temp.shape[0], temp.shape[1], 3))
        temp2[:, :, 0] = temp * colmap[2 * ii, 0]
        temp2[:, :, 1] = temp * colmap[2 * ii, 1]
        temp2[:, :, 2] = temp * colmap[2 * ii, 2]
        plt.subplot(1, nplot, ii + 1)
        plt.imshow(temp2)
        plt.axis('off')

    # Plot the true positives and false negatives
    plt.figure('True Positives and False Negatives')
    for ii in range(nplot):
        temp = heatmap_norm >= allthreshes_samp[2 * ii]
        res = fixations * temp
        res_neg = fixations * (1 - temp)
        plt.subplot(1, nplot, ii + 1)
        plt.imshow(temp)
        plt.axis('off')

        # Plot true positives
        J, I = np.where(res == 1)
        scatter_plot_heatmap(I, J, tpmap)

        # Plot false positives
        J, I = np.where(res_neg == 1)
        scatter_plot_heatmap(I, J, fpmap)

    plt.show()

    return score

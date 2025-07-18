import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import gabor

def extract_Ifeature(img):
    return np.mean(img, axis=2)

def extract_Rfeature(img):
    return img[:, :, 2] - (img[:, :, 1] + img[:, :, 0]) / 2

def extract_Gfeature(img):
    return img[:, :, 1] - (img[:, :, 2] + img[:, :, 0]) / 2

def extract_Bfeature(img):
    return img[:, :, 0] - (img[:, :, 1] + img[:, :, 2]) / 2

def extract_Yfeature(img):
    R, G, B = img[:, :, 2], img[:, :, 1], img[:, :, 0]
    return R + G - 2 * (np.abs(R - G) + B)

def Ifeature_diff(I1, I2):
    I2_resized = cv2.resize(I2, (I1.shape[1], I1.shape[0]), interpolation=cv2.INTER_NEAREST)
    return np.abs(I1 - I2_resized)

def RGBfeature_diff(Rc, Gc, Rs, Gs):
    Rs_resized = cv2.resize(Rs, (Rc.shape[1], Rc.shape[0]), interpolation=cv2.INTER_NEAREST)
    Gs_resized = cv2.resize(Gs, (Gc.shape[1], Gc.shape[0]), interpolation=cv2.INTER_NEAREST)
    return np.abs((Rc - Gc) - (Gs_resized - Rs_resized))

def directionfeature_diff(oc, os):
    os_resized = cv2.resize(os, (oc.shape[1], oc.shape[0]), interpolation=cv2.INTER_NEAREST)
    return np.abs(oc - os_resized)

def gabor_features(gray, frequencies=(0.25,), thetas=(0, np.pi/4, np.pi/2, 3*np.pi/4)):
    features = []
    for theta in thetas:
        filt_real, _ = gabor(gray, frequency=frequencies[0], theta=theta)
        features.append(filt_real)
    return features

def build_gaussian_pyramid(img, levels=9):
    pyramid = [img]
    for _ in range(1, levels):
        img = cv2.GaussianBlur(img, (3, 3), 0)
        img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_LINEAR)
        pyramid.append(img)
    return pyramid

def normalize_and_resize(feature_list, shape):
    normed = []
    for f in feature_list:
        f_resized = cv2.resize(f, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
        f_norm = cv2.normalize(f_resized, None, 0, 1, cv2.NORM_MINMAX)
        normed.append(f_norm)
    return np.mean(normed, axis=0)

def itti(img):

    if len(img.shape) == 2:  # Check if the image is grayscale
        img = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB for processing
    else:
        img = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
    
    img = img.astype(np.float32) / 255.0  # Ensure the image is in float32 for further processing
    H, W = img.shape[:2]

    # 构建金字塔
    pyramid = build_gaussian_pyramid(img, levels=9)

    # 提取特征
    I_pyr = [extract_Ifeature(p) for p in pyramid]
    R_pyr = [extract_Rfeature(p) for p in pyramid]
    G_pyr = [extract_Gfeature(p) for p in pyramid]
    B_pyr = [extract_Bfeature(p) for p in pyramid]
    Y_pyr = [extract_Yfeature(p) for p in pyramid]

    gray_pyr = [cv2.cvtColor((p * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0 for p in pyramid]
    gabor_pyr = [gabor_features(gray) for gray in gray_pyr]  # shape: [level][orientation]

    # 对比尺度对
    scales = [(2, 5), (2, 6), (3, 6), (3, 7), (4, 7), (4, 8)]

    # 强度通道
    I_diff = [Ifeature_diff(I_pyr[c], I_pyr[s]) for (c, s) in scales]
    I_map = normalize_and_resize(I_diff, (H, W))

    # 颜色通道：RG 与 BY
    RG_diff = [RGBfeature_diff(R_pyr[c], G_pyr[c], R_pyr[s], G_pyr[s]) for (c, s) in scales]
    BY_diff = [RGBfeature_diff(B_pyr[c], B_pyr[c], Y_pyr[s], Y_pyr[s]) for (c, s) in scales]
    C_map = normalize_and_resize(RG_diff + BY_diff, (H, W))

    # 方向通道：合并 4 个方向
    O_diff_all = []
    for ori in range(4):
        O_diff = [directionfeature_diff(gabor_pyr[c][ori], gabor_pyr[s][ori]) for (c, s) in scales]
        O_map = normalize_and_resize(O_diff, (H, W))
        O_diff_all.append(O_map)
    O_map = np.mean(O_diff_all, axis=0)

    # 最终显著性图 = 平均 I、C、O
    saliency = (I_map + C_map + O_map) / 3
    saliency = cv2.normalize(saliency, None, 0, 1, cv2.NORM_MINMAX)

    # 保存显著性图
    saliency_img = (saliency * 255).astype(np.uint8)
    return saliency_img
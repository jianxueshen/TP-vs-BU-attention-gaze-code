import os
import argparse
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as image
import matplotlib.cm as cm
import cv2
import colorsys
from scipy.ndimage import gaussian_filter
from numba import njit,jit
from line_profiler import profile

def draw_display(dispsize):
    screen = np.zeros((dispsize[1], dispsize[0], 3), dtype='int')

    dpi = 100.0

    figsize = (dispsize[0] / dpi, dispsize[1] / dpi)

    fig = plt.figure(figsize=figsize, dpi=dpi, frameon=False)
    ax = plt.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.axis([0, dispsize[0], 0, dispsize[1]])
    ax.imshow(screen)  # , origin='upper')

    return fig,ax

@njit
def gaussian(x, sx, y=None, sy=None):

    if y == None:
        y = x
    if sy == None:
        sy = sx

    xo = x / 2
    yo = y / 2

    M = np.zeros([x, y], dtype=float)

    for i in range(x):
        for j in range(y):
            M[i, j] = np.exp(
                -1.0 * (((float(i) - xo) ** 2 / (2 * sx * sx)) + ((float(j) - yo) ** 2 / (2 * sy * sy))))

    return M

def create_gaussian_template(gwh=100, gsdwh=None):
    gsdwh = gwh / 6 if gsdwh is None else gsdwh
    template = np.zeros((gwh, gwh), dtype=np.float32)
    template[gwh // 2, gwh // 2] = 1  # Set the central point
    template = gaussian_filter(template, sigma=gsdwh)
    lowbound = np.mean(template[template > 0])
    template[template<lowbound] = 0
    return template

def draw_bound(map):
    
    rgb_image = cv2.cvtColor(map, cv2.COLOR_RGBA2RGB)
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

    mean_value = np.mean(gray_image[gray_image!=0])
    thread_value = mean_value * 1.5

    #print(mean_value)

    _, binary_image = cv2.threshold(gray_image, thread_value, 255, cv2.THRESH_BINARY)
    #hsv_image  = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
    #lower_red = np.array(low_bound)
    #upper_red = np.array(high_bound)
    
    #thresh = cv2.adaptiveThreshold(gray_image,127,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,101,2)

    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = []
    for contour in contours:
        add_contour = True
        for point in contour:
            x, y = point[0]
            if x == 0 or y == 0 or x == gray_image.shape[1] - 1 or y == gray_image.shape[0] - 1:
                add_contour = False
                break
        if add_contour:
            filtered_contours.append(contour)
    print(len(filtered_contours))
    contour_image = map.copy()
    cv2.drawContours(contour_image, filtered_contours, -1, (0, 255, 0), 2)
    cv2.imshow('contour', contour_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return contour_image


def draw_heatmap(raw, dispsize, imagefile=None, alpha=0.5, savefilename=None,
                 draw_b=None, gaussianwh=200, gaussiansd=None):


    #fig,ax = draw_display(dispsize)

    gwh = gaussianwh
    gsdwh = gwh / 6 if (gaussiansd is None) else gaussiansd
    gaus = gaussian(gwh, gsdwh)

    strt = int(gwh / 2)
    heatmapsize = dispsize[1] + 2 * strt, dispsize[0] + 2 * strt
    heatmap = np.zeros(heatmapsize)
    #print(heatmapsize)
    gazepoints = []
    if len(raw[0]) == 2:
        gazepoints = list(map(lambda q: (int(round(q[0])), int(round(q[1])), 1), raw))
    else:
        gazepoints =  list(map(lambda q: (int(round(q[0])), int(round(q[1])), int(round(q[2]))), raw))
   

    for gaze_point in gazepoints:
        y, x, intensity = gaze_point
        x_center = x + strt
        y_center = y + strt
        if strt <= x_center < dispsize[1]+strt and strt <= y_center < dispsize[0]+strt:
            # Gaze point is within the display boundaries
            current_gaussian = (gaussian(gwh, gsdwh) * intensity)
            heatmap[x_center-strt:x_center+strt, y_center-strt: y_center+strt] += current_gaussian
        #if (not 0 < x < dispsize[0]) or (not 0 < y < dispsize[1]):
        #    hadj = [0, gwh]
        #    vadj = [0, gwh]
        #    if 0 > x:
        #        hadj[0] = abs(x)
        #        x = 0
        #    elif dispsize[0] < x:
        #        hadj[1] = gwh - int(x - dispsize[0])
        #    if 0 > y:
        #        vadj[0] = abs(y)
        #        y = 0
        #    elif dispsize[1] < y:
        #        vadj[1] = gwh - int(y - dispsize[1])
        #    # add adjusted Gaussian to the current heatmap
        #    try:
        #        heatmap[y:y + vadj[1], x:x + hadj[1]] += gaus[vadj[0]:vadj[1], hadj[0]:hadj[1]] * intensity
        #    except:
        #        # fixation was probably outside of display
        #        pass
        #else:
        #    # add Gaussian to the current heatmap
        #    #heatmap[y:y + gwh, x:x + gwh] += gaus * gazepoints[i][2]
        #    current_gaussian = (gaus * intensity)
        #    #current_gaussian[np.isnan(current_gaussian)] = 0  # Replace NaN values with 0
        #    heatmap[x:x + gwh, y:y + gwh] += current_gaussian
    # resize heatmap
    heatmap = heatmap[strt:dispsize[1] + strt, strt:dispsize[0] + strt]
    # remove zeros
    #print(heatmap.shape)
    
    lowbound = np.mean(heatmap[heatmap > 0])
    heatmap[heatmap < lowbound] = 0

    #heatbound = np.max(heatmap[heatmap > 0])*0.8
    #print('max heat:', maxheat)

    # draw heatmap on top of image
    #ax.imshow(heatmap, cmap='jet', alpha=1)
    #ax.invert_yaxis()
#
    ## FINISH PLOT
    ## invert the y axis, as (0,0) is top left on a display
    #
    ## save the figure if a file name was provided
#
    #if savefilename != None:
    #    #cv2.imwrite(savefilename,heatmap)
    #    fig.savefig(savefilename)
#
    #if draw_b:
    #    map = cv2.imread(savefilename, cv2.IMREAD_UNCHANGED)
    #    map=draw_bound(map)
    #    cv2.imwrite(savefilename, map)
#
    #if imagefile != None:
    #    # check if the path to the image exists
    #    if not os.path.isfile(imagefile):
    #        raise Exception("ERROR in draw_display: imagefile not found at '%s'" % imagefile)
    #    # load image
    #    img = cv2.imread(imagefile)
    #    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #    
    #    heatimg = cv2.imread(savefilename, cv2.IMREAD_UNCHANGED)
    #    heatimg=cv2.cvtColor(heatimg, cv2.COLOR_RGBA2RGB)
#
    #    if img.size != heatimg.size:
    #        hh,hw=heatimg.shape[:2]
    #        img = cv2.resize(src=img,dsize=(hw,hh))
#
    #    #print(img.shape)
    #    #print(heatimg.shape)
#
    #    overlap = cv2.addWeighted(img, 1, heatimg, 0.8, 0)
    #    cv2.imwrite(savefilename, overlap)
        

    return heatmap


#def main(input_path, video_path, output_path):
#    # Read the input video
#    cap = cv2.VideoCapture(video_path)
#    if not cap.isOpened():
#        print("Error: Video file not found.")
#        return
#
#    # Get video frame properties
#    frame_width = int(cap.get(3))
#    frame_height = int(cap.get(4))
#    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#
#    # Define the size of the heatmap display
#    dispsize = (frame_width, frame_height)
#
#    # Initialize video writer for the output
#    fourcc = cv2.VideoWriter_fourcc(*'XVID')
#    out = cv2.VideoWriter(output_path, fourcc, 30.0, (frame_width, frame_height))
#
#    # Read gazepoints from the CSV file
#    gazepoints = gazepoint(input_path)
#
#    # Iterate through video frames
#    frame_number = 0
#    while True:
#        ret, frame = cap.read()
#        if not ret:
#            break
#
#        # Draw the heatmap on the current frame
#        heatmap_frame = draw_heatmap(gazepoints, dispsize)
#        heatmap_frame = cv2.cvtColor(heatmap_frame, cv2.COLOR_BGR2RGB)
#
#        # Overlay the heatmap on the frame
#        alpha = 0.5  # You can adjust the transparency
#        output_frame = cv2.addWeighted(frame, 1 - alpha, heatmap_frame, alpha, 0)
#
#        # Write the frame to the output video
#        out.write(output_frame)
#
#        frame_number += 1
#        print(f"Processed frame {frame_number}/{frame_count}")
#
#    # Release video objects
#    cap.release()
#    out.release()
#    cv2.destroyAllWindows()
#
def blank_heatmap(dispsize):
    heatmap = np.zeros((dispsize[1], dispsize[0]), dtype=np.float32)
    return heatmap


def draw_heatmap_v2(raw, dispsize, gaussianwh=100, gaussiansd=None):
    gwh = gaussianwh
    gsdwh = gwh / 6 if gaussiansd is None else gaussiansd

    heatmap = np.zeros((dispsize[1], dispsize[0]), dtype=np.float32)

    for gaze_point in raw:
        x, y = map(int, map(round, gaze_point[:2]))
        if 0 <= x < dispsize[0] and 0 <= y < dispsize[1]:
            intensity = 1 if len(gaze_point) == 2 else int(round(gaze_point[2]))
            heatmap[y, x] += intensity

    heatmap = gaussian_filter(heatmap, sigma=gsdwh)
    lowbound = np.mean(heatmap[heatmap > 0])
    heatmap[heatmap < lowbound] = 0

    return heatmap

def heatmap_gaze(raw, dispsize):
    heatmap = np.zeros((dispsize[1], dispsize[0]), dtype=np.float32)
    for gaze_point in raw:
        x, y = map(int, map(round, gaze_point[:2]))
        if 0 <= x < dispsize[0] and 0 <= y < dispsize[1]:
            intensity = 1 if len(gaze_point) == 2 else int(round(gaze_point[2]))
            heatmap[y, x] += intensity
    return heatmap

@profile
def heatmap_gauss(heatmap, gaussianwh=101, gaussiansd=None):
    gwh = gaussianwh
    heatmap = cv2.GaussianBlur(heatmap, (gwh, gwh), 0)
    lowbound = np.mean(heatmap[heatmap > 0])
    heatmap[heatmap < lowbound] = 0

    return heatmap




@njit
def add_gaussian_to_heatmap(heatmap, template, x, y):
    gwh = template.shape[0]
    half_gwh = gwh // 2

    # Calculate heatmap boundaries
    x_min = max(x - half_gwh, 0)
    x_max = min(x + half_gwh, heatmap.shape[1])
    y_min = max(y - half_gwh, 0)
    y_max = min(y + half_gwh, heatmap.shape[0])

    # Calculate template boundaries
    template_x_min = max(half_gwh - x, 0)
    template_x_max = gwh - max(x + half_gwh - heatmap.shape[1], 0)
    template_y_min = max(half_gwh - y, 0)
    template_y_max = gwh - max(y + half_gwh - heatmap.shape[0], 0)

    # Ensure the slices are not empty
    if x_max > x_min and y_max > y_min:
        heatmap[y_min:y_max, x_min:x_max] += template[template_y_min:template_y_max, template_x_min:template_x_max]

    return heatmap


'''
if __name__ == "__main__":
    input_path = r'D:\eyetracking\picture_test\retest1_gaze.csv'
    display_width = 1920
    display_height = 1080
    alpha = 0.5
    output_name = 'retest1_heat_test.png'
    background_image = r'D:\eyetracking\picture_test\resizedtest1.jpg'
    ngaussian = 200
    sd = None
    drawcontours = True

    draw_heatmap(gazepoints, (display_width, display_height), imagefile=background_image, alpha=alpha, savefilename=None,
                 draw_b=drawcontours, gaussianwh=ngaussian, gaussiansd=sd)
'''

import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.image as image
from matplotlib.patches import Circle
import pandas as pd
import numpy as np
import cv2
import math
from line_profiler import profile

def draw_display(dispsize,imagefile):
    """

    """

    screen = np.zeros((dispsize[0], dispsize[1], 3), dtype='int')

    if imagefile != None:
        # check if the path to the image exists
        if not os.path.isfile(imagefile):
            raise Exception("ERROR in draw_display: imagefile not found at '%s'" % imagefile)
        # load image
        img = cv2.imread(imagefile)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if img.shape != (dispsize[0], dispsize[1],3):
            img = cv2.resize(src=img,dsize=(dispsize[0], dispsize[1]))

        screen = img

    dpi = 100.0

    figsize = (dispsize[0] / dpi, dispsize[1] / dpi)

    fig = plt.figure(figsize=figsize, dpi=dpi, frameon=False)
    ax = plt.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.axis([0, dispsize[0], 0, dispsize[1]])
    ax.imshow(screen)  # , origin='upper')

    return fig,ax


def gazePlot(data,size=(1920,1080),imagefile=None, savefilename=None, show_fig=None):
	
    color_shuweihong = (184,180,227)
    color_heyelv = (64, 104, 27)
    #paths = [] 
    img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    data = np.array(data)
    index, x, y, d = data[:,0], data[:, 1], data[:, 2], data[:, 3]
    
    #current_path = []
    
    for i in range(len(x)):
         if not math.isnan(x[i]) and not math.isnan(y[i]) and not math.isnan(d[i]):
            circle_center = (int(x[i]), int(y[i]))
            #print("path circlr is ",circle_center)
            circle_radius = int(d[i] / 100) if d[i] < 3000 else 100
            cv2.circle(img, circle_center, circle_radius, color_heyelv, -1)
            if i > 0 and not math.isnan(x[i - 1]) and not math.isnan(y[i - 1]):
                cv2.line(img, (int(x[i - 1]), int(y[i - 1])), circle_center, color_heyelv, 1) 

            cv2.putText(img, str(int(index[i])), (circle_center[0]-20,circle_center[1]+8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_shuweihong, 2)

    #cv2.imshow("gazepath",img)      
    return img
    #if i == len(x) - 1 or (i + 1 < len(x) and data[i, 0] != data[i + 1, 0]):

    #fig,ax = draw_display(dispsize,imagefile)
    ##ax.imshow(imagefile)
    #
    #for i in range(len(x)):
    #    circle = Circle((x[i], y[i]), radius=d[i]/50, color='green', alpha=0.5)
    #    ax.add_patch(circle)
    #    if i>0:
    #        plt.plot([x[i - 1], x[i]], [y[i - 1], y[i]], 'g-', linewidth=1)
    #    ax.annotate(str(i + 1), (x[i], y[i]), color='pink', fontsize=14,
    #            ha='center', va='center')
#
    #ax.invert_yaxis()
 #
    #if show_fig:
    #    plt.show()
#
    #if savefilename != None:
    #    #cv2.imwrite(savefilename,heatmap)
    #    fig.savefig(savefilename)
#
#
    #return fig
'''
if __name__ == "__main__":

    input_path = r'D:\video\tank\tanktrue_4.csv'

    #output_name = 'retest1_gazepath.png'
    #background_image = r'D:\eyetracking\picture_test\resizedtest1.jpg'
    col_name = ['Fixation Index','Fixation Point X[px]','Fixation Point Y[px]','Fixation Duration[ms]']
    data = pd.read_csv(input_path)[col_name][1:].drop_duplicates().dropna()
    img = gazePlot(data)
    cv2.imshow("test path",img)
    cv2.imwrite('output_image_path.jpg', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
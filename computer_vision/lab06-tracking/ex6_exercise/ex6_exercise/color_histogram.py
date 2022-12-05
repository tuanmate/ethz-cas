import cv2
import numpy as np

def color_histogram(xmin, ymin, xmax, ymax, frame, hist_bin):
    #print("Computing histogram...", end="")
    roi = frame[ymin:ymax+1, xmin:xmax+1]
    flat_frame = roi.reshape(-1, 3)
    hist, _ = np.histogramdd(flat_frame, hist_bin, [[0, 255]]*3, density=False)
    hist = hist / np.sum(hist)
    #print("done")
    return hist
    
import numpy as np

def color_histogram(xmin, ymin, xmax, ymax, frame, hist_bin):
    roi = frame[ymin:ymax, xmin:xmax]
    flat_frame = roi.reshape(-1, 3)
    hist, _ = np.histogramdd(flat_frame, hist_bin, [[0, 255]]*3, density=False)
    hist = hist / np.sum(hist)
    return hist
    
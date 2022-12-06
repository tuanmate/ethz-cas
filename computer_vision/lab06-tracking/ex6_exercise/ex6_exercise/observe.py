import numpy as np

from color_histogram import color_histogram
from chi2_cost import chi2_cost

def observe(particles, frame, bbox_height, bbox_width, hist_bin, hist, sigma_observe):
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]
    half_w = bbox_width*0.5
    half_h = bbox_height*0.5

    weights = list()
    for p in particles:
        xmin = min(max(0, round(p[0]-half_w)), frame_width-1)
        ymin = min(max(0, round(p[1]-half_h)), frame_height-1)
        xmax = min(max(0, round(p[0]+half_w)), frame_width)
        ymax = min(max(0, round(p[1]+half_h)), frame_height)

        #xmin = max(0, round(p[0]-half_w))
        #xmax = min(round(p[0]+half_w), frame_width)
        #if xmin == 0:
        #    xmax = bbox_width-1
        #if xmax == frame_width:
        #    xmin = frame_width-bbox_width
        #ymin = max(0, round(p[1]-half_h))
        #ymax = min(round(p[1]+half_h), frame_height)
        #if ymin == 0:
        #    ymax = bbox_height-1
        #if ymax == frame_height:
        #    ymin = frame_height-bbox_height

        current_hist = color_histogram(xmin, ymin, xmax, ymax, frame, hist_bin)
        chi_dist = chi2_cost(hist, current_hist)
        weight = 1./(np.sqrt(2 * np.pi) * sigma_observe) * np.exp(-1*(chi_dist * chi_dist) / (2 * sigma_observe * sigma_observe))
        weights.append(weight)

    weights = np.array(weights).reshape(-1, 1)
    weights /= np.sum(weights)

    return weights
import numpy as np

from color_histogram import color_histogram
from chi2_cost import chi2_cost

def observe(particles, frame, bbox_height, bbox_width, hist_bin, hist, sigma_observe):
    #print("Observing...", end='')
    #print(" - ", particles.shape)
    #print(" - ", frame.shape)
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]
    half_w = bbox_width*0.5
    half_h = bbox_height*0.5

    weights = list()
    for p in particles:
        #print("   * particle:", p)
        xmin = min(max(0, round(p[1]-half_w)), frame_width-2)
        ymin = min(max(0, round(p[0]-half_h)), frame_height-2)
        xmax = min(max(0, round(p[1]+half_w)), frame_width-1)
        ymax = min(max(0, round(p[0]+half_h)), frame_height-1)
        #print("   - bbox: ",xmin, ymin, xmax, ymax)
        current_hist = color_histogram(xmin, ymin, xmax, ymax, frame, hist_bin)
        #print("   - hists:", hist.shape, current_hist.shape)
        chi_dist = chi2_cost(hist, current_hist)
        #print("   - chi dist: ", chi_dist)
        weight = 1./(np.sqrt(2 * np.pi) * sigma_observe) * np.exp((chi_dist * chi_dist) / (2 * sigma_observe * sigma_observe))
        weights.append(weight)

    weights = np.array(weights).reshape(-1, 1)
    weights /= np.sum(weights)

    #print("done")
    #print(weights)
    return weights
import sys
import cv2
from itertools import product
from pathlib import Path

from functions.extract_harris import extract_harris
from functions.vis_utils import plot_image_with_keypoints

SIGMAS = [0.5, 1.0, 2.0]
KS = [0.04, 0.05, 0.06]
#THRESHOLDS = [1e-6, 5e-6, 1e-5]
THRESHOLDS = [1e-6, 1e-5, 1e-4]

CONFIGS = product(SIGMAS, KS, THRESHOLDS)

input_path = Path(sys.argv[1])
img = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)

for sigma, k, thresh in CONFIGS:
    out_path = f"{input_path.stem}_{sigma}_{k}_{thresh}".replace('.','')+f"{input_path.suffix}"
    print(f"Extracting using {sigma}, {k}, {thresh}")
    corners1, C1 = extract_harris(img, sigma, k, thresh)
    plot_image_with_keypoints(out_path, img, corners1)
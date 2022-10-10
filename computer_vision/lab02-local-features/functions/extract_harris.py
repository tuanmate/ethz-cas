import numpy as np
import scipy
import cv2

# Harris corner detector
def extract_harris(img, sigma = 1.0, k = 0.05, thresh = 1e-5):
    '''
    Inputs:
    - img:      (h, w) gray-scaled image
    - sigma:    smoothing Gaussian sigma. suggested values: 0.5, 1.0, 2.0
    - k:        Harris response function constant. suggest interval: (0.04 - 0.06)
    - thresh:   scalar value to threshold corner strength. suggested interval: (1e-6 - 1e-4)
    Returns:
    - corners:  (q, 2) numpy array storing the keypoint positions [x, y]
    - C:     (h, w) numpy array storing the corner strength
    '''
    # Convert to float
    img = img.astype(float) / 255.0

    # Compute image gradients
    # TODO: implement the computation of the image gradients Ix and Iy here.
    # You may refer to scipy.signal.convolve2d for the convolution.
    # Do not forget to use the mode "same" to keep the image size unchanged.
    
    # Discrete derivation kernel in x direction
    dx = np.array([[-0.5, 0, 0.5]])

    # Discrete derivation kernel in y direction
    dy = dx.transpose()

    # Compute image gradients
    Ix = scipy.signal.convolve2d(img, dx, mode='same')
    Iy = scipy.signal.convolve2d(img, dy, mode='same')
    
    # Compute auto-correlation matrix:
    # M = [[Gx2, Gxy], [Gxy, Gy2]]
    Ix2 = Ix * Ix
    Iy2 = Iy * Iy
    Ixy = Ix * Iy

    # Size of Gaussian-kernel
    ksize = (3, 3)
    Gx2 = cv2.GaussianBlur(Ix2, ksize=ksize, sigmaX=sigma, borderType=cv2.BORDER_REPLICATE)
    Gy2 = cv2.GaussianBlur(Iy2, ksize=ksize, sigmaX=sigma, borderType=cv2.BORDER_REPLICATE)
    Gxy = cv2.GaussianBlur(Ixy, ksize=ksize, sigmaX=sigma, borderType=cv2.BORDER_REPLICATE)

    # Compute Harris response function
    C = Gx2*Gy2 - Gxy*Gxy - k * (Gx2 + Gy2)*(Gx2 + Gy2)

    # Detection with threshold
    # Perform non-maximum suppression and thresholding
    local_max = scipy.ndimage.maximum_filter(C, size=3)
    y, x = np.where((local_max == C) & (C > thresh))

    corners = np.array([x, y]).transpose()
    return corners, C


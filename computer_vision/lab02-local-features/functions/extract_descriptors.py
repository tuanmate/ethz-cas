import numpy as np

def filter_keypoints(img, keypoints, patch_size = 9):
    # Filter out keypoints that are too close to the edges

    # The patch should be centered around the keypoint, hence its size should be
    # odd
    assert(patch_size % 2 == 1)

    # Minimum distance from the edges:
    d = patch_size // 2 

    height, width = img.shape
    # Points' coordinates must be "min. distance" far away from both edges,
    # along both axises
    min_w_idx = min_h_idx = d
    max_w_idx = (width-1)-d
    max_h_idx = (height-1)-d

    # These intervals define a sub-rectangle in the image, with the following
    # top-left/bottom-right coordinates:
    tl = np.array([min_w_idx, min_h_idx])
    br = np.array([max_w_idx, max_h_idx])

    # Keep keypoints only from this sub-rectangle
    filtered_idx = np.all(np.logical_and(tl <= keypoints, keypoints <= br), axis=1)
    filtered_keypoints = keypoints[filtered_idx]

    return filtered_keypoints


# The implementation of the patch extraction is already provided here
def extract_patches(img, keypoints, patch_size = 9):
    '''
    Extract local patches for each keypoint
    Inputs:
    - img:          (h, w) gray-scaled images
    - keypoints:    (q, 2) numpy array of keypoint locations [x, y]
    - patch_size:   size of each patch (with each keypoint as its center)
    Returns:
    - desc:         (q, patch_size * patch_size) numpy array. patch descriptors for each keypoint
    '''
    h, w = img.shape[0], img.shape[1]
    img = img.astype(float) / 255.0
    offset = int(np.floor(patch_size / 2.0))
    ranges = np.arange(-offset, offset + 1)
    desc = np.take(img, ranges[:,None] * w + ranges + (keypoints[:, 1] * w + keypoints[:, 0])[:, None, None]) # (q, patch_size, patch_size)
    desc = desc.reshape(keypoints.shape[0], -1) # (q, patch_size * patch_size)
    return desc


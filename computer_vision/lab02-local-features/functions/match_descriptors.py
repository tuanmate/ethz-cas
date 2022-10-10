import numpy as np

def ssd(desc1, desc2):
    '''
    Sum of squared differences
    Inputs:
    - desc1:        - (q1, feature_dim) descriptor for the first image
    - desc2:        - (q2, feature_dim) descriptor for the first image
    Returns:
    - distances:    - (q1, q2) numpy array storing the squared distance
    '''
    assert desc1.shape[1] == desc2.shape[1]
    # Create meshgrid for all combinations of keypoint index pairs (indexing the
    # keypoints from the two images)
    idx1 = np.arange(desc1.shape[0])
    idx2 = np.arange(desc2.shape[0])

    # from the docs: "with inputs of length M and N, the outputs are of
    # shape (N, M) for ‘xy’ indexing and (M, N) for ‘ij’ indexing"
    grid_idx1, grid_idx2 = np.meshgrid(idx1, idx2, indexing='ij')

    # Calculate the sum of the square of feature-differences for
    # all combinations of keypoint pairs 
    ssd = np.sum(np.square(desc1[grid_idx1,:] - desc2[grid_idx2,:]), -1)

    return ssd

def match_descriptors(desc1, desc2, method = "one_way", ratio_thresh=0.5):
    '''
    Match descriptors
    Inputs:
    - desc1:        - (q1, feature_dim) descriptor for the first image
    - desc2:        - (q2, feature_dim) descriptor for the first image
    Returns:
    - matches:      - (m x 2) numpy array storing the indices of the matches
    '''
    assert desc1.shape[1] == desc2.shape[1]
    distances = ssd(desc1, desc2)
    q1, q2 = desc1.shape[0], desc2.shape[0]
    matches = None

    if method == "one_way": # Query the nearest neighbor for each keypoint in image 1
        # x and y are indexing the keypoints in the first and second image,
        # rescpecively
        x = np.arange(q1)
        y = np.argmin(distances, 1)
        matches = np.array([x, y]).transpose()

    elif method == "mutual":
        # Partial matches: same as for "one_way"
        x1 = np.arange(q1)
        y1 = np.argmin(distances, 1)

        # For actual matches also verify partial matches in the other way: 
        # We should get back the indices of the first image's
        # keypoints from the original partial matches
        x2 = np.argmin(distances, 0)
        x = x1[np.where(x1 == x2[y1])]

        # Get the indices of the matching keypoints' from the second image
        y = y1[np.where(x1 == x2[y1])]
        
        matches = np.array([x, y]).transpose()

    elif method == "ratio":
        # Get first two elements of sorted (along the second axis, in ascending
        # order) "one-way" distances
        closest_matches = np.partition(distances, (0,1), 1)
        
        # avoid division by 0, no exception raised when dividing by NaN
        closest_matches[closest_matches[:,1]==0, 1] = np.nan

        # Consider points only where first and second nearest-neighbor's ratio
        # is below the threshold
        x = np.where((~np.isnan(closest_matches[:,1])) & 
                     (closest_matches[:, 0]/closest_matches[:, 1] < ratio_thresh))[0]
        y = np.argmin(distances, 1)[x]
        matches = np.array([x, y]).transpose()

    else:
        raise NotImplementedError
    return matches


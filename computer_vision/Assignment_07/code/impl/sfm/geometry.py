import numpy as np

from impl.dlt import BuildProjectionConstraintMatrix
from impl.util import MakeHomogeneous, HNormalize
from impl.sfm.corrs import GetPairMatches
# from impl.opt import ImageResiduals, OptimizeProjectionMatrix

# # Debug
# import matplotlib.pyplot as plt
# from impl.vis import Plot3DPoints, PlotCamera, PlotProjectedPoints


def EstimateEssentialMatrix(K, im1, im2, matches):
  # TODO
  # Normalize coordinates (to points on the normalized image plane)

  K_inv = np.linalg.inv(K)
  kps1 = np.append(im1.kps, [[1]]*im1.kps.shape[0], axis=1)
  kps2 = np.append(im2.kps, [[1]]*im2.kps.shape[0], axis=1)
  normalized_kps1 = np.dot(K_inv, kps1.transpose()).transpose()
  normalized_kps2 = np.dot(K_inv, kps2.transpose()).transpose()

  
  # Assemble constraint matrix as equation 2.1
  constraint_matrix = np.zeros((matches.shape[0], 9))
  for i in range(matches.shape[0]):
    # TODO
    # Add the constraints
    n = matches[i,0]
    m = matches[i,1]
    x1 = normalized_kps1[n, 0]
    y1 = normalized_kps1[n, 1]
    x2 = normalized_kps2[m, 0]
    y2 = normalized_kps2[m, 1]

    constraint_row = np.array([x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1])
    constraint_matrix[i, :] = constraint_row
    
  
  # Solve for the nullspace of the constraint matrix
  _, _, vh = np.linalg.svd(constraint_matrix)
  vectorized_E_hat = vh[-1,:]

  # TODO
  # Reshape the vectorized matrix to it's proper shape again
  E_hat = vectorized_E_hat.reshape(3,3)

  # TODO
  # We need to fulfill the internal constraints of E
  # The first two singular values need to be equal, the third one zero.
  # Since E is up to scale, we can choose the two equal singluar values arbitrarily
  U, D, V = np.linalg.svd(E_hat)
  D[0] = 1
  D[1] = 1
  D[2] = 0
  D = np.diag(D)
  E = U @ D @ V

  # This is just a quick test that should tell you if your estimated matrix is not correct
  # It might fail if you estimated E in the other direction (i.e. kp2' * E * kp1)
  # You can adapt it to your assumptions.
  for i in range(matches.shape[0]):
    kp1 = normalized_kps1[matches[i,0],:]
    kp2 = normalized_kps2[matches[i,1],:]

    assert(abs(kp1.transpose() @ E @ kp2) < 0.01)

  return E


def DecomposeEssentialMatrix(E):

  u, s, vh = np.linalg.svd(E)

  # Determine the translation up to sign
  t_hat = u[:,-1]

  W = np.array([
    [0, -1, 0],
    [1, 0, 0],
    [0, 0, 1]
  ])

  # Compute the two possible rotations
  R1 = u @ W @ vh
  R2 = u @ W.transpose() @ vh

  # Make sure the orthogonal matrices are proper rotations (Determinant should be 1)
  if np.linalg.det(R1) < 0:
    R1 *= -1

  if np.linalg.det(R2) < 0:
    R2 *= -1

  # Assemble the four possible solutions
  sols = [
    (R1, t_hat),
    (R2, t_hat),
    (R1, -t_hat),
    (R2, -t_hat)
  ]

  return sols

def TriangulatePoints(K, im1, im2, matches):

  R1, t1 = im1.Pose()
  R2, t2 = im2.Pose()
  P1 = K @ np.append(R1, np.expand_dims(t1, 1), 1)
  P2 = K @ np.append(R2, np.expand_dims(t2, 1), 1)

  # Ignore matches that already have a triangulated point
  new_matches = np.zeros((0, 2), dtype=int)

  num_matches = matches.shape[0]
  for i in range(num_matches):
    p3d_idx1 = im1.GetPoint3DIdx(matches[i, 0])
    p3d_idx2 = im2.GetPoint3DIdx(matches[i, 1])
    if p3d_idx1 == -1 and p3d_idx2 == -1:
      new_matches = np.append(new_matches, matches[[i]], 0)


  num_new_matches = new_matches.shape[0]

  points3D = np.zeros((num_new_matches, 3))

  for i in range(num_new_matches):

    kp1 = im1.kps[new_matches[i, 0], :]
    kp2 = im2.kps[new_matches[i, 1], :]

    # H & Z Sec. 12.2
    A = np.array([
      kp1[0] * P1[2] - P1[0],
      kp1[1] * P1[2] - P1[1],
      kp2[0] * P2[2] - P2[0],
      kp2[1] * P2[2] - P2[1]
    ])

    _, _, vh = np.linalg.svd(A)
    homogeneous_point = vh[-1]
    points3D[i] = homogeneous_point[:-1] / homogeneous_point[-1]


  # We need to keep track of the correspondences between image points and 3D points
  im1_corrs = new_matches[:,0]
  im2_corrs = new_matches[:,1]

  # TODO
  # Filter points behind the cameras by transforming them into each camera space and checking the depth (Z)
  # Make sure to also remove the corresponding rows in `im1_corrs` and `im2_corrs`

  # Filter points behind the first camera
  filtered_idx = np.zeros((num_new_matches), dtype=bool)
  rt1 = np.append(R1, np.expand_dims(t1, 1), 1)
  for i in range(num_new_matches):
    p = rt1 @ np.append(points3D[i], [1], 0) 
    filtered_idx[i] = p[2] > 0

  im1_corrs = im1_corrs[filtered_idx]
  im2_corrs = im2_corrs[filtered_idx]
  points3D = points3D[filtered_idx]

  # Filter points behind the second camera
  filtered_idx = np.zeros((points3D.shape[0]), dtype=bool)
  rt2 = np.append(R2, np.expand_dims(t2, 1), 1)
  for i in range(points3D.shape[0]):
    p = rt2 @ np.append(points3D[i], [1], 0) 
    filtered_idx[i] = p[2] > 0

  im1_corrs = im1_corrs[filtered_idx]
  im2_corrs = im2_corrs[filtered_idx]
  points3D = points3D[filtered_idx]

  return points3D, im1_corrs, im2_corrs

def EstimateImagePose(points2D, points3D, K):  

  # TODO
  # We use points in the normalized image plane.
  # This removes the 'K' factor from the projection matrix.
  # We don't normalize the 3D points here to keep the code simpler.
  points2D_hom = np.append(points2D, [[1]]*points2D.shape[0], axis=1)
  normalized_points2D = (np.linalg.inv(K) @ points2D_hom.transpose()).transpose()

  constraint_matrix = BuildProjectionConstraintMatrix(normalized_points2D, points3D)

  # We don't use optimization here since we would need to make sure to only optimize on the se(3) manifold
  # (the manifold of proper 3D poses). This is a bit too complicated right now.
  # Just DLT should give good enough results for this dataset.

  # Solve for the nullspace
  _, _, vh = np.linalg.svd(constraint_matrix)
  P_vec = vh[-1,:]
  P = np.reshape(P_vec, (3, 4), order='C')

  # Make sure we have a proper rotation
  u, s, vh = np.linalg.svd(P[:,:3])
  R = u @ vh

  if np.linalg.det(R) < 0:
    R *= -1

  _, _, vh = np.linalg.svd(P)
  C = np.copy(vh[-1,:])

  t = -R @ (C[:3] / C[3])

  return R, t

def TriangulateImage(K, image_name, images, registered_images, matches):

  # TODO 
  # Loop over all registered images and triangulate new points with the new image.
  # Make sure to keep track of all new 2D-3D correspondences, also for the registered images

  image = images[image_name]
  points3D = np.zeros((0,3))
  # You can save the correspondences for each image in a dict and refer to the `local` new point indices here.
  # Afterwards you just add the index offset before adding the correspondences to the images.
  corrs = {}

  for registered_image_name in registered_images:
    registered_image = images[registered_image_name]
    pair_matches = GetPairMatches(registered_image_name, image_name, matches)

    points, reg_corrs, im_corrs = TriangulatePoints(K, registered_image, image, pair_matches)
    
    if not points.shape[0]:
      continue

    offset = points3D.shape[0]
    n_points = points.shape[0]
    points3D = np.append(points3D, points, 0)

    points3d_idx = list(range(offset, offset+n_points))

    if image_name in corrs.keys():
      corrs[image_name]["corrs"] = np.append(corrs[image_name]["corrs"], im_corrs, 0)
      corrs[image_name]["point3d_idx"] = np.append(corrs[image_name]["point3d_idx"], points3d_idx, 0)
    else:
      corrs[image_name] = dict()
      corrs[image_name]["corrs"] = im_corrs
      corrs[image_name]["point3d_idx"] = points3d_idx

    corrs[registered_image_name] = dict()
    corrs[registered_image_name]["corrs"] = reg_corrs
    corrs[registered_image_name]["point3d_idx"] = points3d_idx

  return points3D, corrs
  

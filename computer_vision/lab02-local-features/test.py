import unittest
import numpy as np

import functions.extract_descriptors as ext_desc
import functions.match_descriptors as match_desc

class TestFeatures(unittest.TestCase):
    def test_gradients(self):
        pass

    def test_auto_corr_matrix(self):
        pass

    def test_harris_response(self):
        pass

    def test_detection_criteria(self):
        pass

    def test_local_descriptors(self):
        img = np.random.rand(40, 50)
        keypoints = np.array([[0,0], [39, 49], [3,3], [3,4], [4,3], [4,4], [45,
        35], [45,36], [46,35], [46,36], [25, 20]])

        with self.assertRaises(AssertionError):
            ext_desc.filter_keypoints(img, keypoints, 8)        

        gt = np.array([[4,4], [45,35], [25, 20]])
        filtered = ext_desc.filter_keypoints(img, keypoints)
        self.assertTrue(np.array_equal(gt, filtered))

    def test_ssd(self):
        desc1 = np.array([[1, 2, 3],
                          [4, 5, 6],
                          [2, 6, 1]])
        desc2 = np.array([[2, 4, 6],
                          [8,10,12],
                          [2, 3, 4],
                          [5, 6, 7]])
        dist = match_desc.ssd(desc1, desc2)

        gt = np.array([[14, 194,  3, 48],
                       [ 5,  77, 12,  3],
                       [29, 173, 18, 45]])
        self.assertTrue(np.array_equal(gt, dist))

    def test_match_descriptors(self):
        desc1 = np.array([[1, 2, 3],
                          [4, 5, 6],
                          [2, 6, 1]], dtype=np.float32)
        desc2 = np.array([[2, 4, 6],
                          [8,10,12],
                          [2, 3, 4],
                          [5, 6, 7]], dtype=np.float32)
        
        # distances:
        # np.array([[14, 194,  3, 48],
        #           [ 5,  77, 12,  3],
        #           [29, 173, 18, 45]])

        
        with self.assertRaises(NotImplementedError):
            match_desc.match_descriptors(desc1, desc2, "unknown")

        matches = match_desc.match_descriptors(desc1, desc1, "one_way")
        gt = np.array([[0,0],[1,1],[2,2]])
        self.assertTrue(np.array_equal(gt, matches))

        matches = match_desc.match_descriptors(desc1, desc2, "one_way")
        gt = np.array([[0,2],[1,3],[2,2]])
        self.assertTrue(np.array_equal(gt, matches))

        matches = match_desc.match_descriptors(desc2, desc1, "one_way")
        gt = np.array([[0,1],[1,1],[2,0],[3,1]])
        self.assertTrue(np.array_equal(gt, matches))
        
        matches = match_desc.match_descriptors(desc1, desc2, "mutual")
        gt = np.array([[0,2],[1,3]])
        self.assertTrue(np.array_equal(gt, matches))

        matches = match_desc.match_descriptors(desc2, desc1, "mutual")
        gt = np.array([[2,0],[3,1]])
        self.assertTrue(np.array_equal(gt, matches))

        matches = match_desc.match_descriptors(desc1, desc2, "ratio")
        gt = np.array([[0,2]])
        self.assertTrue(np.array_equal(gt, matches))

        matches = match_desc.match_descriptors(desc2, desc1, "ratio")
        gt = np.array([[0,1],[1,1],[2,0],[3,1]])
        self.assertTrue(np.array_equal(gt, matches))

        # Check division with 0
        desc1 = np.array([[1, 2, 3],
                          [1, 2, 7]], dtype=np.float32)
        desc2 = np.array([[1, 2, 3],
                          [1, 2, 3],
                          [1, 2, 8]], dtype=np.float32)
        matches = match_desc.match_descriptors(desc1, desc2, "ratio")
        gt = np.array([[1, 2]])
        self.assertTrue(np.array_equal(gt, matches))

if __name__ == '__main__':
    unittest.main()
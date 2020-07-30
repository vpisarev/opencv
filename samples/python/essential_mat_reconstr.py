import numpy as np, cv2, matplotlib.pyplot as plt, time
from mpl_toolkits.mplot3d import axes3d, Axes3D
import pdb

def getEpipolarError(F, pts1_, pts2_, inliers):
    pts1 = np.concatenate((pts1_.T, np.ones((1, pts1_.shape[0]))))[:,inliers]
    pts2 = np.concatenate((pts2_.T, np.ones((1, pts2_.shape[0]))))[:,inliers]
    lines2 = np.dot(F  , pts1)
    lines1 = np.dot(F.T, pts2)

    return np.median((np.abs(np.sum(pts1 * lines1, axis=0)) / np.sqrt(lines1[0,:]**2 + lines1[1,:]**2) +
                      np.abs(np.sum(pts2 * lines2, axis=0)) / np.sqrt(lines2[0,:]**2 + lines2[1,:]**2))/2)

if __name__ == '__main__':
    with open('../data/essential_mat_data.txt', 'r') as f:
        image1 = cv2.imread(f.readline()[:-1]) # remove '\n'
        image2 = cv2.imread(f.readline()[:-1])
        K = np.array([[float(x) for x in f.readline().split(' ')],
                      [float(x) for x in f.readline().split(' ')],
                      [float(x) for x in f.readline().split(' ')]])

    print('find keypoints and compute descriptors')
    detector = cv2.SIFT_create(nfeatures=20000)
    keypoints1, descriptors1 = detector.detectAndCompute(cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY), None)
    keypoints2, descriptors2 = detector.detectAndCompute(cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY), None)

    matcher = cv2.FlannBasedMatcher(dict(algorithm=0, trees=5), dict(checks=32))
    print('match with FLANN, size of descriptors', descriptors1.shape, descriptors2.shape)
    matches_vector = matcher.knnMatch(descriptors1, descriptors2, k=2)

    print('find good keypoints')
    pts1 = []; pts2 = []
    for m in matches_vector:
        # compare best and second match using Lowe ratio test
        if  m[0].distance / m[1].distance < 0.75:
            pts1.append(keypoints1[m[0].queryIdx].pt)
            pts2.append(keypoints2[m[0].trainIdx].pt)
    pts1 = np.array(pts1); pts2 = np.array(pts2)
    print('points size', pts1.shape[0])

    print('Essential matrix RANSAC')
    start = time.time()
    E, inliers = cv2.findEssentialMat(pts1, pts2, K, cv2.RANSAC, 0.999, 1.0)
    print('RANSAC time', time.time() - start, 'seconds')
    print('Median error to epipolar lines', getEpipolarError
          (np.dot(np.linalg.inv(K).T, np.dot(E, np.linalg.inv(K))), pts1, pts2, inliers.squeeze()),
           'number of inliers', inliers.sum())

    print('Decompose essential matrix')
    R1, R2, t = cv2.decomposeEssentialMat(E)

    # Assume relative pose. Fix the first camera
    P1 = np.concatenate((K, np.zeros((3,1))), axis=1) #   K [I | 0]
    P2s = [np.dot(K, np.concatenate((R1,  t), axis=1)), # K[R1 |  t]
           np.dot(K, np.concatenate((R1, -t), axis=1)), # K[R1 | -t]
           np.dot(K, np.concatenate((R2,  t), axis=1)), # K[R2 |  t]
           np.dot(K, np.concatenate((R2, -t), axis=1))] # K[R2 | -t]

    obj_pts_per_cam = []
    # enumerate over all P2 projection matrices
    for cam_idx, P2 in enumerate(P2s):
        obj_pts = []
        for i, (pt1, pt2) in enumerate(zip(pts1, pts2)):
            if not inliers[i]:
                continue
            # find object point by triangulation of image points by projection matrices
            obj_pt = cv2.triangulatePoints(P1, P2, pt1, pt2)
            obj_pt /= obj_pt[3]
            # check if reprojected point has positive depth
            if obj_pt[2] > 0:
                obj_pts.append([obj_pt[0], obj_pt[1], obj_pt[2]])
        obj_pts_per_cam.append(obj_pts)

    best_cam_idx = np.array([len(obj_pts_per_cam[0]),len(obj_pts_per_cam[1]),
                             len(obj_pts_per_cam[2]),len(obj_pts_per_cam[3])]).argmax()
    max_pts = len(obj_pts_per_cam[best_cam_idx])
    print('Number of object points', max_pts)

    # filter object points to have reasonable depth
    MAX_DEPTH = 6.
    obj_pts = []
    for pt in obj_pts_per_cam[best_cam_idx]:
        if pt[2] < MAX_DEPTH:
            obj_pts.append(pt)
    obj_pts = np.array(obj_pts).reshape(len(obj_pts), 3)

    # visualize image points
    for i, (pt1, pt2) in enumerate(zip(pts1, pts2)):
        if inliers[i]:
            cv2.circle(image1, (int(pt1[0]), int(pt1[1])), 7, (255,0,0), -1)
            cv2.circle(image2, (int(pt2[0]), int(pt2[1])), 7, (255,0,0), -1)

    # concatenate two images
    image1 = np.concatenate((image1, image2), axis=1)
    # resize concatenated image
    new_img_size = 1200. * 800.
    image1 = cv2.resize(image1, (int(np.sqrt(image1.shape[1] * new_img_size / image1.shape[0])),
                                 int(np.sqrt (image1.shape[0] * new_img_size / image1.shape[1]))))

    # plot object points
    fig = plt.figure(figsize=(13.0, 11.0))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('equal')
    ax.scatter(obj_pts[:,0], obj_pts[:,1], obj_pts[:,2], c='r', marker='o', s=3)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('depth')
    ax.view_init(azim=-80, elev=110)

    # save figures
    cv2.imshow("matches", image1)
    cv2.imwrite('matches_E.png', image1)
    plt.savefig('reconstruction_3D.png')

    # cv2.waitKey(0)
    plt.show()

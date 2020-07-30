// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test {
enum TestSolver { Homogr, Fundam, Essen, PnP};
/*
* @rng -- reference to random generator
* @pts1 -- 2xN image points
* @pts2 -- for PnP is 3xN object points, otherwise 2xN image points.
* @two_calib -- True if two cameras have different calibration.
* @K1 -- intrinsic matrix of the first camera. For PnP only one camera.
* @K2 -- only if @two_calib is True.
* @pts_size -- required size of points.
* @inlier_ratio -- required inlier ratio
* @noise_std -- standard deviation of Gaussian noise of image points.
* @gt_inliers -- has size of number of inliers. Contains indices of inliers.
*/
int generatePoints (cv::RNG &rng, cv::Mat &pts1, cv::Mat &pts2, cv::Mat &K1, cv::Mat &K2,
                    bool two_calib, int pts_size, TestSolver test_case, double inlier_ratio,
                    double noise_std, std::vector<int> &gt_inliers);
/*
* for test case = 0, 1, 2 (homography and epipolar geometry): pts1 and pts2 are 3xN
* for test_case = 3 (PnP): pts1 are 3xN and pts2 are 4xN
* all points are of the same type as model
*/
double getError (TestSolver test_case, int pt_idx, const cv::Mat &pts1, const cv::Mat &pts2, const cv::Mat &model);
/*
* @inl_size -- number of ground truth inliers
* @pts1 and pts2 are of the same size as from function generatePoints(...)
*/
void checkInliersMask (TestSolver test_case, int inl_size, double thr,  const cv::Mat &pts1_,
                       const cv::Mat &pts2_, const cv::Mat &model, const cv::Mat &mask);

int generatePoints (cv::RNG &rng, cv::Mat &pts1, cv::Mat &pts2, cv::Mat &K1, cv::Mat &K2,
                    bool two_calib, int pts_size, TestSolver test_case, double inlier_ratio, double noise_std,
                    std::vector<int> &gt_inliers) {

    auto eulerAnglesToRotationMatrix = [] (double pitch, double yaw, double roll) {
        // Calculate rotation about x axis
        cv::Matx33d R_x (1, 0, 0, 0, cos(roll), -sin(roll), 0, sin(roll), cos(roll));
        // Calculate rotation about y axis
        cv::Matx33d R_y (cos(pitch), 0, sin(pitch), 0, 1, 0, -sin(pitch), 0, cos(pitch));
        // Calculate rotation about z axis
        cv::Matx33d R_z (cos(yaw), -sin(yaw), 0, sin(yaw), cos(yaw), 0, 0, 0, 1);
        return cv::Mat(R_z * R_y * R_x); // Combined rotation matrix
    };

    const double pitch_min = -CV_PI / 6, pitch_max = CV_PI / 6; // 30 degrees
    const double yaw_min = -CV_PI / 6, yaw_max = CV_PI / 6;
    const double roll_min = -CV_PI / 6, roll_max = CV_PI / 6;

    cv::Mat R = eulerAnglesToRotationMatrix(rng.uniform(pitch_min, pitch_max),
            rng.uniform(yaw_min, yaw_max), rng.uniform(roll_min, roll_max));

    // generate random translation,
    // if test for homography fails try to fix translation to zero vec so H is related by transl.
    cv::Vec3d t (rng.uniform(-0.5f, 0.5f), rng.uniform(-0.5f, 0.5f), rng.uniform(1.0f, 2.0f));

    // generate random calibration
    auto getRandomCalib = [&] () {
        return cv::Mat(cv::Matx33d(rng.uniform(100.0, 1000.0), 0, rng.uniform(100.0, 100.0),
                       0, rng.uniform(100.0, 1000.0), rng.uniform(-100.0, 100.0),
                       0, 0, 1.));
    };
    K1 = getRandomCalib();
    K2 = two_calib ? getRandomCalib() : K1.clone();

    // compute size of inliers and outliers
    const int inl_size = static_cast<int>(inlier_ratio * pts_size);
    const int out_size = pts_size - inl_size;

    // all points will have top 'inl_size' of their points inliers
    gt_inliers.clear(); gt_inliers.reserve(inl_size);
    for (int i = 0; i < inl_size; i++)
        gt_inliers.emplace_back(i);

    // double precision to multiply points by models
    const int pts_type = CV_64F;
    cv::Mat points3d;
    if (test_case == TestSolver::Homogr) {
        points3d.create(2, inl_size, pts_type);
        rng.fill(points3d, cv::RNG::UNIFORM, 0.0, 1.0); // keep small range
        // inliers must be planar points, let their 3D coordinate be 1
        cv::vconcat(points3d, cv::Mat::ones(1, inl_size, points3d.type()), points3d);
    } else if (test_case == TestSolver::Fundam || test_case == TestSolver::Essen) {
        // create 3D points which are inliers
        points3d.create(3, inl_size, pts_type);
        rng.fill(points3d, cv::RNG::UNIFORM, 0.0, 1.0);
    } else if (test_case == TestSolver::PnP) {
        //pts1 are image points, pts2 are object points
        pts2.create(3, inl_size, pts_type); // 3D inliers
        rng.fill(pts2, cv::RNG::UNIFORM, 0, 1);

        // Make sure the shape is in front of the camera
        cv::Mat points3d_transformed = R * pts2 + t * cv::Mat::ones(1, pts2.cols, pts2.type());
        double min_dist, max_dist;
        cv::minMaxIdx(points3d_transformed.row(2), &min_dist, &max_dist);
        if (min_dist < 0) t(2) -= min_dist + 1.0;

        // project 3D points (pts2) on image plane (pts1)
        pts1 = K1 * (R * pts2 + t * cv::Mat::ones(1, pts2.cols, pts2.type()));
        cv::divide(pts1.row(0), pts1.row(2), pts1.row(0));
        cv::divide(pts1.row(1), pts1.row(2), pts1.row(1));
        // make 2D points
        pts1 = pts1.rowRange(0,2);

        // create random outliers
        cv::Mat pts_outliers = cv::Mat(5, out_size, pts2.type());
        rng.fill(pts_outliers, cv::RNG::UNIFORM, 0, 1000);

        // merge inliers with random image points = outliers
        cv::hconcat(pts1, pts_outliers.rowRange(0,2), pts1);
        // merge 3D inliers with 3D outliers
        cv::hconcat(pts2, pts_outliers.rowRange(2, 5), pts2);

        // add Gaussian noise to image points
        cv::Mat noise (pts1.rows, pts1.cols, pts1.type());
        rng.fill(noise, cv::RNG::NORMAL, 0, noise_std); pts1 += noise;
        return inl_size;
    } else
        CV_Error(1, "Unknown solver!");

    // Make sure the shape is in front of the camera
    cv::Mat points3d_transformed = R * points3d + t * cv::Mat::ones(1, points3d.cols, points3d.type());
    double min_dist, max_dist;
    cv::minMaxIdx(points3d_transformed.row(2), &min_dist, &max_dist);
    if (min_dist < 0)
        t(2) -= min_dist + 1.0;
    //

    if (test_case != TestSolver::PnP) {
        // project 3D point on image plane
        // use two relative scenes. The first camera is P1 = K1 [I | 0], the second P2 = K2 [R | t]
        pts1 = K1 * points3d;
        pts2 = K2 * (R * points3d + t * cv::Mat::ones(1, points3d.cols, points3d.type()));

        // normalize by 3 coordinate
        cv::divide(pts1.row(0), pts1.row(2), pts1.row(0));
        cv::divide(pts1.row(1), pts1.row(2), pts1.row(1));
        cv::divide(pts2.row(0), pts2.row(2), pts2.row(0));
        cv::divide(pts2.row(1), pts2.row(2), pts2.row(1));

        // get 2D points
        pts1 = pts1.rowRange(0,2); pts2 = pts2.rowRange(0,2);

        // generate random outliers as 2D image points
        cv::Mat pts1_outliers(pts1.rows, out_size, pts1.type()),
                pts2_outliers(pts2.rows, out_size, pts2.type());
        rng.fill(pts1_outliers, cv::RNG::UNIFORM, 0, 1000);
        rng.fill(pts2_outliers, cv::RNG::UNIFORM, 0, 1000);
        // for epipolar geometry merge inliers and outliers
        cv::hconcat(pts1, pts1_outliers, pts1);
        cv::hconcat(pts2, pts2_outliers, pts2);

        // add normal / Gaussian noise to image points
        cv::Mat noise1 (pts1.rows, pts1.cols, pts1.type()), noise2 (pts2.rows, pts2.cols, pts2.type());
        rng.fill(noise1, cv::RNG::NORMAL, 0, noise_std); pts1 += noise1;
        rng.fill(noise2, cv::RNG::NORMAL, 0, noise_std); pts2 += noise2;
    }

    return inl_size;
}

double getError (TestSolver test_case, int pt_idx, const cv::Mat &pts1, const cv::Mat &pts2, const cv::Mat &model) {
    cv::Mat pt1 = pts1.col(pt_idx), pt2 = pts2.col(pt_idx);
    if (test_case == TestSolver::Homogr) { // reprojection error
        // compute Euclidean distance between given and reprojected points
        cv::Mat est_pt2 = model * pt1; est_pt2 /= est_pt2.at<double>(2);
        if (false) {
            cv::Mat est_pt1 = model.inv() * pt2; est_pt1 /= est_pt1.at<double>(2);
            return (cv::norm(est_pt1 - pt1) + cv::norm(est_pt2 - pt2)) / 2;
        }
        return cv::norm(est_pt2 - pt2);
    } else
    if (test_case == TestSolver::Fundam || test_case == TestSolver::Essen) {
        cv::Mat l2 = model     * pt1;
        cv::Mat l1 = model.t() * pt2;
        if (test_case == TestSolver::Fundam) // sampson error
            return pow(pt2.dot(l2),2) / (pow(l1.at<double>(0), 2) + pow(l1.at<double>(1), 2) +
                                         pow(l2.at<double>(0), 2) + pow(l2.at<double>(1), 2));
        else // symmetric geometric distance
            return (fabs(pt1.dot(l1)) / sqrt(pow(l1.at<double>(0),2) + pow(l1.at<double>(1),2)) +
                    fabs(pt2.dot(l2)) / sqrt(pow(l2.at<double>(0),2) + pow(l2.at<double>(1),2)))/2;
    } else
    if (test_case == TestSolver::PnP) { // PnP, reprojection error
        cv::Mat img_pt = model * pt2; img_pt /= img_pt.at<double>(2);
        return cv::norm(pt1 - img_pt);
    } else
        CV_Error(1, "undefined test_case");
}

void checkInliersMask (TestSolver test_case, int inl_size, double thr, const cv::Mat &pts1_,
                       const cv::Mat &pts2_, const cv::Mat &model, const cv::Mat &mask) {
    CV_Assert(!model.empty() && !mask.empty());

    cv::Mat pts1 = pts1_, pts2 = pts2_;
    if (pts1.type() != model.type()) {
        pts1.convertTo(pts1, model.type());
        pts2.convertTo(pts2, model.type());
    }
    // convert to homogeneous
    cv::vconcat(pts1, cv::Mat::ones(1, pts1.cols, pts1.type()), pts1);
    cv::vconcat(pts2, cv::Mat::ones(1, pts2.cols, pts2.type()), pts2);

    const auto * const mask_ptr = mask.ptr<uchar>();
    int num_found_inliers = 0;
    for (int i = 0; i < pts1.cols; i++)
        if (mask_ptr[i]) {
            ASSERT_LT(getError(test_case, i, pts1, pts2, model), thr);
            num_found_inliers++;
        }
    // check if RANSAC found at least 80% of inliers
    ASSERT_GT(num_found_inliers, 0.8 * inl_size);
}

TEST(usac_Homography, accuracy) {
    std::vector<int> gt_inliers;
    const int pts_size = 1500;
    cv::RNG &rng = cv::theRNG();
    for (double inl_ratio = 0.05; inl_ratio < 0.91; inl_ratio += 0.01) {
        cv::Mat pts1, pts2, K1, K2;
        int inl_size = generatePoints(rng,pts1, pts2, K1, K2, false /*two calib*/,
           pts_size, TestSolver ::Homogr, inl_ratio, 0.1 /*noise std*/, gt_inliers);
        // compute max_iters with standard upper bound rule for RANSAC with 1.2x tolerance
        const double conf = 0.99, thr = 2., max_iters = 1.2 * log(1 - conf) /
                 log(1 - pow(inl_ratio, 4 /* sample size */));
        cv::Mat mask, H = cv::findHomography(pts1, pts2,USAC_DEFAULT, thr, mask,
                                                   int(max_iters), conf);
        checkInliersMask(TestSolver::Homogr, inl_size, thr, pts1, pts2, H, mask);
    }
}

TEST(usac_Homography_parallel, accuracy) {
    std::vector<int> gt_inliers;
    const int pts_size = 1500;
    cv::RNG &rng = cv::theRNG();
    for (double inl_ratio = 0.05; inl_ratio < 0.91; inl_ratio += 0.01) {
        cv::Mat pts1, pts2, K1, K2;
        int inl_size = generatePoints(rng,pts1, pts2, K1, K2, false /*two calib*/,
          pts_size, TestSolver ::Homogr, inl_ratio, 0.2 /*noise std*/, gt_inliers);
        const double conf = 0.99, thr = 2., max_iters = 1.5 * log(1 - conf) /
                  log(1 - pow(inl_ratio, 4 /* sample size */));
        cv::Mat mask, H = cv::findHomography(pts1, pts2,USAC_PARALLEL, thr, mask,
                                                   int(max_iters), conf);
        checkInliersMask(TestSolver::Homogr, inl_size, thr, pts1, pts2, H, mask);
    }
}

TEST(usac_Fundamental, accuracy) {
    std::vector<int> gt_inliers;
    const int pts_size = 1500;
    cv::RNG &rng = cv::theRNG();
    // start from 20% otherwise max_iters will be too big
    for (double inl_ratio = 0.2; inl_ratio < 0.91; inl_ratio += 0.01) {
        cv::Mat pts1, pts2, K1, K2;
        int inl_size = generatePoints(rng,pts1, pts2, K1, K2, false /*two calib*/,
          pts_size, TestSolver ::Fundam, inl_ratio, 0.1 /*noise std*/, gt_inliers);
        const double conf = 0.99, thr = 1., max_iters = 1.2 * log(1 - conf) /
                    log(1 - pow(inl_ratio, 7 /* sample size */));
        cv::Mat mask, F = cv::findFundamentalMat(pts1, pts2,USAC_DEFAULT, thr, conf,
                             int(max_iters), mask);
        checkInliersMask(TestSolver::Fundam, inl_size, thr, pts1, pts2, F, mask);
    }
}

TEST(usac_Fundamental8pts, accuracy) {
    std::vector<int> gt_inliers;
    const int pts_size = 1500;
    cv::RNG &rng = cv::theRNG();
    for (double inl_ratio = 0.20; inl_ratio < 0.91; inl_ratio += 0.01) {
        cv::Mat pts1, pts2, K1, K2;
        int inl_size = generatePoints(rng,pts1, pts2, K1, K2, false /*two calib*/,
                      pts_size, TestSolver ::Fundam, inl_ratio, 0.1 /*noise std*/, gt_inliers);
        const double conf = 0.99, thr = 1., max_iters = 1.2 * log(1 - conf) /
            log(1 - pow(inl_ratio, 8 /* sample size */));
        cv::Mat mask, F = cv::findFundamentalMat(pts1, pts2,USAC_FM_8PTS, thr, conf,
                                                       int(max_iters), mask);
        checkInliersMask(TestSolver::Fundam, inl_size, thr, pts1, pts2, F, mask);
    }
}

TEST(usac_Essential, accuracy) {
    std::vector<int> gt_inliers;
    const int pts_size = 2000;
    cv::RNG &rng = cv::theRNG();
    // findEssentilaMat has by default number of maximum iterations equal to 1000.
    // It means that with 99% confidence we assume at least 34.08% of inliers
    for (double inl_ratio = 0.35; inl_ratio < 0.91; inl_ratio += 0.01) {
        cv::Mat pts1, pts2, K1, K2;
        int inl_size = generatePoints(rng,pts1, pts2, K1, K2, false /*two calib*/,
          pts_size, TestSolver ::Fundam, inl_ratio, 0.01 /*noise std, works bad with high noise*/, gt_inliers);
        const double conf = 0.99, thr = 1.;
        cv::Mat mask, E = cv::findEssentialMat(pts1, pts2, K1, USAC_DEFAULT, conf, thr, mask);
        // calibrate points
        cv::Mat cpts1_3d, cpts2_3d;
        cv::vconcat(pts1, cv::Mat::ones(1, pts1.cols, pts1.type()), cpts1_3d);
        cv::vconcat(pts2, cv::Mat::ones(1, pts2.cols, pts2.type()), cpts2_3d);
        cpts1_3d = K1.inv() * cpts1_3d; cpts2_3d = K1.inv() * cpts2_3d;
        checkInliersMask(TestSolver::Essen, inl_size, thr / ((K1.at<double>(0,0) + K1.at<double>(1,1)) / 2)+1e-5,
                cpts1_3d.rowRange(0,2), cpts2_3d.rowRange(0,2), E, mask);
    }
}
}

// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test {
   TEST(usac_Homography, accuracy) {
        cv::RNG rng; // random generator

        // generate random rotation matrix
        cv::Vec3d vec;
        rng.fill(vec, cv::RNG::UNIFORM, 0, 1);
        vec = vec / cv::norm(vec) * rng.uniform(0.0f, float(2 * CV_PI));
        cv::Mat R;
        cv::Rodrigues(vec, R);

        // generate random translation
        cv::Vec3d t = cv::Vec3d(rng.uniform(-0.5f, 0.5f), rng.uniform(-0.5f, 0.5f), rng.uniform(1.0f, 2.0f));

        // generate random calibration
        cv::Matx33d K = cv::Matx33d::zeros();
        K(0, 0) = rng.uniform(100, 1000);
        K(1, 1) = rng.uniform(100, 1000);
        K(0, 2) = rng.uniform(-100, 100);
        K(1, 2) = rng.uniform(-100, 100);
        K(2, 2) = 1;

        // define number of points and inlier ratio
        const int pts_size = 5000;
        const double inlier_ratio = 0.5;
        // compute inlier number
        const int inl_size = static_cast<int>(inlier_ratio * pts_size);
        const int out_size = pts_size - inl_size;

        cv::Mat inliers, outliers, points3d;
        inliers.create(2, inl_size, CV_64F);
        rng.fill(inliers, cv::RNG::UNIFORM, 0, 1);
        // inliers must be planar points, let their 3D coordinate be 1
        cv::vconcat(inliers, cv::Mat::ones(1, inl_size, inliers.type()), inliers);
        // outliers are random 3D points, not related by plane
        outliers.create(3, out_size, inliers.type());
        rng.fill(outliers, cv::RNG::UNIFORM, 0, 1);

        // merge inliers and outliers
        cv::hconcat(inliers, outliers, points3d);

        // project 3D point on image plane
        // use two relative scenes. The first camera is P1 = K [I | 0], the second P2 = K [R | t]
        cv::Mat pts1 = K * points3d;
        cv::Mat pts2 = K * (R * points3d + t * cv::Mat::ones(1, points3d.cols, points3d.type()));
        // normalize
        cv::divide(pts1.row(0), pts1.row(2), pts1.row(0));
        cv::divide(pts1.row(1), pts1.row(2), pts1.row(1));
        cv::divide(pts2.row(0), pts2.row(2), pts2.row(0));
        cv::divide(pts2.row(1), pts2.row(2), pts2.row(1));

        // add normal noise to image points
        cv::Mat noise1 (3, pts1.cols, pts1.type()), noise2 (3, pts2.cols, pts2.type());
        rng.fill(noise1, cv::RNG::NORMAL, 0, 0.1); pts1 += noise1;
        rng.fill(noise2, cv::RNG::NORMAL, 0, 0.1); pts2 += noise2;

        // run RANSAC
        cv::Mat mask;
        // compute max_iters with standard upper bound rule for RANSAC with 1.5x tolerance
        const double conf = 0.99, thr = 2., max_iters = 1.5 * log(1 - conf) /
                         log(1 - pow(static_cast<float>(inl_size) / pts_size, 4 /* sample size */));
        cv::Mat H = findHomography(pts1.rowRange(0,2).t(), pts2.rowRange(0,2).t(),
                USAC, thr, mask, static_cast<int>(max_iters), conf);
        CV_Assert(!H.empty());

        // convert image points to homogeneous by extending to 3D
        cv::Mat pts1_3d, pts2_3d;
        cv::vconcat(pts1.rowRange(0,2), cv::Mat::ones(1, pts_size, pts1.type()), pts1_3d);
        cv::vconcat(pts2.rowRange(0,2), cv::Mat::ones(1, pts_size, pts2.type()), pts2_3d);

        // project points by found H in both directions
        cv::Mat pts2_est = H * pts1_3d, pts1_est = H.inv() * pts2_3d;
        // normalize by 3 coordinate
        cv::divide(pts2_est.row(0), pts2_est.row(2), pts2_est.row(0));
        cv::divide(pts2_est.row(1), pts2_est.row(2), pts2_est.row(1));
        cv::divide(pts1_est.row(0), pts1_est.row(2), pts1_est.row(0));
        cv::divide(pts1_est.row(1), pts1_est.row(2), pts1_est.row(1));

        // compute difference
        cv::Mat diff2 = pts2.rowRange(0, 2) - pts2_est.rowRange(0,2);
        cv::Mat diff1 = pts1.rowRange(0, 2) - pts1_est.rowRange(0,2);

        // check inliers' mask
        const auto * const mask_ptr = mask.ptr<uchar>();
        int num_found_inliers = 0;
        for (int i = 0; i < pts_size; i++)
            // if RANSAC's output is inlier then point must be inliers
            if (mask_ptr[i]) {
                // compute Euclidean distance = between given and reprojected points
                // CV_Assert(cv::norm(diff1.col(i)) < thr); // in general this distance can be ommited
                CV_Assert(cv::norm(diff2.col(i)) < thr);
                num_found_inliers++;
            }

        // std::cout << "number of found inliers " << num_found_inliers << "\n";
        // check if RANSAC found at least 80% of inliers
        assert(num_found_inliers > 0.8 * inl_size);
    }
}

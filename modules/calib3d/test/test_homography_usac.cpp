// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test {
   TEST(usac_Homography, accuracy) {
        cv::RNG rng;
        cv::Vec3d vec;
        rng.fill(vec, cv::RNG::UNIFORM, 0, 1);
        vec = vec / cv::norm(vec) * rng.uniform(0.0f, float(2 * CV_PI));
        cv::Mat R;
        cv::Rodrigues(vec, R);
        int inl_size = 100, out_size = 100;
        int pts_size = inl_size + out_size;
        cv::Mat inliers, outliers, points3d;
        inliers.create(2, inl_size, CV_64F);
        outliers.create(3, out_size, inliers.type());
        rng.fill(inliers, cv::RNG::UNIFORM, 0, 1);
        rng.fill(outliers, cv::RNG::UNIFORM, 0, 1);
        // planar points
        cv::vconcat(inliers, cv::Mat::ones(1, inl_size, inliers.type()), inliers);
        cv::hconcat(inliers, outliers, points3d);

        cv::Vec3d t = cv::Vec3d(rng.uniform(-0.5f, 0.5f), rng.uniform(-0.5f, 0.5f), rng.uniform(1.0f, 2.0f));
        cv::Mat tvec = (cv::Mat_<double>(3,1) << t(0), t(1), t(2));
        cv::Matx33d K = cv::Matx33d::zeros();
        K(0, 0) = rng.uniform(100, 1000);
        K(1, 1) = rng.uniform(100, 1000);
        K(0, 2) = rng.uniform(-100, 100);
        K(1, 2) = rng.uniform(-100, 100);
        K(2, 2) = 1;

        cv::Mat pts1 = K * points3d;
        cv::Mat pts2 = K * (R * points3d + t * cv::Mat::ones(1, points3d.cols, points3d.type()));
        cv::divide(pts1.row(0), pts1.row(2), pts1.row(0));
        cv::divide(pts1.row(1), pts1.row(2), pts1.row(1));
        cv::divide(pts2.row(0), pts2.row(2), pts2.row(0));
        cv::divide(pts2.row(1), pts2.row(2), pts2.row(1));

        cv::Mat mask;
        const double thr = 2.;
        cv::Mat H = findHomography(pts1.rowRange(0,2).t(), pts2.rowRange(0,2).t(),
                        cv::USAC, thr, mask, 1000, 0.99);
        CV_Assert(!H.empty());

        int num_inliers = cv::countNonZero(mask);
        CV_Assert(num_inliers >= .8 * inl_size);

        cv::Mat pts1_3d, pts2_3d;
        cv::vconcat(pts1.rowRange(0,2), cv::Mat::ones(1, pts_size, pts1.type()), pts1_3d);
        cv::vconcat(pts2.rowRange(0,2), cv::Mat::ones(1, pts_size, pts2.type()), pts2_3d);

        cv::Mat pts2_est = H * pts1_3d, pts1_est = H.inv() * pts2_3d;
        cv::divide(pts2_est.row(0), pts2_est.row(2), pts2_est.row(0));
        cv::divide(pts2_est.row(1), pts2_est.row(2), pts2_est.row(1));

        cv::divide(pts1_est.row(0), pts1_est.row(2), pts1_est.row(0));
        cv::divide(pts1_est.row(1), pts1_est.row(2), pts1_est.row(1));

        cv::Mat diff2 = pts2.rowRange(0, 2) - pts2_est.rowRange(0,2);
        cv::Mat diff1 = pts1.rowRange(0, 2) - pts1_est.rowRange(0,2);

        const auto * const mask_ptr = mask.ptr<uchar>();
        for (int i = 0; i < pts_size; i++)
            if (mask_ptr[i]) {
                // CV_Assert(cv::norm(diff1.col(i)) < thr);
                CV_Assert(cv::norm(diff2.col(i)) < thr);
            }
    }
}

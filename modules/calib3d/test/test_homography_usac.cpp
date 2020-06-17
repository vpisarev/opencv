// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test {
   TEST(usac_Homography, accuracy) {
        const int num_pts = 22;
        double pts [num_pts * 4] =
                {91.5756, 439.012, 159.738, 388.957,
                307.711, 611.892, 257.551, 590.321,
                531.211, 499.496, 409.495, 516.323,
                424.878, 221.616, 422.832, 253.312,
                69.727, 353.533, 170.103, 298.957,
                217.151, 353.236, 266.283, 330.142,
                358.228, 295.223, 366.547, 305.375,
                266.12, 308.995, 309.387, 300.048,
                466.221, 477.328, 380.627, 487.531,
                259.118, 226.169, 327.665, 219.816,
                499.487, 193.696, 468.713, 244.616,
                31.5425, 627.094, 70.3884, 567.573,
                231.86, 571.773, 220.264, 541.758,
                324.057, 395.71, 319.998, 391.036,
                45.2243, 340.475, 157.284, 280.222,
                386.941, 136.943, 107.162, 201.487,
                103.157, 632.299, 118.421, 583.454,
                37.3541, 503.456, 105.351, 444.065,
                63.3359, 368.337, 161.435, 312.27,
                422.551, 293.356, 403.436, 316.365,
                102.968, 610.616, 124.512, 561.851,
                258.133, 367.283, 287.451, 352.357};

        cv::Mat pts_mat (num_pts, 4, CV_64F, pts);

        cv::Mat mask;
        const double thr = 2.;
        cv::Mat H = cv::findHomography(pts_mat.colRange(0,2), pts_mat.colRange(2,4),
                cv::USAC, thr, mask, 500, 0.99);
        CV_Assert(!H.empty());

        int num_inliers = cv::countNonZero(mask);
        CV_Assert(num_inliers > 0);

        cv::Mat pts1_3d;
        cv::vconcat(pts_mat.colRange(0,2).t(), cv::Mat::ones(1, num_pts, pts_mat.type()), pts1_3d);
        cv::Mat pts2_est = H * pts1_3d;
        cv::divide(pts2_est.row(0), pts2_est.row(2), pts2_est.row(0));
        cv::divide(pts2_est.row(1), pts2_est.row(2), pts2_est.row(1));

        cv::Mat diff = pts_mat.colRange(2,4) - pts2_est.rowRange(0,2).t();
        const auto * const mask_ptr = mask.ptr<uchar>();
        for (int i = 0; i < num_pts; i++)
            if (mask_ptr[i])
                CV_Assert(cv::norm(diff.row(i)) < thr);
    }
}

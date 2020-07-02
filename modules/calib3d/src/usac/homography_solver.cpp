// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../usac.hpp"
#ifdef HAVE_EIGEN
#include <Eigen/Eigen>
#endif

namespace cv { namespace usac {
class HomographyMinimalSolver4ptsGEMImpl : public HomographyMinimalSolver4ptsGEM {
private:
    const double * const points;
public:
    explicit HomographyMinimalSolver4ptsGEMImpl (const Mat &points_) :
        points ((double *) points_.data) {}

    int estimate (const std::vector<int>& sample, std::vector<Mat> &models) const override {
        /*
          IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.

          By downloading, copying, installing or using the software you agree to this license.
          If you do not agree to this license, do not download, install,
          copy or use the software.
                                  BSD 3-Clause License

         Copyright (C) 2014, Olexa Bilaniuk, Hamid Bazargani & Robert Laganiere, all rights reserved.

         Redistribution and use in source and binary forms, with or without modification,
         are permitted provided that the following conditions are met:

           * Redistribution's of source code must retain the above copyright notice,
             this list of conditions and the following disclaimer.

           * Redistribution's in binary form must reproduce the above copyright notice,
             this list of conditions and the following disclaimer in the documentation
             and/or other materials provided with the distribution.

           * The name of the copyright holders may not be used to endorse or promote products
             derived from this software without specific prior written permission.

         This software is provided by the copyright holders and contributors "as is" and
         any express or implied warranties, including, but not limited to, the implied
         warranties of merchantability and fitness for a particular purpose are disclaimed.
         In no event shall the Intel Corporation or contributors be liable for any direct,
         indirect, incidental, special, exemplary, or consequential damages
         (including, but not limited to, procurement of substitute goods or services;
         loss of use, data, or profits; or business interruption) however caused
         and on any theory of liability, whether in contract, strict liability,
         or tort (including negligence or otherwise) arising in any way out of
         the use of this software, even if advised of the possibility of such damage.
        */

        /**
         * Bilaniuk, Olexa, Hamid Bazargani, and Robert Laganiere. "Fast Target
         * Recognition on Mobile Devices: Revisiting Gaussian Elimination for the
         * Estimation of Planar Homographies." In Computer Vision and Pattern
         * Recognition Workshops (CVPRW), 2014 IEEE Conference on, pp. 119-125.
         * IEEE, 2014.
         */

        const int smpl0 = 4*sample[0], smpl1 = 4*sample[1], smpl2 = 4*sample[2], smpl3 = 4*sample[3];
        const auto x0 = points[smpl0], y0 = points[smpl0+1], X0 = points[smpl0+2], Y0 = points[smpl0+3];
        const auto x1 = points[smpl1], y1 = points[smpl1+1], X1 = points[smpl1+2], Y1 = points[smpl1+3];
        const auto x2 = points[smpl2], y2 = points[smpl2+1], X2 = points[smpl2+2], Y2 = points[smpl2+3];
        const auto x3 = points[smpl3], y3 = points[smpl3+1], X3 = points[smpl3+2], Y3 = points[smpl3+3];
        const double x0X0 = x0*X0, x1X1 = x1*X1, x2X2 = x2*X2, x3X3 = x3*X3;
        const double x0Y0 = x0*Y0, x1Y1 = x1*Y1, x2Y2 = x2*Y2, x3Y3 = x3*Y3;
        const double y0X0 = y0*X0, y1X1 = y1*X1, y2X2 = y2*X2, y3X3 = y3*X3;
        const double y0Y0 = y0*Y0, y1Y1 = y1*Y1, y2Y2 = y2*Y2, y3Y3 = y3*Y3;

        double minor[2][4] = {{x0-x2, x1-x2, x2, x3-x2},
                              {y0-y2, y1-y2, y2, y3-y2}};

        double major[3][8] = {{x2X2-x0X0, x2X2-x1X1, -x2X2, x2X2-x3X3, x2Y2-x0Y0, x2Y2-x1Y1, -x2Y2, x2Y2-x3Y3},
                              {y2X2-y0X0, y2X2-y1X1, -y2X2, y2X2-y3X3, y2Y2-y0Y0, y2Y2-y1Y1, -y2Y2, y2Y2-y3Y3},
                              {X0-X2    , X1-X2    , X2   , X3-X2    , Y0-Y2    , Y1-Y2    , Y2   , Y3-Y2    }};

        /**
         * int i;
         * for(i=0;i<8;i++) major[2][i]=-major[2][i];
         * Eliminate column 0 of rows 1 and 3
         * R(1)=(x0-x2)*R(1)-(x1-x2)*R(0),     y1'=(y1-y2)(x0-x2)-(x1-x2)(y0-y2)
         * R(3)=(x0-x2)*R(3)-(x3-x2)*R(0),     y3'=(y3-y2)(x0-x2)-(x3-x2)(y0-y2)
         */

        double scalar1=minor[0][0], scalar2=minor[0][1];
        minor[1][1]=minor[1][1]*scalar1-minor[1][0]*scalar2;

        major[0][1]=major[0][1]*scalar1-major[0][0]*scalar2;
        major[1][1]=major[1][1]*scalar1-major[1][0]*scalar2;
        major[2][1]=major[2][1]*scalar1-major[2][0]*scalar2;

        major[0][5]=major[0][5]*scalar1-major[0][4]*scalar2;
        major[1][5]=major[1][5]*scalar1-major[1][4]*scalar2;
        major[2][5]=major[2][5]*scalar1-major[2][4]*scalar2;

        scalar2=minor[0][3];
        minor[1][3]=minor[1][3]*scalar1-minor[1][0]*scalar2;

        major[0][3]=major[0][3]*scalar1-major[0][0]*scalar2;
        major[1][3]=major[1][3]*scalar1-major[1][0]*scalar2;
        major[2][3]=major[2][3]*scalar1-major[2][0]*scalar2;

        major[0][7]=major[0][7]*scalar1-major[0][4]*scalar2;
        major[1][7]=major[1][7]*scalar1-major[1][4]*scalar2;
        major[2][7]=major[2][7]*scalar1-major[2][4]*scalar2;

        /**
         * Eliminate column 1 of rows 0 and 3
         * R(3)=y1'*R(3)-y3'*R(1)
         * R(0)=y1'*R(0)-(y0-y2)*R(1)
         */

        scalar1=minor[1][1];scalar2=minor[1][3];
        major[0][3]=major[0][3]*scalar1-major[0][1]*scalar2;
        major[1][3]=major[1][3]*scalar1-major[1][1]*scalar2;
        major[2][3]=major[2][3]*scalar1-major[2][1]*scalar2;

        major[0][7]=major[0][7]*scalar1-major[0][5]*scalar2;
        major[1][7]=major[1][7]*scalar1-major[1][5]*scalar2;
        major[2][7]=major[2][7]*scalar1-major[2][5]*scalar2;

        scalar2=minor[1][0];
        minor[0][0]=minor[0][0]*scalar1-minor[0][1]*scalar2;

        major[0][0]=major[0][0]*scalar1-major[0][1]*scalar2;
        major[1][0]=major[1][0]*scalar1-major[1][1]*scalar2;
        major[2][0]=major[2][0]*scalar1-major[2][1]*scalar2;

        major[0][4]=major[0][4]*scalar1-major[0][5]*scalar2;
        major[1][4]=major[1][4]*scalar1-major[1][5]*scalar2;
        major[2][4]=major[2][4]*scalar1-major[2][5]*scalar2;

        /**
         * Eliminate columns 0 and 1 of row 2
         * R(0)/=x0'
         * R(1)/=y1'
         * R(2)-= (x2*R(0) + y2*R(1))
         */

        scalar1=1.0f/minor[0][0];
        major[0][0]*=scalar1;
        major[1][0]*=scalar1;
        major[2][0]*=scalar1;
        major[0][4]*=scalar1;
        major[1][4]*=scalar1;
        major[2][4]*=scalar1;

        scalar1=1.0f/minor[1][1];
        major[0][1]*=scalar1;
        major[1][1]*=scalar1;
        major[2][1]*=scalar1;
        major[0][5]*=scalar1;
        major[1][5]*=scalar1;
        major[2][5]*=scalar1;

        scalar1=minor[0][2];scalar2=minor[1][2];
        major[0][2]-=major[0][0]*scalar1+major[0][1]*scalar2;
        major[1][2]-=major[1][0]*scalar1+major[1][1]*scalar2;
        major[2][2]-=major[2][0]*scalar1+major[2][1]*scalar2;

        major[0][6]-=major[0][4]*scalar1+major[0][5]*scalar2;
        major[1][6]-=major[1][4]*scalar1+major[1][5]*scalar2;
        major[2][6]-=major[2][4]*scalar1+major[2][5]*scalar2;

        /* Only major matters now. R(3) and R(7) correspond to the hollowed-out rows. */
        scalar1=major[0][7];
        major[1][7]/=scalar1;
        major[2][7]/=scalar1;
        const double m17 = major[1][7], m27 = major[2][7];
        scalar1=major[0][0];major[1][0]-=scalar1*m17;major[2][0]-=scalar1*m27;
        scalar1=major[0][1];major[1][1]-=scalar1*m17;major[2][1]-=scalar1*m27;
        scalar1=major[0][2];major[1][2]-=scalar1*m17;major[2][2]-=scalar1*m27;
        scalar1=major[0][3];major[1][3]-=scalar1*m17;major[2][3]-=scalar1*m27;
        scalar1=major[0][4];major[1][4]-=scalar1*m17;major[2][4]-=scalar1*m27;
        scalar1=major[0][5];major[1][5]-=scalar1*m17;major[2][5]-=scalar1*m27;
        scalar1=major[0][6];major[1][6]-=scalar1*m17;major[2][6]-=scalar1*m27;

        /* One column left (Two in fact, but the last one is the homography) */
        major[2][3]/=major[1][3];
        const double m23 = major[2][3];

        major[2][0]-=major[1][0]*m23;
        major[2][1]-=major[1][1]*m23;
        major[2][2]-=major[1][2]*m23;
        major[2][4]-=major[1][4]*m23;
        major[2][5]-=major[1][5]*m23;
        major[2][6]-=major[1][6]*m23;
        major[2][7]-=major[1][7]*m23;

        // check if homography does not contain NaN values
        for (int i = 0; i < 8; i++)
            if (std::isnan(major[2][i])) return 0;

        /* Homography is done. */
        models = std::vector<Mat>(1, Mat_<double>(3,3));
        auto * H_ = (double *) models[0].data;
        H_[0]=major[2][0];
        H_[1]=major[2][1];
        H_[2]=major[2][2];

        H_[3]=major[2][4];
        H_[4]=major[2][5];
        H_[5]=major[2][6];

        H_[6]=major[2][7];
        H_[7]=major[2][3];
        H_[8]=1.0;

        return 1;
    }

    int getMaxNumberOfSolutions () const override { return 1; }
    int getSampleSize() const override { return 4; }
};

Ptr<HomographyMinimalSolver4ptsGEM> HomographyMinimalSolver4ptsGEM::create(const Mat &points_) {
    return makePtr<HomographyMinimalSolver4ptsGEMImpl>(points_);
}

class HomographyNonMinimalSolverImpl : public HomographyNonMinimalSolver {
private:
    const Ptr<NormTransform> normTr;
public:
    explicit HomographyNonMinimalSolverImpl (const Mat &points_) :
        normTr (NormTransform::create(points_)) {}

    /*
     * Find Homography matrix using (weighted) non-minimal estimation.
     * Use Principal Component Analysis. Use normalized points.
     */
    int estimate (const std::vector<int>& sample, int sample_size, std::vector<Mat> &models,
            const std::vector<double>& weights) const override {
        if (sample_size < getMinimumRequiredSampleSize())
            return 0;

        Mat T1, T2, norm_points_;
        normTr->getNormTransformation(norm_points_, sample, sample_size, T1, T2);

        /*
         * @norm_points is matrix 4 x inlier_size
         * @weights is vector of inliers_size
         * weights[i] is weight of i-th inlier
         */
        const auto * const norm_points = (double *) norm_points_.data;

        double a1[9] = {0, 0, -1, 0, 0, 0, 0, 0, 0},
               a2[9] = {0, 0, 0, 0, 0, -1, 0, 0, 0},
               AtA[81] = {0};

        if (weights.empty()) {
            for (int i = 0; i < sample_size; i++) {
                const int smpl = 4*i;
                const double x1 = norm_points[smpl  ], y1 = norm_points[smpl+1],
                             x2 = norm_points[smpl+2], y2 = norm_points[smpl+3];

                a1[0] = -x1;
                a1[1] = -y1;
                a1[6] = x2*x1;
                a1[7] = x2*y1;
                a1[8] = x2;

                a2[3] = -x1;
                a2[4] = -y1;
                a2[6] = y2*x1;
                a2[7] = y2*y1;
                a2[8] = y2;

                for (int j = 0; j < 9; j++)
                    for (int z = j; z < 9; z++)
                        AtA[j*9+z] += a1[j]*a1[z] + a2[j]*a2[z];
            }
        } else {
            for (int i = 0; i < sample_size; i++) {
                const int smpl = 4*i;
                const double weight = weights[i];
                const double x1 = norm_points[smpl  ], y1 = norm_points[smpl+1],
                        x2 = norm_points[smpl+2], y2 = norm_points[smpl+3];
                const double minus_weight_times_x1 = -weight * x1,
                             minus_weight_times_y1 = -weight * y1,
                                   weight_times_x2 =  weight * x2,
                                   weight_times_y2 =  weight * y2;

                a1[0] = minus_weight_times_x1;
                a1[1] = minus_weight_times_y1;
                a1[2] = -weight;
                a1[6] = weight_times_x2 * x1;
                a1[7] = weight_times_x2 * y1;
                a1[8] = weight_times_x2;

                a2[3] = minus_weight_times_x1;
                a2[4] = minus_weight_times_y1;
                a2[5] = -weight;
                a2[6] = weight_times_y2 * x1;
                a2[7] = weight_times_y2 * y1;
                a2[8] = weight_times_y2;

                for (int j = 0; j < 9; j++)
                    for (int z = j; z < 9; z++)
                        AtA[j*9+z] += a1[j]*a1[z] + a2[j]*a2[z];
            }
        }

        // copy symmetric part of covariance matrix
        for (int j = 1; j < 9; j++)
            for (int z = 0; z < j; z++)
                AtA[j*9+z] = AtA[z*9+j];

#ifdef HAVE_EIGEN
        Eigen::Matrix<double, 9, 9> cov (AtA);
        Mat H = Mat_<double>(3,3);
        Eigen::HouseholderQR<Eigen::Matrix<double, 9, 9>> qr(cov);
        Eigen::MatrixXd Q = qr.householderQ();
        if (Q.cols() != 9) return 0;
        // extract the last nullspace
        Eigen::Map<Eigen::Matrix<double, 9, 1>>((double *)H.data) = Q.col(8);
#else
        Matx<double, 9, 9> AtA_ (AtA);
        Mat D, Vt;
        eigen(AtA_, D, Vt);
//        Mat U, W, Vt;
//        SVD::compute(AtA_, W, U, Vt, SVD::FULL_UV + SVD::MODIFY_A);
        if (Vt.rows != 9) return 0;
        Mat H = Vt.row(Vt.rows-1).reshape(0 /* same num of channels*/, 3);
#endif

        models = std::vector<Mat>{ T2.inv() * H * T1 };
        return 1;
    }

    int getMinimumRequiredSampleSize() const override { return 4; }
    int getMaxNumberOfSolutions () const override { return 1; }
private:

};

Ptr<HomographyNonMinimalSolver> HomographyNonMinimalSolver::create(const Mat &points_) {
    return makePtr<HomographyNonMinimalSolverImpl>(points_);
}
}}
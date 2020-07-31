// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../usac.hpp"
#if defined(HAVE_EIGEN)
#include <Eigen/Eigen>
#include <Eigen/QR>
#elif defined(HAVE_LAPACK)
// #include <lapacke.h>
#include "opencv_lapack.h"
#endif


namespace cv { namespace usac {
class PnPMinimalSolver6PtsImpl : public PnPMinimalSolver6Pts {
private:
    const Mat * points_mat;
    const float * const points;
public:
    // linear 6 points required (11 equations)
    int getSampleSize() const override { return 6; }
    int getMaxNumberOfSolutions () const override { return 1; }

    explicit PnPMinimalSolver6PtsImpl (const Mat &points_) :
        points_mat(&points_), points ((float*)points_.data) {}
    /*
        DLT:
        d x = P X, x = (u, v, 1), X = (X, Y, Z, 1), P = K[R t]
        is 3x4 projection matrix with rows p1, p2, p3. d is depth

        u = p1^T X / p3^T X
        v = p2^T X / p3^T X

        (p1^T - u p3^T) X = 0
        (p2^T - v p3^T) X = 0

        (p11 - u p31) X + (p12 - u p32) Y + (p13 - u p33) Z + (p14 - u p34) = 0
        (p12 - v p31) X + (p22 - v p32) Y + (p23 - v p33) Z + (p24 - v p34) = 0

        [X, Y, Z, 1, 0, 0, 0, 0, -u X, -u Y, -u Z, -u] [p11]    [0]
        [0, 0, 0, 0, X, Y, Z, 1, -v X, -v Y, -v Z, -v] [p12]    [0]
        .                                                     = [0]
        .
        .                                              [p34]    [0]

        minimum 11 equations, each point gives 2 equation, so at least 6 points are required.

        @points is array Nx5
        u1 v1 X1 Y1 Z1
        ...
        uN vN XN YN ZN
        @P is output projection matrix

        A1 =
        [X1, Y1, Z1, 1, 0, 0, 0, 0, -u1 X1, -u1 Y1, -u1 Z1, -u1] [p11]    [0]
        [X2, Y2, Z2, 1, 0, 0, 0, 0, -u2 X2, -u2 Y2, -u2 Z2, -u2] [p12]    [0]
        [X3, Y3, Z3, 1, 0, 0, 0, 0, -u3 X3, -u3 Y3, -u3 Z3, -u3] [p13]    [0]
        [X4, Y4, Z4, 1, 0, 0, 0, 0, -u4 X4, -u4 Y4, -u4 Z4, -u4] [p14]    [0]
        [X5, Y5, Z5, 1, 0, 0, 0, 0, -u5 X5, -u5 Y5, -u5 Z5, -u5] [p21]    [0]
                                                                 [p22]
        A2 = (without first 4 columns)
        [0, 0, 0, 0, X1, Y1, Z1, 1, -v1 X1, -v1 Y1, -v1 Z1, -v1] [p23]  = [0]
        [0, 0, 0, 0, X2, Y2, Z2, 1, -v2 X2, -v2 Y2, -v2 Z2, -v2] [p24]    [0]
        [0, 0, 0, 0, X3, Y3, Z3, 1, -v3 X3, -v3 Y3, -v3 Z3, -v3] [p31]    [0]
        [0, 0, 0, 0, X4, Y4, Z4, 1, -v4 X4, -v4 Y4, -v4 Z4, -v4] [p32]    [0]
        [0, 0, 0, 0, X5, Y5, Z5, 1, -v5 X5, -v5 Y5, -v5 Z5, -v5] [p33]    [0]
        [0, 0, 0, 0, X6, Y6, Z6, 1, -v6 X6, -v6 Y6, -v6 Z6, -v6] [p34=1]  [0]

        P = null A; dim null A = n - rank(A) = 12 - 11 = 1
    */

    int estimate (const std::vector<int> &sample, std::vector<Mat> &models) const override {
        std::vector<double> A1 (5*12, 0), A2(7*8, 0);

        int cnt1 = 0, cnt2 = 0;
        for (int i = 0; i < 6; i++) {
            const int smpl = 5 * sample[i];
            const double u = points[smpl    ], v = points[smpl + 1];
            const double X = points[smpl + 2], Y = points[smpl + 3], Z = points[smpl + 4];

            if (i != 5) {
                A1[cnt1++] = X;
                A1[cnt1++] = Y;
                A1[cnt1++] = Z;
                A1[cnt1++] = 1;
                cnt1 += 4; // skip zeros
                A1[cnt1++] = -u * X;
                A1[cnt1++] = -u * Y;
                A1[cnt1++] = -u * Z;
                A1[cnt1++] = -u;
            }

            A2[cnt2++] = X;
            A2[cnt2++] = Y;
            A2[cnt2++] = Z;
            A2[cnt2++] = 1;
            A2[cnt2++] = -v * X;
            A2[cnt2++] = -v * Y;
            A2[cnt2++] = -v * Z;
            A2[cnt2++] = -v;
        }
        Math::eliminateUpperTriangluar(A1, 5, 12);

        int offset = 4*12;
        // add last eliminated row of A1
        for (int i = 0; i < 8; i++)
            A2[cnt2++] = A1[offset + i + 4/* skip 4 first cols*/];

        Math::eliminateUpperTriangluar(A2, 7, 8);
        // fixed scale to 1. In general the projection matrix is up-to-scale.
        // P = alpha * P^, alpha = 1 / P^_[3,4]

        Mat P = Mat_<double>(3,4);
        auto * p = (double *) P.data;
        p[11] = 1;

        // start from the last row
        for (int i = 6; i >= 0; i--) {
            double acc = 0;
            for (int j = i+1; j < 8; j++)
                acc -= A2[i*8+j]*p[j+4];

            p[i+4] = acc / A2[i*8+i];
            // due to numerical errors return 0 solutions
            if (std::isnan(p[i+4]))
                return 0;
        }

        for (int i = 3; i >= 0; i--) {
            double acc = 0;
            for (int j = i+1; j < 12; j++)
                acc -= A1[i*12+j]*p[j];

            p[i] = acc / A1[i*12+i];
            if (std::isnan(p[i]))
                return 0;
        }

        models = std::vector<Mat>{P};
        return 1;
    }
    Ptr<MinimalSolver> clone () const override {
        return makePtr<PnPMinimalSolver6PtsImpl>(*points_mat);
    }
};
Ptr<PnPMinimalSolver6Pts> PnPMinimalSolver6Pts::create(const Mat &points_) {
    return makePtr<PnPMinimalSolver6PtsImpl>(points_);
}

class PnPNonMinimalSolverImpl : public PnPNonMinimalSolver {
private:
    const Mat * points_mat;
    const float * const points;
public:
    explicit PnPNonMinimalSolverImpl (const Mat &points_) :
        points_mat(&points_), points ((float*)points_.data) {}

    int estimate (const std::vector<int> &sample, int sample_size,
          std::vector<Mat> &models, const std::vector<double> &/*weights*/) const override {
        double AtA [12*12] = {0};
        double a1[12] = {0, 0, 0, -1, 0, 0, 0,  0, 0, 0, 0, 0},
               a2[12] = {0, 0, 0,  0, 0, 0, 0, -1, 0, 0, 0, 0};

        for (int i = 0; i < sample_size; i++) {
            const int smpl = 5 * sample[i];
            const double u = points[smpl], v = points[smpl + 1];
            const double X = points[smpl + 2], Y = points[smpl + 3], Z = points[smpl + 4];

            a1[0 ] = -X;
            a1[1 ] = -Y;
            a1[2 ] = -Z;
            a1[8 ] = u * X;
            a1[9 ] = u * Y;
            a1[10] = u * Z;
            a1[11] = u;

            a2[4 ] = -X;
            a2[5 ] = -Y;
            a2[6 ] = -Z;
            a2[8 ] = v * X;
            a2[9 ] = v * Y;
            a2[10] = v * Z;
            a2[11] = v;

            // fill covarinace matrix
            for (int j = 0; j < 12; j++)
                for (int z = j; z < 12; z++)
                    AtA[j * 12 + z] += a1[j] * a1[z] + a2[j] * a2[z];
        }

        // copy symmetric part of covariance matrix
        for (int j = 1; j < 12; j++)
            for (int z = 0; z < j; z++)
                AtA[j*12+z] = AtA[z*12+j];

#ifdef HAVE_EIGEN
        Eigen::Matrix<double, 12, 12> cov (AtA);
        Eigen::HouseholderQR<Eigen::MatrixXd> qr(cov);
        Eigen::MatrixXd Q = qr.householderQ();
        if (Q.cols() != 12)
            return 0;
        // extract the last nullspace
        models = std::vector<Mat>(1, Mat_<double>(3,4));
        Eigen::Map<Eigen::Matrix<double, 12, 1>>((double *)models[0].data) = Q.col(11);
#else
        Matx<double, 12, 12> AtA_ (AtA), Vt;
        Vec<double, 10> D;
        if (! eigen(AtA_, D, Vt)) return 0;
        models = std::vector<Mat>{ Mat(Vt.row(11).reshape<3,4>()) };
#endif
        return 1;
    }

    int getMinimumRequiredSampleSize() const override { return 6; }
    int getMaxNumberOfSolutions () const override { return 1; }
    Ptr<NonMinimalSolver> clone () const override {
        return makePtr<PnPNonMinimalSolverImpl>(*points_mat);
    }
};
Ptr<PnPNonMinimalSolver> PnPNonMinimalSolver::create(const Mat &points_) {
    return makePtr<PnPNonMinimalSolverImpl>(points_);
}

class P3PSolverImpl : public P3PSolver {
private:
    /*
     * calibrated normalized points
     * K^-1 [u v 1]^T / ||K^-1 [u v 1]^T||
     */
    const Mat * points_mat, * calib_norm_points_mat, * K_mat;
    const Mat &K;
    const float * const calib_norm_points;
    const float * const points;
    const double VAL_THR = 1e-4;
public:
    /*
     * @points_ is matrix N x 5
     * u v x y z. (u,v) is image point, (x y z) is world point
     */
    P3PSolverImpl (const Mat &points_, const Mat &calib_norm_points_, const Mat &K_) :
        points_mat(&points_), calib_norm_points_mat(&calib_norm_points_), K_mat (&K_),
        K(K_), calib_norm_points((float*)calib_norm_points_.data), points((float*)points_.data) {}

    int estimate (const std::vector<int> &sample, std::vector<Mat> &models) const override {
        /*
         * The description of this solution can be found here:
         * http://cmp.felk.cvut.cz/~pajdla/gvg/GVG-2016-Lecture.pdf
         * pages: 51-59
         */
        const int   idx1 = 5*sample[0],   idx2 = 5*sample[1],   idx3 = 5*sample[2];
        const int c_idx1 = 3*sample[0], c_idx2 = 3*sample[1], c_idx3 = 3*sample[2];

        // find distance between world points d_ij = ||Xi - Xj||
        const double d12 = sqrt(pow(points[idx1+2] - points[idx2+2], 2) +
                                pow(points[idx1+3] - points[idx2+3], 2) +
                                pow(points[idx1+4] - points[idx2+4], 2));
        const double d23 = sqrt(pow(points[idx2+2] - points[idx3+2], 2) +
                                pow(points[idx2+3] - points[idx3+3], 2) +
                                pow(points[idx2+4] - points[idx3+4], 2));
        const double d31 = sqrt(pow(points[idx3+2] - points[idx1+2], 2) +
                                pow(points[idx3+3] - points[idx1+3], 2) +
                                pow(points[idx3+4] - points[idx1+4], 2));

        if (d12 < VAL_THR || d23 < VAL_THR || d31 < VAL_THR)
            return 0;

        // find cosine angles, cos(x1,x2) = K^-1 x1.dot(K^-1 x2) / (||K^-1 x1|| * ||K^-1 x2||)
        // calib_norm_points are already K^-1 x / ||K^-1 x||, so we perform only dot product
        const double c12 = calib_norm_points[c_idx1  ] * calib_norm_points[c_idx2  ] +
                           calib_norm_points[c_idx1+1] * calib_norm_points[c_idx2+1] +
                           calib_norm_points[c_idx1+2] * calib_norm_points[c_idx2+2];
        const double c23 = calib_norm_points[c_idx2  ] * calib_norm_points[c_idx3  ] +
                           calib_norm_points[c_idx2+1] * calib_norm_points[c_idx3+1] +
                           calib_norm_points[c_idx2+2] * calib_norm_points[c_idx3+2];
        const double c31 = calib_norm_points[c_idx3  ] * calib_norm_points[c_idx1  ] +
                           calib_norm_points[c_idx3+1] * calib_norm_points[c_idx1+1] +
                           calib_norm_points[c_idx3+2] * calib_norm_points[c_idx1+2];

        const double c12_p2 = c12*c12, c23_p2 = c23*c23, c31_p2 = c31*c31;
        const double d12_p2 = d12*d12, d12_p4 = d12_p2*d12_p2;
        const double d23_p2 = d23*d23, d23_p4 = d23_p2*d23_p2, d23_p6 = d23_p2*d23_p4, d23_p8 = d23_p4*d23_p4;
        const double d31_p2 = d31*d31, d31_p4 = d31_p2*d31_p2;
        const double a4 = -4*d23_p4*d12_p2*d31_p2*c23_p2+d23_p8-2*d23_p6*d12_p2-2*d23_p6*d31_p2+d23_p4*d12_p4+2*d23_p4*d12_p2*d31_p2+d23_p4*d31_p4;
        const double a3 = 8*d23_p4*d12_p2*d31_p2*c12*c23_p2+4*d23_p6*d12_p2*c31*c23-4*d23_p4*d12_p4*c31*c23+4*d23_p4*d12_p2*d31_p2*c31*c23-4*d23_p8*c12+4*d23_p6*d12_p2*c12+8*d23_p6*d31_p2*c12-4*d23_p4*d12_p2*d31_p2*c12-4*d23_p4*d31_p4*c12;
        const double a2 = -8*d23_p6*d12_p2*c31*c12*c23-8*d23_p4*d12_p2*d31_p2*c31*c12*c23+4*d23_p8*c12_p2-4*d23_p6*d12_p2*c31_p2-8*d23_p6*d31_p2*c12_p2+4*d23_p4*d12_p4*c31_p2+4*d23_p4*d12_p4*c23_p2-4*d23_p4*d12_p2*d31_p2*c23_p2+4*d23_p4*d31_p4*c12_p2+2*d23_p8-4*d23_p6*d31_p2-2*d23_p4*d12_p4+2*d23_p4*d31_p4;
        const double a1 = 8*d23_p6*d12_p2*c31_p2*c12+4*d23_p6*d12_p2*c31*c23-4*d23_p4*d12_p4*c31*c23+4*d23_p4*d12_p2*d31_p2*c31*c23-4*d23_p8*c12-4*d23_p6*d12_p2*c12+8*d23_p6*d31_p2*c12+4*d23_p4*d12_p2*d31_p2*c12-4*d23_p4*d31_p4*c12;
        const double a0 = -4*d23_p6*d12_p2*c31_p2+d23_p8-2*d23_p4*d12_p2*d31_p2+2*d23_p6*d12_p2+d23_p4*d31_p4+d23_p4*d12_p4-2*d23_p6*d31_p2;

        // a4 x^4 + ... + a0 = 0
#ifdef HAVE_EIGEN
        // create companion matrix http://web.mit.edu/18.06/www/Spring17/Eigenvalue-Polynomials.pdf
        Eigen::Matrix<double, 4, 4> companion_mat = Eigen::Matrix<double, 4, 4>::Zero();
        companion_mat(0,1) = 1;
        companion_mat(1,2) = 1;
        companion_mat(2,3) = 1;
        companion_mat(3, 0) = -a0 / a4;
        companion_mat(3, 1) = -a1 / a4;
        companion_mat(3, 2) = -a2 / a4;
        companion_mat(3, 3) = -a3 / a4;
        Eigen::EigenSolver<Eigen::Matrix<double, 4, 4>> eigensolver(companion_mat);
        const Eigen::VectorXcd& eigenvalues = eigensolver.eigenvalues();
#elif defined(HAVE_LAPACK)
        int info, mat_order = 4, lda = 4, ldvl = 1, ldvr = 1, lwork = 16;
        double act_m[16] = {0}, wr[4], wi[4] = {0}, work[16]; // 4 = mat_order, 16 = lwork
        act_m[1] = 1; act_m[6] = 1; act_m[11] = 1;
        act_m[12] = -a0 / a4; act_m[13] = -a1 / a4; act_m[14] = -a2 / a4; act_m[15] = -a3 / a4;
        char jobvl = 'N', jobvr = 'N';// eigen vectors are not computed
        dgeev_(&jobvl, &jobvr, &mat_order, act_m, &lda, wr, wi, nullptr, &ldvl, nullptr, &ldvr, work, &lwork, &info);
        if (info != 0) return 0;
#else
        Mat_<double> coeffs(1, 5);
        auto * coeffs_ = (double *) coeffs.data;
        coeffs_[0] = a0;
        coeffs_[1] = a1;
        coeffs_[2] = a2;
        coeffs_[3] = a3;
        coeffs_[4] = a4;
        std::vector<Complex<double>> roots;
        //\texttt{coeffs} [n] x^{n} +  \texttt{coeffs} [n-1] x^{n-1} + ... +  \texttt{coeffs} [1] x +  \texttt{coeffs} [0] = 0\f]
        solvePoly(coeffs, roots);
#endif
        models = std::vector<Mat>(); models.reserve(4);
        for (int r = 0; r < 4; r++) {
#ifdef HAVE_EIGEN
            if (eigenvalues(r).imag() != 0)
                continue;
            const double n12 = eigenvalues(r).real(), n12_p2 = n12*n12;
#elif defined(HAVE_LAPACK)
            if (wi[i] == 0)
                continue;
            const double n12 = wr[r], n12_p2 = n12*n12;
#else
            if (r >= (int) roots.size())
                break;
            if (fabs(roots[r].im) > 1e-10)
                continue;
            const double n12 = roots[r].re, n12_p2 = n12*n12;
#endif

            const double n13 = (d12_p2*(d23_p2-d31_p2*n12_p2)+(d23_p2-d31_p2)*(d23_p2*(1+n12_p2-2*n12*c12)-d12_p2*n12_p2))
                                / (2*d12_p2*(d23_p2*c31 - d31_p2*c23*n12) + 2*(d31_p2-d23_p2)*d12_p2*c23*n12);

            const double n1 = d12 / sqrt(1 + n12_p2 - 2*n12*c12);
            if (std::isnan(n1))
                continue;
            const double n2 = n1 * n12;
            const double n3 = n1 * n13;

            if (n1 <= 0 || n2 <= 0 || n3 <= 0)
                continue;
            // compute errors
            const double e1 = (sqrt(n1*n1 + n2*n2 - 2*n1*n2*c12) - d12) / d12;
            const double e2 = (sqrt(n2*n2 + n3*n3 - 2*n2*n3*c23) - d23) / d23;
            const double e3 = (sqrt(n3*n3 + n1*n1 - 2*n3*n1*c31) - d31) / d31;

            if (fabs(e1) > VAL_THR || fabs(e2) > VAL_THR || fabs(e3) > VAL_THR)
                continue;

            const Vec3d nX1 (n1*calib_norm_points[c_idx1], n1*calib_norm_points[c_idx1+1], n1*calib_norm_points[c_idx1+2]);
            const Vec3d nX2 (n2*calib_norm_points[c_idx2], n2*calib_norm_points[c_idx2+1], n2*calib_norm_points[c_idx2+2]);
            const Vec3d nX3 (n3*calib_norm_points[c_idx3], n3*calib_norm_points[c_idx3+1], n3*calib_norm_points[c_idx3+2]);

            const Vec3d X1 (points[idx1+2], points[idx1+3], points[idx1+4]);
            const Vec3d X2 (points[idx2+2], points[idx2+3], points[idx2+4]);
            const Vec3d X3 (points[idx3+2], points[idx3+3], points[idx3+4]);

            Vec3d Z2 = nX2 - nX1; Z2 /= norm(Z2);
            Vec3d Z3 = nX3 - nX1; Z3 /= norm(Z3);
            Vec3d Z1 = Z2.cross(Z3); Z1 /= norm(Z1);

            Mat Z;
            hconcat(Z1, Z2, Z);
            hconcat(Z, Z3.cross(Z1), Z);

            Vec3d Zw2 = X2 - X1; Zw2 /= d12;
            Vec3d Zw3 = X3 - X1; Zw3 /= d31;
            Vec3d Zw1 = Zw2.cross(Zw3); Zw1 /= norm(Zw1);

            Mat Zw;
            hconcat(Zw1, Zw2, Zw);
            hconcat(Zw, Zw3.cross(Zw1), Zw);

            Mat R = Z * Zw.inv();
            Mat x1 = (Mat_<double>(3,1) << X1(0), X1(1), X1(2));
            Mat C = x1 - R.t() * nX1;

            Mat P, KR = K * R;
            hconcat(KR, -KR * C, P);
            models.emplace_back(P);
        }
        return static_cast<int>(models.size());
    }
    int getSampleSize() const override { return 3; }
    int getMaxNumberOfSolutions () const override { return 4; }
    Ptr<MinimalSolver> clone () const override {
        return makePtr<P3PSolverImpl>(*points_mat, *calib_norm_points_mat, *K_mat);
    }
};
Ptr<P3PSolver> P3PSolver::create(const Mat &points_, const Mat &calib_norm_pts, const Mat &K) {
    return makePtr<P3PSolverImpl>(points_, calib_norm_pts, K);
}
}}
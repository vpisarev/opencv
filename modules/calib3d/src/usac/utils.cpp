// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../usac.hpp"

namespace cv { namespace usac {
double Utils::getCalibratedThreshold (double threshold, const Mat &K1, const Mat &K2) {
    return threshold / ((K1.at<double>(0, 0) + K1.at<double>(1, 1) +
                         K2.at<double>(0, 0) + K2.at<double>(1, 1)) / 4.0);
}

/*
 * K1, K2 are 3x3 intrinsics matrices
 * points is matrix of size |N| x 4
 * Assume K = [k11 k12 k13
 *              0  k22 k23
 *              0   0   1]
 */
void Utils::calibratePoints (const Mat &K1, const Mat &K2, const Mat &points, Mat &norm_points) {
    const auto * const points_ = (float *) points.data;
    const auto * const k1 = (double *) K1.data;
    const auto inv1_k11 = float(1 / k1[0]); // 1 / k11
    const auto inv1_k12 = float(-k1[1] / (k1[0]*k1[4])); // -k12 / (k11*k22)
    // (-k13*k22 + k12*k23) / (k11*k22)
    const auto inv1_k13 = float((-k1[2]*k1[4] + k1[1]*k1[5]) / (k1[0]*k1[4]));
    const auto inv1_k22 = float(1 / k1[4]); // 1 / k22
    const auto inv1_k23 = float(-k1[5] / k1[4]); // -k23 / k22

    const auto * const k2 = (double *) K2.data;
    const auto inv2_k11 = float(1 / k2[0]);
    const auto inv2_k12 = float(-k2[1] / (k2[0]*k2[4]));
    const auto inv2_k13 = float((-k2[2]*k2[4] + k2[1]*k2[5]) / (k2[0]*k2[4]));
    const auto inv2_k22 = float(1 / k2[4]);
    const auto inv2_k23 = float(-k2[5] / k2[4]);

    const int num_pts = points.rows;
    norm_points = Mat (num_pts, 4, points.type());
    auto * norm_points_ = (float *) norm_points.data;

    for (int i = 0; i < num_pts; i++) {
        const int idx = 4*i;
        (*norm_points_++) = inv1_k11 * points_[idx  ] + inv1_k12 * points_[idx+1] + inv1_k13;
        (*norm_points_++) =                             inv1_k22 * points_[idx+1] + inv1_k23;
        (*norm_points_++) = inv2_k11 * points_[idx+2] + inv2_k12 * points_[idx+3] + inv2_k13;
        (*norm_points_++) =                             inv2_k22 * points_[idx+3] + inv2_k23;
    }
}

/*
 * K is 3x3 intrinsic matrix
 * points is matrix of size |N| x 5, first two columns are image points [u_i, v_i]
 * calib_norm_pts are  K^-1 [u v 1]^T / ||K^-1 [u v 1]^T||
 */
void Utils::calibrateAndNormalizePointsPnP (const Mat &K, const Mat &pts, Mat &calib_norm_pts) {
    const auto * const points = (float *) pts.data;
    const auto * const k = (double *) K.data;
    const auto inv_k11 = float(1 / k[0]);
    const auto inv_k12 = float(-k[1] / (k[0]*k[4]));
    const auto inv_k13 = float((-k[2]*k[4] + k[1]*k[5]) / (k[0]*k[4]));
    const auto inv_k22 = float(1 / k[4]);
    const auto inv_k23 = float(-k[5] / k[4]);

    const int num_pts = pts.rows;
    calib_norm_pts = Mat (num_pts, 3, pts.type());
    auto * calib_norm_pts_ = (float *) calib_norm_pts.data;

    for (int i = 0; i < num_pts; i++) {
        const int idx = 5 * i;
        const float k_inv_u = inv_k11 * points[idx] + inv_k12 * points[idx+1] + inv_k13;
        const float k_inv_v =                         inv_k22 * points[idx+1] + inv_k23;
        const float norm = 1.f / sqrtf(powf(k_inv_u, 2) + powf(k_inv_v, 2) + 1);
        (*calib_norm_pts_++) = k_inv_u * norm;
        (*calib_norm_pts_++) = k_inv_v * norm;
        (*calib_norm_pts_++) =           norm;
    }
}

/*
 * decompose Projection Matrix to calibration, rotation and translation
 * Assume K = [fx  0   tx
 *             0   fy  ty
 *             0   0   1]
 */
void Utils::decomposeProjection (const Mat &P, Mat &K_, Mat &R, Mat &t) {
    const Mat M = P.colRange(0,3);
    double scale = norm(M.row(2)); scale *= scale;
    Matx33d K = Matx33d::eye();
    K(1,2) = M.row(1).dot(M.row(2)) / scale;
    K(0,2) = M.row(0).dot(M.row(2)) / scale;
    K(1,1) = sqrt(M.row(1).dot(M.row(1)) / scale - K(1,2)*K(1,2));
    K(0,0) = sqrt(M.row(0).dot(M.row(0)) / scale - K(0,2)*K(0,2));
    R = K.inv() * M / sqrt(scale);
    if (determinant(M) < 0) R *= -1;
    t = R * M.inv() * P.col(3);
    K_ = Mat(K);
}

Matx33d Math::getSkewSymmetric(const Vec3d &v) {
     return Matx33d(0,    -v[2], v[1],
                   v[2],  0,    -v[0],
                  -v[1],  v[0], 0);
}

/*
 * Eliminate matrix of m rows and n columns to be upper triangular.
 */
void Math::eliminateUpperTriangluar (std::vector<double> &a, int m, int n) {
    for (int r = 0; r < m; r++){
        double pivot = a[r*n+r];
        int row_with_pivot = r;

        // find the maximum pivot value among r-th column
        for (int k = r+1; k < m; k++)
            if (fabs(pivot) < fabs(a[k*n+r])) {
                pivot = a[k*n+r];
                row_with_pivot = k;
            }

        // if pivot value is 0 continue
        if (fabs(pivot) < DBL_EPSILON)
            continue;

        // swap row with maximum pivot value with current row
        for (int c = r; c < n; c++)
            std::swap(a[row_with_pivot*n+c], a[r*n+c]);

        // eliminate other rows
        for (int j = r+1; j < m; j++){
            const auto fac = a[j*n+r] / pivot;
            for (int c = r; c < n; c++)
                a[j*n+c] -= fac * a[r*n+c];
        }
    }
}

//////////////////////////////////////// RANDOM GENERATOR /////////////////////////////
class UniformRandomGeneratorImpl : public UniformRandomGenerator {
private:
    int subset_size = 0, max_range = 0;
    RNG rng;
public:
    explicit UniformRandomGeneratorImpl (int state) : rng(state) {}

    // interval is <0; max_range);
    UniformRandomGeneratorImpl (int state, int max_range_, int subset_size_) : rng(state) {
        CV_CheckGT(subset_size_, 0, "UniformRandomGenerator. Subset size must be higher than 0!");
        CV_CheckLE(subset_size_, max_range_, "RandomGenerator. Subset size must be LE than range!");
        subset_size = subset_size_;
        max_range = max_range_;
    }

    int getRandomNumber () override {
        return rng.uniform(0, max_range);
    }

    int getRandomNumber (int max_rng) override {
        return rng.uniform(0, max_rng);
    }

    // closed range
    void resetGenerator (int max_range_) override {
        CV_CheckGE(0, max_range_, "max range must be greater than 0");
        max_range = max_range_;
    }

    void generateUniqueRandomSet (std::vector<int>& sample) override {
        int j, num;
        sample[0] = rng.uniform(0, max_range);
        for (int i = 1; i < subset_size;) {
            num = rng.uniform(0, max_range);
            // check if value is in array
            for (j = i - 1; j >= 0; j--)
                if (num == sample[j])
                    // if so, generate again
                    break;
            // success, value is not in array, so it is unique, add to sample.
            if (j == -1) sample[i++] = num;
        }
    }

    // interval is <0; max_range)
    void generateUniqueRandomSet (std::vector<int>& sample, int max_range_) override {
        /*
         * necessary condition:
         * if subset size is bigger than range then array cannot be unique,
         * so function has infinite loop.
         */
        CV_CheckLE(subset_size, max_range_, "RandomGenerator. Subset size must be LE than range!");
        int num, j;
        sample[0] = rng.uniform(0, max_range_);
        for (int i = 1; i < subset_size;) {
            num = rng.uniform(0, max_range_);
            for (j = i - 1; j >= 0; j--)
                if (num == sample[j])
                    break;
            if (j == -1) sample[i++] = num;
        }
    }

    // interval is <0, max_range)
    void generateUniqueRandomSet (std::vector<int>& sample, int subset_size_, int max_range_) override {
        CV_CheckLE(subset_size_, max_range_, "RandomGenerator. Subset size must be LE than range!");
        int num, j;
        sample[0] = rng.uniform(0, max_range_);
        for (int i = 1; i < subset_size_;) {
            num = rng.uniform(0, max_range_);
            for (j = i - 1; j >= 0; j--)
                if (num == sample[j])
                    break;
            if (j == -1) sample[i++] = num;
        }
    }

    void setSubsetSize (int subset_size_) override {
        subset_size = subset_size_;
    }
};

Ptr<UniformRandomGenerator> UniformRandomGenerator::create (int state) {
    return Ptr<UniformRandomGeneratorImpl>(new UniformRandomGeneratorImpl(state));
}
Ptr<UniformRandomGenerator> UniformRandomGenerator::create
        (int state, int max_range, int subset_size_) {
    return Ptr<UniformRandomGeneratorImpl>(
            new UniformRandomGeneratorImpl(state, max_range, subset_size_));
}
}}
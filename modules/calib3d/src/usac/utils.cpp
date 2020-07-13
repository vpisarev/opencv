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
    const double inv1_k11 = 1 / k1[0]; // 1 / k11
    const double inv1_k12 = -k1[1] / (k1[0]*k1[4]); // -k12 / (k11*k22)
    const double inv1_k13 = (-k1[2]*k1[4] + k1[1]*k1[5]) / (k1[0]*k1[4]); // (-k13*k22 + k12*k23) / (k11*k22)
    const double inv1_k22 = 1 / k1[4]; // 1 / k22
    const double inv1_k23 = -k1[5] / k1[4]; // -k23 / k22

    const auto * const k2 = (double *) K2.data;
    const double inv2_k11 = 1 / k2[0];
    const double inv2_k12 = -k2[1] / (k2[0]*k2[4]);
    const double inv2_k13 = (-k2[2]*k2[4] + k2[1]*k2[5]) / (k2[0]*k2[4]);
    const double inv2_k22 = 1 / k2[4];
    const double inv2_k23 = -k2[5] / k2[4];

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

////////////////////////////////////////////////////////////////////////
bool Math::haveCollinearPoints(const Mat &points_, const std::vector<int>& sample,
                                      double threshold) {
    const auto * const points = (float *) points_.data;
    // Checks if no more than 2 points are on the same line
    // If area of triangle constructed with 3 points is less then threshold then points are collinear:
    //           |x1 y1 1|             |x1      y1      1|
    // (1/2) det |x2 y2 1| = (1/2) det |x2-x1   y2-y1   0| = (1/2) det |x2-x1   y2-y1| < threshold
    //           |x3 y3 1|             |x3-x1   y3-y1   0|             |x3-x1   y3-y1|
    double x1, y1, x2, y2, x3, y3, X1, Y1, X2, Y2, X3, Y3;
    int pt_idx, sample_size = static_cast<int>(sample.size());
    for (int i1 = 0; i1 < sample_size-2; i1++) {
        pt_idx = 4*sample[i1];
        x1 = points[pt_idx  ]; y1 = points[pt_idx+1];
        X1 = points[pt_idx+2]; Y1 = points[pt_idx+3];

        for (int i2 = i1+1; i2 < sample_size-1; i2++){
            pt_idx = 4*sample[i2];
            x2 = points[pt_idx  ]; y2 = points[pt_idx+1];
            X2 = points[pt_idx+2]; Y2 = points[pt_idx+3];

            for (int i3 = i2+1; i3 < sample_size; i3++) {
                pt_idx = 4*sample[i3];
                x3 = points[pt_idx  ]; y3 = points[pt_idx+1];
                X3 = points[pt_idx+2]; Y3 = points[pt_idx+3];

                if (fabs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)) / 2 < threshold)
                    return true;
                if (fabs((X2 - X1) * (Y3 - Y1) - (Y2 - Y1) * (X3 - X1)) / 2 < threshold)
                    return true;
            }
        }
    }
    return false;
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
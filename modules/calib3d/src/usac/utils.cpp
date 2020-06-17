// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../usac.hpp"

/*
    Sort points by their density:
    Find knn nearest neighbors. Compare by sum of the k closest points.
*/
namespace cv { namespace usac {
// Performs Fisher-Yates shuffle
void Utils::random_shuffle (RNG &rng, std::vector<int>& array) {
    int rand_idx, temp_size = array.size(), array_size = array.size();
    for (int i = 0; i < array_size; i++) {
        rand_idx = rng.uniform(0, temp_size); // get random index of array of working interval

        // decrease temp size and swap values of random index to the end (temp size)
        std::swap(array[rand_idx], array[--temp_size]);
    }
}

////////////////////////////////////////////////////////////////////////

bool Math::haveCollinearPoints(const Mat &points_, const std::vector<int>& sample,
                                      double threshold) {
    const auto * const points = (double *) points_.data;
    // Checks if no more than 2 points are on the same line
    // If area of triangle constructed with 3 points is less then threshold then points are collinear:
    //           |x1 y1 1|             |x1      y1      1|
    // (1/2) det |x2 y2 1| = (1/2) det |x2-x1   y2-y1   0| = (1/2) det |x2-x1   y2-y1| < threshold
    //           |x3 y3 1|             |x3-x1   y3-y1   0|             |x3-x1   y3-y1|
    double x1, y1, x2, y2, x3, y3, X1, Y1, X2, Y2, X3, Y3;
    int pt_idx, sample_size = sample.size();
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

double Math::getMedianNaive (const std::vector<double> &v) {
    std::vector<double> work = v;
    std::sort (work.begin(), work.end());
    if (v.size () % 2 == 0)
        return (work[work.size() / 2] + work[work.size() / 2-1]) / 2;
    else
        return work[work.size() / 2];

}

Mat Math::getSkewSymmetric(const Mat &v_) {
    const auto * const v = (double *) v_.data;
    return (Mat_<double>(3,3) << 0, -v[2], v[1],
            v[2], 0, -v[0],
            -v[1], v[0], 0);
}

Mat Math::cross(const Mat &a_, const Mat &b_) {
    const auto * const a = (double *) a_.data;
    const auto * const b = (double *) b_.data;
    return (Mat_<double>(3,1) << a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]);
}

double Math::getMean (const std::vector<double> &array) {
    if (array.empty()) return 0;
    double mean = 0;
    for (double i : array)
        mean += i;
    return mean / array.size();
}

double Math::getStandardDeviation (const std::vector<double> &array) {
    if (array.empty()) return 0;
    double std_dev = 0;
    double mean = getMean(array);
    for (double i : array)
        std_dev += std::pow(i - mean, 2);
    return sqrt (std_dev / static_cast<double>(array.size() - 1));
}

/*
 * Use Gauss Elimination Method to find rank
 */
int Math::rank3x3 (const Mat &A_) {
    Mat A;
    A_.copyTo(A);
    A.convertTo(A, CV_64F); // convert to double
    auto * a = (double *) A.data;
    const int m = A.rows, n = A.cols;

    eliminateUpperTriangluar(a, m, n);

    // find number of non zero rows
    int rank = 0;
    for (int r = 0; r < m; r++)
        // check if row is zeros
        for (int c = 0; c < n; c++)
            if (fabs(a[r*n+c]) > FLT_EPSILON) {
                rank++; // some value in the row is not zero -> increase the rank
                break;
            }

    return rank;
}
/*
 * Eliminate matrix of m rows and n columns to be upper triangular.
 */
void Math::eliminateUpperTriangluar (double * a, int m, int n) {
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
    int subset_size, max_range;
    RNG &rng;
public:
    UniformRandomGeneratorImpl (RNG &rng_) : rng(rng_) {}

    // interval is <0; max_range);
    UniformRandomGeneratorImpl (RNG &rng_, int max_range_, int subset_size_) : rng(rng_) {
        assert(subset_size_ > 0);
        assert(subset_size_ <= max_range_);
        subset_size = subset_size_;
        max_range = max_range_;
    }

    int getRandomNumber () override {
        return rng.uniform(0, max_range);
    }

    // closed range
    void resetGenerator (int max_range_) override {
        assert(0 <= max_range_);
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
        assert(subset_size <= max_range_);
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
        assert(subset_size_ <= max_range_);
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

Ptr<UniformRandomGenerator> UniformRandomGenerator::create (RNG &rng) {
    return Ptr<UniformRandomGeneratorImpl>(new UniformRandomGeneratorImpl(rng));
}
Ptr<UniformRandomGenerator> UniformRandomGenerator::create
        (RNG &rng, int max_range, int subset_size_) {
    return Ptr<UniformRandomGeneratorImpl>(
            new UniformRandomGeneratorImpl(rng, max_range, subset_size_));
}
}}
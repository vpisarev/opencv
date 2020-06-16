// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include "usac.hpp"

namespace cv { namespace usac {
// Performs Fisher-Yates shuffle
void Utils::random_shuffle (std::vector<int>& array) {
    int rand_idx, temp_size = array.size(), array_size = array.size();
    for (int i = 0; i < array_size; i++) {
        rand_idx = random () % temp_size; // get random index of array of working interval

        // decrease temp size and swap values of random index to the end (temp size)
        std::swap(array[rand_idx], array[--temp_size]);
    }
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
    int subset_size, range_size, min_range, max_range;
    bool is_initialized;
public:
    UniformRandomGeneratorImpl () {
        is_initialized = false;
    }

    UniformRandomGeneratorImpl (int min_range, int max_range, int subset_size_) {
        subset_size = subset_size_;
        resetGenerator(min_range, max_range);
        assert(subset_size <= range_size);

        is_initialized = true;
    }

    int getRandomNumber () override {
        return min_range + random() % range_size;
    }

    // closed range
    void resetGenerator (int min_range_, int max_range_) override {
        assert(min_range_ <= max_range_);

        max_range = max_range_;
        min_range = min_range_;
        range_size = max_range - min_range + 1;
    }

    void generateUniqueRandomSet (std::vector<int>& sample) override {
        int j, num;
        sample[0] = random() % range_size;
        for (int i = 1; i < subset_size;) {
            num = random() % range_size;
            // check if value is in array
            for (j = i - 1; j >= 0; j--) {
                if (num == sample[j]) {
                    // if so, generate again
                    break;
                }
            }
            // success, value is not in array, so it is unique, add to sample.
            if (j == -1) sample[i++] = num;
        }
    }

    // closed interval <0; max_range>
    void generateUniqueRandomSet (std::vector<int>& sample, int max_range) override {
        max_range += 1; // make open interval

        /*
         * necessary condition:
         * if subset size is bigger than range then array cannot be unique,
         * so function has infinite loop.
         */
        assert(subset_size <= max_range);

        int num, j;
        sample[0] = random() % max_range;
        for (int i = 1; i < subset_size;) {
            num = random() % max_range;
            for (j = i - 1; j >= 0; j--) {
                if (num == sample[j]) {
                    break;
                }
            }
            if (j == -1) sample[i++] = num;
        }
    }

    void setSubsetSize (int subset_size_) override {
        subset_size = subset_size_;
    }

    bool isInitialized () override {
        return is_initialized;
    }
};

Ptr<UniformRandomGenerator> UniformRandomGenerator::create () {
    return makePtr<UniformRandomGeneratorImpl>();
}
Ptr<UniformRandomGenerator> UniformRandomGenerator::create
        (int min_range, int max_range, int subset_size_) {
    return makePtr<UniformRandomGeneratorImpl>(min_range, max_range, subset_size_);
}

// closed interval <0, max_range>
void UniformRandomGenerator::generateUniqueRandomSet
        (std::vector<int>& sample, int subset_size, int max_range) {
    max_range += 1; // add one to make open interval
    assert(subset_size <= max_range);

    int num, j;
    sample[0] = random() % max_range;
    for (int i = 1; i < subset_size;) {
        num = random() % max_range;
        for (j = i - 1; j >= 0; j--)
            if (num == sample[j])
                break;
        if (j == -1) sample[i++] = num;
    }
}

}}
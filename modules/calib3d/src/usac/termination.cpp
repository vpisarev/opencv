// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../usac.hpp"

////////////////////////////////// STANDARD TERMINATION ///////////////////////////////////////////
namespace cv { namespace usac {
class StandardTerminationCriteriaImpl : public StandardTerminationCriteria {
private:
    const double log_confidence;
    const int MAX_ITERATIONS, MAX_TIME_MCS, points_size;
    int predicted_iterations;
    const int sample_size;

    const bool is_time_limit;
    std::chrono::steady_clock::time_point begin_time;
public:
    StandardTerminationCriteriaImpl (double confidence, int points_size_,
            int sample_size_, int max_iterations_, bool is_time_limit_,
            int max_time_mcs_) :
            log_confidence(log(1 - confidence)), points_size (points_size_),
            sample_size (sample_size_), is_time_limit(is_time_limit_),
            MAX_ITERATIONS(max_iterations_), MAX_TIME_MCS(max_time_mcs_) {
        predicted_iterations = max_iterations_;
    }

    /*
     * Get upper bound iterations for any sample number
     * n is points size, w is inlier ratio, p is desired probability, k is expceted number of iterations.
     * 1 - p = (1 - w^n)^k,
     * k = log_(1-w^n) (1-p)
     * k = ln (1-p) / ln (1-w^n)
     *
     * w^n is probability that all N points are inliers.
     * (1 - w^n) is probability that at least one point of N is outlier.
     * 1 - p = (1-w^n)^k is probability that in K steps of getting at least one outlier is 1% (5%).
     */
    void update (const Mat &model, int inlier_number) override {
        const double predicted_iters = log_confidence / log(1 - std::pow
            (static_cast<double>(inlier_number) / points_size, sample_size));

        // if inlier_prob == 1 then log(0) = -inf, predicted_iters == -0
        // if inlier_prob == 0 then log(1) = 0   , predicted_iters == (+-) inf

        if (! std::isinf(predicted_iters) && predicted_iters < MAX_ITERATIONS)
            predicted_iterations = static_cast<int>(predicted_iters);
    }

    /*
     * important to use inline here, because function is callable from RANSAC while()
     * loop. keep function virtual, if other termination criteria do not depend on
     * iteration or time number.
     */
    inline bool terminate (int current_iteration) const override {
        // check current iteration number is higher than maximum predicted iterations.
        if (current_iteration >= predicted_iterations) return true;

        if (is_time_limit)
            // check running time. By default max time set to maximum possible of int number.
            return std::chrono::duration_cast<std::chrono::microseconds>
                           (std::chrono::steady_clock::now() - begin_time).count() > MAX_TIME_MCS;
        return false;
    }

    void startMeasureTime () override {
        begin_time = std::chrono::steady_clock::now();
    }
    inline int getPredictedNumberIterations () const override {
        return predicted_iterations;
    }
    void reset () override {
        predicted_iterations = MAX_ITERATIONS;
    }
};

Ptr<StandardTerminationCriteria> StandardTerminationCriteria::create(double confidence,
    int points_size_, int sample_size_, int max_iterations_, bool is_time_limit_,
    int max_time_mcs_) {
    return makePtr<StandardTerminationCriteriaImpl>(confidence, points_size_,
                        sample_size_, max_iterations_, is_time_limit_, max_time_mcs_);
}
}}

// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include "usac.hpp"

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
        // if inlier_prob == 0 then log(1) = 0   , predicted_iters == inf

        if (predicted_iters < MAX_ITERATIONS)
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

/////////////////////////////////////// SPRT TERMINATION //////////////////////////////////////////

class SPRTTerminationImpl : public SPRTTermination {
private:
    const std::vector<SPRT_history> &sprt_histories;

    const double log_eta_0;
    const int MAX_ITERATIONS, MAX_TIME_MCS, points_size;
    int predicted_iterations;
    const int sample_size;
    const bool is_time_limit;
    std::chrono::steady_clock::time_point begin_time;
public:
    SPRTTerminationImpl (const std::vector<SPRT_history> &sprt_histories_, double confidence,
            int points_size_, int sample_size_, int max_iterations_,
            bool is_time_limit_, int max_time_mcs_ = std::numeric_limits<int>::max())
            : sprt_histories (sprt_histories_), log_eta_0(log(1-confidence)),
            points_size (points_size_),
            sample_size (sample_size_), is_time_limit(is_time_limit_),
            MAX_ITERATIONS(max_iterations_), MAX_TIME_MCS(max_time_mcs_) {
        predicted_iterations = max_iterations_;
    }

    /*
     * Termination criterion:
     * l is number of tests
     * n(l) = Product from i = 0 to l ( 1 - P_g (1 - A(i)^(-h(i)))^k(i) )
     * log n(l) = sum from i = 0 to l k(i) * ( 1 - P_g (1 - A(i)^(-h(i))) )
     *
     *        log (n0) - log (n(l-1))
     * k(l) = -----------------------
     *          log (1 - P_g*A(l)^-1)
     *
     * A is decision threshold
     * P_g is probability of good model.
     * k(i) is number of samples verified by i-th sprt.
     * n0 is typically set to 0.05
     * this equation does not have to be evaluated before nR < n0
     * nR = (1 - P_g)^k
     */

    void update (const Mat &model, int inlier_size) override {
        if (sprt_histories.empty()) {
            predicted_iterations = std::min(MAX_ITERATIONS, getStandardUpperBound(inlier_size));
            return;
        }

        const double epsilon = (double) inlier_size / points_size; // inlier probability
        const double P_g = pow (epsilon, sample_size); // probability of good sample

        double log_eta_lmin1 = 0;

        int total_number_of_tested_samples = 0;
        const int sprts_size = sprt_histories.size();
        // compute log n(l-1), l is number of tests
        for (int test = 0; test < sprts_size-1; test++) {
            const double h = computeExponentH(sprt_histories[test].epsilon, epsilon, sprt_histories[test].delta);
            // std::cout << sprt_histories[test].epsilon << " " << sprt_histories[test].delta << " " << h << "\n";
            log_eta_lmin1 += log (1 - P_g * (1 - pow (sprt_histories[test].A, -h))) * sprt_histories[test].tested_samples;
            total_number_of_tested_samples += sprt_histories[test].tested_samples;
        }

        if (std::pow(1 - P_g, total_number_of_tested_samples) < log_eta_0) {
            predicted_iterations = std::min(MAX_ITERATIONS, getStandardUpperBound(inlier_size));
            return;
        }

        double numerator = log_eta_0 - log_eta_lmin1;
        // denominator is always negative, so numerator must also be negative
        if (numerator >= 0) {
            predicted_iterations = 0;
            return;
        }

        // use decision threshold A for last test (l-th)
        double denominator = log (1 - P_g * (1 - 1 / sprt_histories[sprts_size-1].A));

        // if denominator is nan or almost zero then return max integer
        if (std::isnan(denominator) || fabs (denominator) < std::numeric_limits<double>::max()) {
            predicted_iterations = std::min(MAX_ITERATIONS, getStandardUpperBound(inlier_size));
            return;
        }

        // predicted number of iterations of SPRT
        predicted_iterations = std::min(MAX_ITERATIONS, static_cast<int >(ceil(numerator / denominator)));

        // compare with standard termination criterion
        predicted_iterations = std::min(predicted_iterations, getStandardUpperBound(inlier_size));
    }

    inline bool terminate (int current_iteration) const override {
        if (current_iteration >= predicted_iterations) return true;
        if (is_time_limit)
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
private:
    inline int getStandardUpperBound(int inlier_size) const {
        const double predicted_iters = log_eta_0 / log(1 - std::pow
                (static_cast<double>(inlier_size) / points_size, sample_size));

        return (predicted_iters < MAX_ITERATIONS) ? static_cast<int>(predicted_iters) : MAX_ITERATIONS;
    }
    /*
     * h(i) must hold
     *
     *     δ(i)                  1 - δ(i)
     * ε (-----)^h(i) + (1 - ε) (--------)^h(i) = 1
     *     ε(i)                  1 - ε(i)
     *
     * ε * a^h + (1 - ε) * b^h = 1
     * Has numerical solution.
     */
    static double computeExponentH (double epsilon, double epsilon_new, double delta) {
        double a = log (delta / epsilon); // log likelihood ratio
        double b = log ((1 - delta) / (1 - epsilon));

        double x0 = log (1 / (1 - epsilon_new)) / b;
        double v0 = epsilon_new * exp (x0 * a);
        double x1 = log ((1 - 2*v0) / (1 - epsilon_new)) / b;
        double v1 = epsilon_new * exp (x1 * a) + (1 - epsilon_new) * exp(x1 * b);
        double h = x0 - (x0 - x1) / (1 + v0 - v1) * v0;

        if (std::isnan(h)) {
            // The equation always has solution for h = 0
            // ε * a^0 + (1 - ε) * b^0 = 1
            // ε + 1 - ε = 1 -> 1 = 1
            return 0;
        }

        return h;
    }
};

Ptr<SPRTTermination> SPRTTermination::create(const std::vector<SPRT_history> &sprt_histories_,
    double confidence, int points_size_, int sample_size_, int max_iterations_,
    bool is_time_limit_, int max_time_mcs_) {
    return makePtr<SPRTTerminationImpl>(sprt_histories_, confidence, points_size_, sample_size_,
                    max_iterations_, is_time_limit_, max_time_mcs_);
}
}}

// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../usac.hpp"

namespace cv { namespace usac {
////////////////////////////////// STANDARD TERMINATION ///////////////////////////////////////////
class StandardTerminationCriteriaImpl : public StandardTerminationCriteria {
private:
    const double log_confidence;
    const int points_size, sample_size, MAX_ITERATIONS;
public:
    StandardTerminationCriteriaImpl (double confidence, int points_size_,
                                     int sample_size_, int max_iterations_) :
            log_confidence(log(1 - confidence)), points_size (points_size_),
            sample_size (sample_size_), MAX_ITERATIONS(max_iterations_)  {}

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
    int update (const Mat &/*model*/, int inlier_number) override {
        const double predicted_iters = log_confidence / log(1 - std::pow
            (static_cast<double>(inlier_number) / points_size, sample_size));

        // if inlier_prob == 1 then log(0) = -inf, predicted_iters == -0
        // if inlier_prob == 0 then log(1) = 0   , predicted_iters == (+-) inf

        if (! std::isinf(predicted_iters) && predicted_iters < MAX_ITERATIONS)
            return static_cast<int>(predicted_iters);
        return MAX_ITERATIONS;
    }

    Ptr<TerminationCriteria> clone () const override {
        return makePtr<StandardTerminationCriteriaImpl>(1-exp(log_confidence), points_size,
                sample_size, MAX_ITERATIONS);
    }
};
Ptr<StandardTerminationCriteria> StandardTerminationCriteria::create(double confidence,
    int points_size_, int sample_size_, int max_iterations_) {
    return makePtr<StandardTerminationCriteriaImpl>(confidence, points_size_,
                        sample_size_, max_iterations_);
}

/////////////////////////////////////// SPRT TERMINATION //////////////////////////////////////////
class SPRTTerminationImpl : public SPRTTermination {
private:
    const std::vector<SPRT_history> &sprt_histories;
    const double log_eta_0;
    const int points_size, sample_size, MAX_ITERATIONS;
public:
    SPRTTerminationImpl (const std::vector<SPRT_history> &sprt_histories_, double confidence,
           int points_size_, int sample_size_, int max_iterations_)
           : sprt_histories (sprt_histories_), log_eta_0(log(1-confidence)),
           points_size (points_size_), sample_size (sample_size_),MAX_ITERATIONS(max_iterations_){}

    /*
     * Termination criterion:
     * l is number of tests
     * n(l) = Product from i = 0 to l ( 1 - P_g (1 - A(i)^(-h(i)))^k(i) )
     * log n(l) = sum from i = 0 to l k(i) * ( 1 - P_g (1 - A(i)^(-h(i))) )
     *
     *        log (n0) - log (n(l-1))
     * k(l) = -----------------------  (9)
     *          log (1 - P_g*A(l)^-1)
     *
     * A is decision threshold
     * P_g is probability of good model.
     * k(i) is number of samples verified by i-th sprt.
     * n0 is typically set to 0.05
     * this equation does not have to be evaluated before nR < n0
     * nR = (1 - P_g)^k
     */
    int update (const Mat &/*model*/, int inlier_size) override {
        if (sprt_histories.empty())
            return std::min(MAX_ITERATIONS, getStandardUpperBound(inlier_size));

        const double epsilon = static_cast<double>(inlier_size) / points_size; // inlier probability
        const double P_g = pow (epsilon, sample_size); // probability of good sample

        double log_eta_lmin1 = 0;

        int total_number_of_tested_samples = 0;
        const int sprts_size_min1 = static_cast<int>(sprt_histories.size())-1;
        if (sprts_size_min1 < 0) return getStandardUpperBound(inlier_size);
        // compute log n(l-1), l is number of tests
        for (int test = 0; test < sprts_size_min1; test++) {
            log_eta_lmin1 += log (1 - P_g * (1 - pow (sprt_histories[test].A,
             -computeExponentH(sprt_histories[test].epsilon, epsilon,sprt_histories[test].delta))))
                         * sprt_histories[test].tested_samples;
            total_number_of_tested_samples += sprt_histories[test].tested_samples;
        }

        // Implementation note: since η > ηR the equation (9) does not have to be evaluated
        // before ηR < η0 is satisfied.
        if (std::pow(1 - P_g, total_number_of_tested_samples) < log_eta_0)
            return std::min(MAX_ITERATIONS, getStandardUpperBound(inlier_size));
        // use decision threshold A for last test (l-th)
        const double predicted_iters_sprt = (log_eta_0 - log_eta_lmin1) /
                log (1 - P_g * (1 - 1 / sprt_histories[sprts_size_min1].A)); // last A
        if (std::isnan(predicted_iters_sprt) || std::isinf(predicted_iters_sprt))
            return getStandardUpperBound(inlier_size);

        if (predicted_iters_sprt < 0) return 0;
        // compare with standard upper bound
        if (predicted_iters_sprt < MAX_ITERATIONS)
            return std::min(static_cast<int>(predicted_iters_sprt),
                    getStandardUpperBound(inlier_size));
        return getStandardUpperBound(inlier_size);
    }

    Ptr<TerminationCriteria> clone () const override {
        return makePtr<SPRTTerminationImpl>(sprt_histories, 1-exp(log_eta_0), points_size,
               sample_size, MAX_ITERATIONS);
    }
private:
    inline int getStandardUpperBound(int inlier_size) const {
        const double predicted_iters = log_eta_0 / log(1 - std::pow
                (static_cast<double>(inlier_size) / points_size, sample_size));
        return (! std::isinf(predicted_iters) && predicted_iters < MAX_ITERATIONS) ?
                static_cast<int>(predicted_iters) : MAX_ITERATIONS;
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
        const double a = log (delta / epsilon); // log likelihood ratio
        const double b = log ((1 - delta) / (1 - epsilon));

        const double x0 = log (1 / (1 - epsilon_new)) / b;
        const double v0 = epsilon_new * exp (x0 * a);
        const double x1 = log ((1 - 2*v0) / (1 - epsilon_new)) / b;
        const double v1 = epsilon_new * exp (x1 * a) + (1 - epsilon_new) * exp(x1 * b);
        const double h = x0 - (x0 - x1) / (1 + v0 - v1) * v0;

        if (std::isnan(h))
            // The equation always has solution for h = 0
            // ε * a^0 + (1 - ε) * b^0 = 1
            // ε + 1 - ε = 1 -> 1 = 1
            return 0;
        return h;
    }
};
Ptr<SPRTTermination> SPRTTermination::create(const std::vector<SPRT_history> &sprt_histories_,
    double confidence, int points_size_, int sample_size_, int max_iterations_) {
    return makePtr<SPRTTerminationImpl>(sprt_histories_, confidence, points_size_, sample_size_,
                    max_iterations_);
}
}}

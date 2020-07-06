// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#if defined(_MSC_VER)
#   pragma warning(push)
#   pragma warning(disable:4800) // warning C4800: 'const int': forcing value to bool 'true' or 'false' (performance warning)
#endif
#include "../precomp.hpp"
#if defined(_MSC_VER)
#   pragma warning(pop)
#endif
#include "../usac.hpp"

namespace cv { namespace usac {
class RansacQualityImpl : public RansacQuality {
private:
    const Ptr<Error> error;
    const int points_size;
    const double threshold;
    double best_score;
public:
    RansacQualityImpl (int points_size_, double threshold_, const Ptr<Error> &error_)
            : error (error_), points_size(points_size_), threshold(threshold_) {
        best_score = std::numeric_limits<double>::max();
    }

    // use inline
    Score getScore (const Mat &model, double threshold_, bool get_inliers,
            std::vector<int> &inliers) const override {
        error->setModelParameters(model);
        int inlier_number = 0;

        if (get_inliers) {
            for (int point = 0; point < points_size; point++) {
                if (error->getError(point) < threshold_)
                    inliers[inlier_number++] = point;
                // if current number of inliers plus all possible are less than
                // max number of inliers then break evaluation.
                if (inlier_number + (points_size - point) < -best_score)
                    break;
            }
        } else {
            for (int point = 0; point < points_size; point++) {
                if (error->getError(point) < threshold_)
                    inlier_number++;
                if (inlier_number + (points_size - point) < -best_score)
                    break;
            }
        }

        // score is negative inlier number! If less then better
        return Score(inlier_number, -static_cast<double>(inlier_number));
    }

    void setBestScore(double best_score_) override {
        best_score = best_score_;
    }

    Score getScore (const Mat &model, bool get_inliers, std::vector<int> &inliers) const override
    { return getScore (model, threshold, get_inliers, inliers); }
    inline Score getScore (const Mat &model) const override
    { std::vector<int> inliers_;
    return getScore (model, threshold, false, inliers_); }

    int getInliers (const Mat &model, std::vector<int> &inliers) const override
    { return getInliers (model, inliers, threshold); }

    // get inliers for given threshold
    int getInliers (const Mat &model, std::vector<int> &inliers, double thr) const override {
        error->setModelParameters(model);
        int num_inliers = 0;
        for (int point = 0; point < points_size; point++)
            if (error->getError(point) < thr)
                inliers[num_inliers++] = point;
        return num_inliers;
    }

    int getInliers (const Mat &model, std::vector<bool> &inliers_mask) const override {
        std::fill(inliers_mask.begin(), inliers_mask.end(), 0);
        error->setModelParameters(model);
        int num_inliers = 0;
        for (int point = 0; point < points_size; point++) {
            if (error->getError(point) < threshold) {
                inliers_mask[point] = true;
                num_inliers++;
            }
        }
        return num_inliers;
    }

    void setModel (const Mat &model) const override
    { error->setModelParameters (model); }

    inline bool isInlier (int point_idx) const override
    { return error->getError (point_idx) < threshold; }
    Ptr<Quality> clone () const override {
        return makePtr<RansacQualityImpl>(points_size, threshold, error->clone());
    }
};

Ptr<RansacQuality> RansacQuality::create(int points_size_, double threshold_,
        const Ptr<Error> &error_) {
    return makePtr<RansacQualityImpl>(points_size_, threshold_, error_);
}

class MsacQualityImpl : public MsacQuality {
protected:
    const Ptr<Error> error;
    const int points_size;
    const double threshold;
    double best_score;
public:
    MsacQualityImpl (int points_size_, double threshold_, const Ptr<Error> &error_)
            : error (error_), points_size (points_size_), threshold (threshold_) {
        best_score = std::numeric_limits<double>::max();
    }

    inline Score getScore (const Mat &model, double threshold_, bool get_inliers,
            std::vector<int> &inliers) const override {
        error->setModelParameters(model);

        double err, sum_errors = 0;
        int inlier_number = 0;
        for (int point = 0; point < points_size; point++) {
            err = error->getError(point);
            if (err < threshold_) {
                if (get_inliers)
                    inliers[inlier_number] = point;
                sum_errors += err;
                inlier_number++;
            } else
                sum_errors += threshold_;

            if (sum_errors > best_score)
                break;
        }
        return Score(inlier_number, sum_errors);
    }

    void setBestScore(double best_score_) override {
        best_score = best_score_;
    }

    Score getScore (const Mat &model, bool get_inliers, std::vector<int> &inliers) const override
    { return getScore (model, threshold, get_inliers, inliers); }
    inline Score getScore (const Mat &model) const override
    { std::vector<int> inliers_; return getScore (model, threshold, false, inliers_); }

    int getInliers (const Mat &model, std::vector<int> &inliers) const override
    { return getInliers (model, inliers, threshold); }

    // get inliers for given threshold
    int getInliers (const Mat &model, std::vector<int> &inliers, double thr) const override {
        error->setModelParameters(model);
        int num_inliers = 0;
        for (int point = 0; point < points_size; point++)
            if (error->getError(point) < thr)
                inliers[num_inliers++] = point;
        return num_inliers;
    }

    int getInliers (const Mat &model, std::vector<bool> &inliers_mask) const override {
        std::fill(inliers_mask.begin(), inliers_mask.end(), 0);
        error->setModelParameters(model);
        int num_inliers = 0;
        for (int point = 0; point < points_size; point++) {
            if (error->getError(point) < threshold) {
                inliers_mask[point] = true;
                num_inliers++;
            }
        }
        return num_inliers;
    }

    inline void setModel (const Mat &model) const override
    { error->setModelParameters (model); }

    inline bool isInlier (int point_idx) const override
    { return error->getError (point_idx) < threshold; }
    Ptr<Quality> clone () const override {
        return makePtr<MsacQualityImpl>(points_size, threshold, error->clone());
    }
};
Ptr<MsacQuality> MsacQuality::create(int points_size_, double threshold_,
        const Ptr<Error> &error_) {
    return makePtr<MsacQualityImpl>(points_size_, threshold_, error_);
}

///////////////////////////////////// SPRT VERIFIER MSAC //////////////////////////////////////////
class SPRTImpl : public SPRT {
private:
    RNG rng;
    const Ptr<Error> err;
    const int points_size;
    const double inlier_threshold, t_M, m_S;

    // 10 - RANSAC, 20 - MSAC, for everything else is not computed
    const int score_type;

    double current_epsilon, current_delta, current_A, delta_to_epsilon, complement_delta_to_complement_epsilon;
    // time t_M needed to instantiate a model hypothesis given a sample
    // Let m_S be the number of models that are verified per sample

    int random_pool_idx, highest_inlier_number, current_sprt_idx; // i
    std::vector<SPRT_history> sprt_histories;
    std::vector<int> points_random_pool;

    Score score;
    bool last_model_is_good;
    double lowest_sum_errors;
public:
    explicit SPRTImpl (int state, const Ptr<Error>&err_, int points_size_,
        double inlier_threshold_, double prob_pt_of_good_model, double prob_pt_of_bad_model,
        double time_sample, double avg_num_models, int score_type_) : rng(state), err(err_),
        points_size(points_size_), inlier_threshold (inlier_threshold_),
        t_M (time_sample), m_S (avg_num_models), score_type (score_type_) {

        // Generate array of random points for randomized evaluation
        points_random_pool = std::vector<int> (points_size_);
        // fill values from 0 to points_size-1
        for (int i = 0; i < points_size; i++)
            points_random_pool[i] = i;
        Utils::random_shuffle (rng, points_random_pool);
        ///////////////////////////////

        // reserve (approximately) some space for sprt vector.
        sprt_histories.reserve(20);

        createTest(prob_pt_of_good_model, prob_pt_of_bad_model);

        highest_inlier_number = 0;
        lowest_sum_errors = std::numeric_limits<double>::max();
        last_model_is_good = false;
    }

    /*
     *                      p(x(r)|Hb)                  p(x(j)|Hb)
     * lambda(j) = Product (----------) = lambda(j-1) * ----------
     *                      p(x(r)|Hg)                  p(x(j)|Hg)
     * Set j = 1
     * 1.  Check whether j-th data point is consistent with the
     * model
     * 2.  Compute the likelihood ratio λj eq. (1)
     * 3.  If λj >  A, decide the model is ’bad’ (model ”re-jected”),
     * else increment j or continue testing
     * 4.  If j = N the number of correspondences decide model ”accepted”
     *
     * Verifies model and returns model score.

     * Returns true if model is good, false - otherwise.
     * @model: model to verify
     * @current_hypothesis: current RANSAC iteration
     * Return: true if model is good, false - otherwise.
     */
    inline bool isModelGood (const Mat &model) override {
        // update error object with current model
        err->setModelParameters(model);

        double lambda = 1, sum_errors = 0;
        last_model_is_good = true;
        random_pool_idx = rng.uniform(0, points_size);

        int tested_point, tested_inliers = 0;
        if (score_type == 20)
            for (tested_point = 0; tested_point < points_size; tested_point++) {
                if (random_pool_idx >= points_size)
                    random_pool_idx = 0;
                const double error = err->getError (points_random_pool[random_pool_idx++]);
                if (error < inlier_threshold) {
                    sum_errors += error;
                    tested_inliers++;
                    lambda *= delta_to_epsilon;
                } else {
                    sum_errors += inlier_threshold;
                    lambda *= complement_delta_to_complement_epsilon;
                }
                if (lambda > current_A || sum_errors > lowest_sum_errors /* no case it would be better */) {
                    last_model_is_good = false;
                    tested_point++;
                    break;
                }
            }
        else
            for (tested_point = 0; tested_point < points_size; tested_point++) {
                // reset pool index if it overflows
                if (random_pool_idx >= points_size)
                    random_pool_idx = 0;
                if (err->getError (points_random_pool[random_pool_idx++]) < inlier_threshold) {
                    tested_inliers++;
                    lambda *= delta_to_epsilon;
                } else
                    lambda *= complement_delta_to_complement_epsilon;
                if (lambda > current_A || (tested_inliers + (points_size - tested_point)) < highest_inlier_number) {
                    last_model_is_good = false;
                    tested_point++;
                    break;
                }
            }

        // increase number of samples processed by current test
        sprt_histories[current_sprt_idx].tested_samples++;
        if (last_model_is_good) {
            score.inlier_number = tested_inliers;
            if (score_type == 20) {
                score.score = sum_errors;
                lowest_sum_errors = sum_errors;
            } else if (score_type == 10)
                score.score = -static_cast<double>(tested_inliers);

            if (tested_inliers > highest_inlier_number) {
                highest_inlier_number = tested_inliers; // update max inlier number
                /*
                 * Model accepted and the largest support so far:
                 * design (i+1)-th test (εi + 1= εˆ, δi+1 = δ, i := i + 1).
                 * Store the current model parameters θ
                 */
                createTest(static_cast<double>(tested_inliers)
                         / points_size, current_delta);
            }
        } else {
            /*
             * Since almost all tested models are ‘bad’, the probability
             * δ can be estimated as the average fraction of consistent data points
             * in rejected models.
             */
            double delta_estimated = static_cast<double> (tested_inliers) / tested_point;
            if (delta_estimated > 0 && fabs(current_delta - delta_estimated)
                                       / current_delta > 0.05)
                /*
                 * Model rejected: re-estimate δ. If the estimate δ_ differs
                 * from δi by more than 5% design (i+1)-th test (εi+1 = εi,
                 * δi+1 = δˆ, i := i + 1)
                 */
                createTest(current_epsilon, delta_estimated);
        }
        return last_model_is_good;
    }

    const std::vector<SPRT_history> &getSPRTvector () const override {
        return sprt_histories;
    }
    inline bool getScore (Score &score_) const override {
        if (!last_model_is_good) return false;
        if (score_type != 20 || score_type != 10)
            return false;
        score_ = score;
        return true;
    }
    void update (int highest_inlier_number_) override {
        if (highest_inlier_number_ > highest_inlier_number) {
            // update probability of point belonging to good model and create new test.
            highest_inlier_number = highest_inlier_number_;
            if (sprt_histories[current_sprt_idx].tested_samples == 0)
                sprt_histories[current_sprt_idx].tested_samples = 1;

            createTest(static_cast<double>(highest_inlier_number)
                       / points_size, current_delta);
        }
    }
    Ptr<ModelVerifier> clone () const override {
        return makePtr<SPRTImpl>(rng.state/3+3, err->clone(), points_size, inlier_threshold,
                sprt_histories[current_sprt_idx].epsilon, sprt_histories[current_sprt_idx].delta,
                t_M, m_S, score_type);
    }
private:

    // Saves sprt test to sprt history and update current epsilon, delta and threshold.
    void createTest (double epsilon, double delta) {
        // if epsilon is closed to 1 then set them to 0.99 to avoid numerical problems
        if (epsilon > 0.999999) epsilon = 0.99;
        // avoid delta going too high as it is very unlikely
        if (delta   > 0.8) delta = 0.8;

        SPRT_history new_sprt_history;
        new_sprt_history.epsilon = epsilon;
        new_sprt_history.delta = delta;
        new_sprt_history.A = estimateThresholdA (epsilon, delta);

        sprt_histories.emplace_back(new_sprt_history);

        current_A = new_sprt_history.A;
        current_delta = delta;
        current_epsilon = epsilon;

        delta_to_epsilon = delta / epsilon;
        complement_delta_to_complement_epsilon = (1 - delta) / (1 - epsilon);

        current_sprt_idx = static_cast<int>(sprt_histories.size()) - 1;
    }

    /*
    * A(0) = K1/K2 + 1
    * A(n+1) = K1/K2 + 1 + log (A(n))
    * K1 = t_M / P_g
    * K2 = m_S/(P_g*C)
    * t_M is time needed to instantiate a model hypotheses given a sample
    * P_g = epsilon ^ m, m is the number of data point in the Ransac sample.
    * m_S is the number of models that are verified per sample.
    *                   p (0|Hb)                  p (1|Hb)
    * C = p(0|Hb) log (---------) + p(1|Hb) log (---------)
    *                   p (0|Hg)                  p (1|Hg)
    */
    double estimateThresholdA (double epsilon, double delta) {
        const double C = (1 - delta) * log ((1 - delta) / (1 - epsilon)) +
                         delta * (log(delta / epsilon));
        // K = K1/K2 + 1 = (t_M / P_g) / (m_S / (C * P_g)) + 1 = (t_M * S)/m_S + 1
        const double K = (t_M * C) / m_S + 1;
        double An, An_1 = K;
        // compute A using a recursive relation
        // A* = lim(n->inf)(An), the series typically converges within 4 iterations
        for (int i = 0; i < 10; i++) {
            An = K + log(An_1);
            if (fabs(An - An_1) < FLT_EPSILON)
                break;
            An_1 = An;
        }
        return An;
    }
};
Ptr<SPRT> SPRT::create (int state, const Ptr<Error> &err_, int points_size_,
      double inlier_threshold_, double prob_pt_of_good_model, double prob_pt_of_bad_model,
      double time_sample, double avg_num_models, int score_type_) {
    return Ptr<SPRTImpl>(new SPRTImpl (state, err_, points_size_,
    inlier_threshold_, prob_pt_of_good_model, prob_pt_of_bad_model, time_sample, avg_num_models,
                                                       score_type_));
}
}}
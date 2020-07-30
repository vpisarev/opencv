// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../usac.hpp"

namespace cv { namespace usac {
/*
* Reference:
* http://cmp.felk.cvut.cz/~matas/papers/chum-dagm03.pdf
*/
class InnerLocalOptimizationImpl : public InnerLocalOptimization {
private:
    const Ptr<Estimator> estimator;
    const Ptr<Quality> quality;
    const Ptr<Sampler> lo_sampler;

    Score lo_score;
    std::vector<Mat> lo_models;
    std::vector<int> inliers_of_best_model, lo_sample;
    std::vector<double> weights;
    int lo_inner_max_iterations, lo_sample_size;
public:

    InnerLocalOptimizationImpl (const Ptr<Estimator> &estimator_,
            const Ptr<Quality> &quality_, const Ptr<Sampler> &lo_sampler_,
            int points_size, int lo_inner_iterations_)
            : estimator (estimator_), quality (quality_), lo_sampler (lo_sampler_) {

        lo_inner_max_iterations = lo_inner_iterations_;
        lo_sample_size = lo_sampler->getSampleSize();

        // Allocate max memory to avoid reallocation
        inliers_of_best_model = std::vector<int>(points_size);
        lo_sample = std::vector<int>(lo_sample_size);
        lo_models = std::vector<Mat> (estimator->getMaxNumSolutionsNonMinimal());
    }

    // Implementation of Inner Locally Optimized Ransac
    bool refineModel (const Mat &so_far_the_best_model, const Score &best_model_score,
                      Mat &new_model, Score &new_model_score) override {
        new_model_score = Score(); // set score to inf (worst case)

        // get inliers from so far the best model.
        int num_inliers_of_best_model = quality->getInliers(so_far_the_best_model,
                                                           inliers_of_best_model);

        // Inner Local Optimization Ransac.
        for (int iters = 0; iters < lo_inner_max_iterations; iters++) {
            // Generate sample of lo_sample_size from inliers from the best model.
            int num_estimated_models;
            if (num_inliers_of_best_model > lo_sample_size) {
                // if there are many inliers take limited number at random.
                lo_sampler->generateSample (lo_sample, num_inliers_of_best_model);
                // get inliers from maximum inliers from lo
                for (int smpl = 0; smpl < lo_sample_size; smpl++)
                    lo_sample[smpl] = inliers_of_best_model[lo_sample[smpl]];

                num_estimated_models = estimator->estimateModelNonMinimalSample
                        (lo_sample, lo_sample_size, lo_models, weights);
                if (num_estimated_models == 0) continue;
            } else {
                // if model was not updated in first iteration, so break.
                if (iters > 0) break;
                // if inliers are less than limited number of sample then take all of them for estimation
                // if it fails -> end Lo.
                num_estimated_models = estimator->estimateModelNonMinimalSample(
                        inliers_of_best_model, num_inliers_of_best_model, lo_models, weights);
                if (num_estimated_models == 0)
                    return false;
            }

            for (int model_idx = 0; model_idx < num_estimated_models; model_idx++) {
                // get score of new estimated model
                lo_score = quality->getScore(lo_models[model_idx]);

                if (best_model_score.isBetter(lo_score))
                    continue;

                if (lo_score.isBetter(new_model_score)) {
                    // update best model
                    lo_models[model_idx].copyTo(new_model);
                    new_model_score = lo_score;
                }
            }

            if (num_inliers_of_best_model < new_model_score.inlier_number)
                // update inliers of the best model.
                num_inliers_of_best_model = quality->getInliers(new_model,inliers_of_best_model);

        }
        return true;
    }
    Ptr<LocalOptimization> clone(int state) const override {
        return makePtr<InnerLocalOptimizationImpl>(estimator->clone(), quality->clone(),
           lo_sampler->clone(state), (int)inliers_of_best_model.size(), lo_inner_max_iterations);
    }
};
Ptr<InnerLocalOptimization> InnerLocalOptimization::create
(const Ptr<Estimator> &estimator_, const Ptr<Quality> &quality_,
       const Ptr<Sampler> &lo_sampler_, int points_size, int lo_inner_iterations_) {
    return makePtr<InnerLocalOptimizationImpl>(estimator_, quality_, lo_sampler_,
            points_size, lo_inner_iterations_);
}

/////////////////////////////////////////// FINAL MODEL POLISHER ////////////////////////
class LeastSquaresPolishingImpl : public LeastSquaresPolishing {
private:
    const Ptr<Estimator> estimator;
    const Ptr<Quality> quality;
    Score score;
    int lsq_iterations;
    std::vector<int> inliers;
    std::vector<Mat> models;
    std::vector<double> weights;
public:

    LeastSquaresPolishingImpl(const Ptr<Estimator> &estimator_, const Ptr<Quality> &quality_,
            int points_size, int lsq_iterations_) :
            estimator(estimator_), quality(quality_) {
        lsq_iterations = lsq_iterations_;
        // allocate memory for inliers array and models
        inliers = std::vector<int>(points_size);
        models = std::vector<Mat>(estimator->getMaxNumSolutionsNonMinimal());
    }

    bool polishSoFarTheBestModel(const Mat &model, const Score &best_model_score,
                                 Mat &new_model, Score &out_score) override {
        // get inliers from input model
        int inlier_number = quality->getInliers(model, inliers);
        if (inlier_number < estimator->getMinimalSampleSize())
            return false;

        out_score = Score(); // set the worst case

        // several all-inlier least-squares refines model better than only one but for
        // big amount of points may be too time-consuming.
        for (int lsq_iter = 0; lsq_iter < lsq_iterations; lsq_iter++) {
            bool model_updated = false;

            // estimate non minimal models with all inliers
            const int num_models = estimator->estimateModelNonMinimalSample(inliers,
                                                      inlier_number, models, weights);
            for (int model_idx = 0; model_idx < num_models; model_idx++) {
                score = quality->getScore(models[model_idx]);

                if (best_model_score.isBetter(score))
                    continue;

                if (score.isBetter(out_score)) {
                    models[model_idx].copyTo(new_model);
                    out_score = score;
                    model_updated = true;
                }
            }

            if (!model_updated)
                // if model was not updated at the first iteration then return false
                // otherwise if all-inliers LSQ has not updated model then no sense
                // to do it again -> return true (model was updated before).
                return lsq_iter > 0;

            // if number of inliers doesn't increase more than 5% then break
            if (fabs(static_cast<double>(out_score.inlier_number) - static_cast<double>
                (best_model_score.inlier_number)) / best_model_score.inlier_number < 0.05)
                return true;

            if (lsq_iter != lsq_iterations - 1)
                // if not the last LSQ normalization then get inliers for next normalization
                inlier_number = quality->getInliers(new_model, inliers);
        }
        return true;
    }
};
Ptr<LeastSquaresPolishing> LeastSquaresPolishing::create (const Ptr<Estimator> &estimator_,
         const Ptr<Quality> &quality_, int points_size, int lsq_iterations_) {
    return makePtr<LeastSquaresPolishingImpl>(estimator_, quality_, points_size, lsq_iterations_);
}
}}

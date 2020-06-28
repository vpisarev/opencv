// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../usac.hpp"

namespace cv { namespace usac {
class RansacOutputImpl : public RansacOutput {
private:
    Mat model;
    int seconds, milliseconds, microseconds;
    // vector of number_inliers size
    std::vector<int> inliers;
    // vector of points size, true if inlier, false-outlier
    std::vector<bool> inliers_mask;
    // vector of points size, value of i-th index corresponds to error of i-th point if i is inlier.
    std::vector<double> errors;

    // the best found score of RANSAC
    double score;

    int time_mcs, number_inliers;
    int number_iterations; // number of iterations of main RANSAC
    int number_estimated_models, number_good_models;
public:
    RansacOutputImpl (const Mat &model_,
        const std::vector<bool> &inliers_mask_,
        int time_mcs_, double score_,
        int number_inliers_, int number_iterations_,
        int number_estimated_models_,
        int number_good_models_) {

        model = model_.clone();

        inliers.reserve(number_inliers_);
        const int num_pts = inliers_mask_.size();
        for (int i = 0; i < num_pts; i++)
            if (inliers_mask_[i])
                inliers.emplace_back(i);
        inliers_mask = inliers_mask_;

        time_mcs = time_mcs_;

        score = score_;
        number_inliers = number_inliers_;
        number_iterations = number_iterations_;
        number_estimated_models = number_estimated_models_;
        number_good_models = number_good_models_;

        microseconds = time_mcs % 1000;
        milliseconds = ((time_mcs - microseconds)/1000) % 1000;
        seconds = ((time_mcs - 1000*milliseconds - microseconds)/(1000*1000)) % 60;
    }

    /*
     * Return inliers' indices.
     * size of vector = number of inliers
     */
    const std::vector<int > &getInliers() const override {
        return inliers;
    }

    /*
     * Return inliers mask. Vector of points size. 1-inlier, 0-outlier.
     */
    const std::vector<bool> &getInliersMask() const override { return inliers_mask; }

    /*
     * Return inliers' errors. Vector of points size.
     */
    int getTimeMicroSeconds() const override {return time_mcs; }
    int getTimeMicroSeconds1() const override {return microseconds; }
    int getTimeMilliSeconds2() const override {return milliseconds; }
    int getTimeSeconds3() const override {return seconds; }
    int getNumberOfInliers() const override { return number_inliers; }
    int getNumberOfMainIterations() const override { return number_iterations; }
    int getNumberOfGoodModels () const override { return number_good_models; }
    int getNumberOfEstimatedModels () const override { return number_estimated_models; }
    const Mat &getModel() const override { return model; }
};

Ptr<RansacOutput> RansacOutput::create(const Mat &model_,
            const std::vector<bool> &inliers_mask_, int time_mcs_, double score_,
           int number_inliers_, int number_iterations_,
           int number_estimated_models_, int number_good_models_) {
    return makePtr<RansacOutputImpl>(model_, inliers_mask_, time_mcs_,
            score_, number_inliers_, number_iterations_,
            number_estimated_models_, number_good_models_);
}

class Ransac {
protected:
    const Model &params;
    const Estimator &estimator;
    Quality &quality;
    Sampler &sampler;
    TerminationCriteria &termination_criteria;
    ModelVerifier &model_verifier;
    Degeneracy &degeneracy;
    LocalOptimization * const local_optimization;
    FinalModelPolisher &model_polisher;

    int points_size;
public:

    Ransac (const Model &params_, int points_size_, const Estimator &estimator_, Quality &quality_,
            Sampler &sampler_, TerminationCriteria &termination_criteria_,
            ModelVerifier &model_verifier_, Degeneracy &degeneracy_,
            LocalOptimization * const local_optimization_, FinalModelPolisher &model_polisher_) :

            params (params_), estimator (estimator_), quality (quality_), sampler (sampler_),
            termination_criteria (termination_criteria_), model_verifier (model_verifier_), 
            degeneracy (degeneracy_), local_optimization (local_optimization_),
            model_polisher (model_polisher_) {
        points_size = points_size_;

        // do some asserts
        assert(params.getSampleSize() == estimator.getMinimalSampleSize());
    }

    /*
     * Reset RANSAC and all components (e.g., sampler) to initial state. Assume parameters of RANSAC are not changed.
     */
    void reset () {
        sampler.reset();
        termination_criteria.reset();
        model_verifier.reset();
        // other components don't need to be reset.
    }

    bool run(Ptr<RansacOutput> &ransac_output) {
        if (points_size < params.getSampleSize())
            return false;

        auto begin_time = std::chrono::steady_clock::now();

        // reallocate memory for models
        // do not use loop over models (i.e., auto &m : models)!
        std::vector<Mat> models(estimator.getMaxNumSolutions());

        // allocate memory for sample
        std::vector<int> sample (estimator.getMinimalSampleSize());

        // check if LO
        const bool LO = params.getLO() != LocalOptimMethod ::NullLO;
        const bool is_magsac = params.getLO() == LocalOptimMethod::SIGMA;
        // only for test
        int number_of_estimated_models = 0, number_of_good_models = 0;
        double avg_num_models_per_sample = 0;

        Score current_score, best_score, lo_score, best_non_degenerate_model_score;

        Mat best_model;

        // start measure time termination criteria. If time limit was set then ransac will terminate when time exceeds it.
        termination_criteria.startMeasureTime();

        // number of iterations (number of tested samples)
        int iters = 0;
        while (!termination_criteria.terminate(++iters)) {
            sampler.generateSample(sample);

            auto number_of_models = estimator.estimateModels(sample, models);

            // test
            avg_num_models_per_sample += number_of_models;
            //

            for (int i = 0; i < number_of_models; i++) {
                // test
                number_of_estimated_models++;

                bool is_good_model = model_verifier.isModelGood(models[i]);

                if (!is_good_model && iters > params.getMaxNumHypothesisToTestBeforeRejection()) {
                    // do not skip bad model until predefined number of iterations reached
                    continue;
                }
                // test
                number_of_good_models++;

                if (is_magsac) {
                    if (iters == 1)
                        models[i].copyTo(best_model);
                    local_optimization->refineModel(best_model, best_score, models[i], current_score);
                } else {
                    if (! model_verifier.getScore(current_score))
                        current_score = quality.getScore(models[i]);
                }

                if (current_score.better(best_score)) {
                    // if number of non degenerate models is zero then input model is good
                    int num_non_degenerate_models = degeneracy.recoverIfDegenerate(sample, models[i]);

                    if (num_non_degenerate_models > 0) {
                        const std::vector<Mat> &non_degenerate_models = degeneracy.getRecoveredModels();
                        // Iterate over non degenerate models to find their quality and save the best one.
                        Mat best_non_degenerate_model = non_degenerate_models[0];
                        best_non_degenerate_model_score = quality.getScore(non_degenerate_models[0]);
                        for (int m = 1; m < num_non_degenerate_models; m++) {
                            current_score = quality.getScore(non_degenerate_models[m]);
                            if (current_score.better(best_non_degenerate_model_score)) {
                                best_non_degenerate_model_score = current_score;
                                best_non_degenerate_model = non_degenerate_models[m];
                            }
                        }

                        // check if best non degenerate model is better than so far the best model
                        if (best_non_degenerate_model_score.better(best_score)) {
                            best_score = best_non_degenerate_model_score;
                            best_non_degenerate_model.copyTo(best_model);
                        } else {
                            // non degenerate models are worse then so far the best model.
                            continue;
                        }
                    } else {
                        // copy current score to best score
                        best_score = current_score;
                        // remember best model
                        models[i].copyTo(best_model);
                        quality.setBestScore(best_score.score);
                    }

                    // update upper bound of iterations
                    termination_criteria.update(best_model, best_score.inlier_number);
                    if (termination_criteria.terminate(iters))
                        break;

                    if (LO && !is_magsac) {
                        // update model by Local optimizaion
                        Mat lo_model;
                        if (local_optimization->refineModel(best_model, best_score, lo_model, lo_score)) {
                            if (lo_score.better(best_score)) {
                                best_score = lo_score;
                                lo_model.copyTo(best_model);
                                quality.setBestScore(best_score.score);

                                // update termination again
                                termination_criteria.update(best_model, best_score.inlier_number);
                            }
                        }
                    }
                } // end of if so far the best score
            } // end loop of number of models
        } // end main while loop

        // if best model has 0 inliers then return fail
        if (best_score.inlier_number == 0)
            return false;

        // polish final model
        if (params.getFinalPolisher() != PolishingMethod::NonePolisher) {
            Mat polished_model;
            Score polisher_score;
            if (model_polisher.polishSoFarTheBestModel (best_model, best_score,
                                                        polished_model, polisher_score)) {
                if (polisher_score.better(best_score)) {
                    best_score = polisher_score;
                    polished_model.copyTo(best_model);
                }
            }
        }

        // ================= here is ending ransac main implementation ===========================
        if (params.getTrace() >= 1) {
            // get final inliers from the best model
            std::vector<bool> inliers_mask (points_size);
            quality.getInliers(best_model, inliers_mask);

            // Store results
            ransac_output = RansacOutput::create(best_model, inliers_mask,
                std::chrono::duration_cast<std::chrono::microseconds>
               (std::chrono::steady_clock::now() - begin_time).count(), best_score.score,
               best_score.inlier_number, iters, number_of_estimated_models, number_of_good_models);
        }
        return true;
    }
};

Mat findHomography (InputArray srcPoints, InputArray dstPoints, int method, double thr,
    OutputArray mask, const int maxIters, const double confidence) {

    Mat points;
    hconcat(srcPoints.getMat(), dstPoints.getMat(), points);
    points.convertTo(points, CV_64F); // convert to double
    int points_size = points.rows;

    Ptr<Model> params = Model::create(thr, EstimationMethod ::Homography,
                                      SamplingMethod::Uniform, confidence, maxIters, ScoreMethod::MSAC);

    params->setTrace(mask.needed());
    params->setLocalOptimization(LocalOptimMethod ::InLORsc);
    params->setPolisher(PolishingMethod ::LSQPolisher);
    params->setVerifier(VerificationMethod ::SprtVerifier);

    RNG rng;
    Ptr<Error> error = ReprojectionErrorForward::create(points);
    Ptr<Degeneracy> degeneracy = HomographyDegeneracy::create(points, params->getSampleSize());
    Ptr<MinimalSolver> h_min = HomographyMinimalSolver4ptsGEM::create(points);
    Ptr<NonMinimalSolver> h_non_min = HomographyNonMinimalSolver::create(points);
    Ptr<Estimator> estimator = HomographyEstimator::create(h_min, h_non_min, degeneracy);
    Ptr<Quality> quality = MsacQuality::create(points_size, params->getThreshold(), error);
    Ptr<ModelVerifier> verifier = SPRT::create(rng, error, points_size, params->getSampleSize(),
                  params->getThreshold(), params->getSPRTepsilon(), params->getSPRTdelta(),
                  params->getTimeForModelEstimation(), params->getSPRTavgNumModels(), 1);
    Ptr<FinalModelPolisher> polisher = LeastSquaresPolishing::create(estimator, quality, degeneracy, points_size);
    Ptr<Sampler> sampler = UniformSampler::create(rng, params->getSampleSize(), points_size);
    Ptr<TerminationCriteria> termination = StandardTerminationCriteria::create(
            params->getConfidence(), points_size, params->getSampleSize(), params->getMaxIters(), params->isTimeLimit());

    Ptr<Sampler> lo_sampler = UniformSampler::create(rng, params->getMaxSampleSizeLO(), points_size);
    Ptr<LocalOptimization> inner_lo_rsc = InnerLocalOptimization::create(estimator, quality, lo_sampler, degeneracy, points_size);

    Ransac ransac (*params, points_size, *estimator, *quality, *sampler,
                    *termination, *verifier, *degeneracy, inner_lo_rsc, *polisher);

    Ptr<RansacOutput> ransac_output;
    if (!ransac.run (ransac_output)) return Mat();

    if (mask.needed()) {
        const std::vector<bool> &inliers_mask = ransac_output->getInliersMask();
        mask.create(1, points_size, CV_8U);
        auto * maskptr = mask.getMat().ptr<uchar>();
        for (int i = 0; i < points_size; i++)
            maskptr[i] = (uchar) inliers_mask[i];
    }
    return ransac_output->getModel();
}

class ModelImpl : public Model {
private:
    // main parameters:
    double threshold, confidence;
    int sample_size, max_iterations;

    EstimationMethod estimator;
    SamplingMethod sampler;
    ScoreMethod score;

    // optional default parameters:

    // termination parameters
    // -> time limit for RANSAC running
    bool time_limit = false;
    int max_time_mcs = std::numeric_limits<int>::max();

    // for neighborhood graph
    int k_nearest_neighbors = 8; // for FLANN
    int cell_size = 25; // pixels, for grid neighbors searching
    double radius = 15; // pixels, for radius-search neighborhood graph
    int flann_search_params = 32;
    NeighborSearchMethod neighborsType = NeighborSearchMethod::Grid;

    // Local Optimization parameters
    LocalOptimMethod lo = LocalOptimMethod ::NullLO;
    int lo_sample_size=14, lo_inner_iterations=15, lo_iterative_iterations=5,
            lo_threshold_multiplier=4, lo_iter_sample_size = 30;
    bool sample_size_limit = true; // parameter for Iterative LO-RANSAC
    const int num_iters_before_LO = 10;

    // Graph cut parameters
    const double spatial_coherence_term = 0.1;

    // apply polisher for final RANSAC model
    PolishingMethod polisher = PolishingMethod ::LSQPolisher;

    // preemptive verification test
    VerificationMethod verifier = VerificationMethod ::NullVerifier;
    const int max_hypothesis_test_before_verification = 5;

    // number of points to be verified for Tdd test
    int num_tdd = 1;

    // sprt parameters
    double sprt_eps, sprt_delta, avg_num_models, time_for_model_est;

    // randomization of RANSAC
    bool reset_random_generator = false;

    // fundamental degeneracy
    double homography_thr = 4.;

    // MLESAC parameters
    int iters_EM = 3;
    double outlier_range = 400., gamma_init = 0.5;

    // PROSAC parameters
    int growth_max_samples = 2e5;
    double beta_prob = 0.05, non_rand_prob = 0.95;

    // density sort for PROSAC, if points are not ordered before, then they could be sorted by density
    // inside USAC framework
    bool density_sort = false;

    // estimator error
    ErrorMetric est_error;

    // fill with zeros (if is not known)
    int img1_width = 0, img1_height = 0, img2_width = 2, img2_height = 0;

    // progressive napsac
    double relax_coef = 0.1;
    int sampler_length = 20;
    // for building neighborhood graphs
    const std::vector<int> grid_cell_number = {16, 8, 4, 2};

    //for final least squares polisher
    int final_lsq_iters = 3;

    int trace = 2;
    Mat descriptor;

    // magsac parameters for H, F, E
    double DoF = 4, sigma_quantile = 3.64, upper_incomplete_of_sigma_quantile = 0.00365,
    lower_incomplete_of_sigma_quantile = 1.30122, C = 0.25, maximum_thr = 10.;
public:
    ModelImpl (double threshold_, EstimationMethod estimator_, SamplingMethod sampler_, double confidence_=0.95,
               int max_iterations_=5000, ScoreMethod score_ =ScoreMethod::RANSAC) {
        estimator = estimator_;
        switch (estimator_) {
            case (EstimationMethod::Similarity): sample_size = 2; est_error = ErrorMetric ::FORW_REPR_ERR; break;
            case (EstimationMethod::Affine): sample_size = 3; est_error = ErrorMetric ::FORW_REPR_ERR; break;
            case (EstimationMethod::Homography): sample_size = 4; est_error = ErrorMetric ::FORW_REPR_ERR;
                // time_for_model_est = 1.03;
                break;
            case (EstimationMethod::HomographyQR): sample_size = 4; est_error = ErrorMetric ::FORW_REPR_ERR;
                // time_for_model_est = 5.5099;
                break;
            case (EstimationMethod::Fundamental): sample_size = 7; est_error = ErrorMetric ::SAMPSON_ERR; break;
            case (EstimationMethod::Fundamental8): sample_size = 8; est_error = ErrorMetric ::SAMPSON_ERR; break;
            case (EstimationMethod::Essential): sample_size = 5; est_error = ErrorMetric ::SGD_ERR; break;
            case (EstimationMethod::P3P): sample_size = 3; est_error = ErrorMetric ::RERPOJ; break;
            case (EstimationMethod::P6P): sample_size = 6; est_error = ErrorMetric ::RERPOJ; break;
            default: assert(0 && "Estimator has not implemented yet!");
        }

        // TODO: experiment with SPRT values and accordingly for GEM and QR solvers.
        sprt_eps = 0.05; // lower bound estimate is 5% of inliers
        sprt_delta = 0.01; avg_num_models = 1;
        time_for_model_est = 100;
        // for lower time sprt becomes super strict, so for the same iteration number ransac will always be faster but less accurate.
        if (isEssential() || isFundamental()) {
            lo_sample_size = 14;
            // epipolar geometry usually have more inliers
            sprt_eps = 0.1; // lower bound estimate is 10% of inliers
            sprt_delta = 0.05;
            if (sample_size == 7) { // F seven points
                avg_num_models = 2.38;
                time_for_model_est = 200;
            } else if (sample_size == 5) { // E five points
                avg_num_models = 4;
                time_for_model_est = 400;
            } else if (sample_size == 6) { // E six points
                avg_num_models = 5;
            }
        } else if (estimator_ == EstimationMethod::P3P) {
            avg_num_models = 2;
            time_for_model_est = 300;
        } else if (estimator_ == EstimationMethod::P6P) {
            avg_num_models = 1;
            time_for_model_est = 250;
        }

        /*
         * Measured reprojected error in homography is (x-x')^2 + (y-y)^2 (without squared root),
         * so threshold must be squared.
         */
        threshold = threshold_;
        if (est_error == ErrorMetric::FORW_REPR_ERR || est_error == ErrorMetric::SYMM_REPR_ERR || est_error == ErrorMetric ::RERPOJ)
            threshold *= threshold_;

        sampler = sampler_;
        confidence = confidence_;
        max_iterations = max_iterations_;
        score = score_;
    }
    void setVerifier (VerificationMethod verifier_) override {
        verifier = verifier_;
    }
    void setPolisher (PolishingMethod polisher_) override {
        polisher = polisher_;
    }
    void setError (ErrorMetric error_) override {
        est_error = error_;
    }
    void setLocalOptimization (LocalOptimMethod lo_) override {
        lo = lo_;
    }
    void setKNearestNeighhbors (int knn_) override {
        k_nearest_neighbors = knn_;
    }
    void setNeighborsType (NeighborSearchMethod neighbors) override {
        neighborsType = neighbors;
    }
    void setCellSize (int cell_size_) override {
        cell_size = cell_size_;
    }
    void setResetRandomGenerator (bool reset) override {
        reset_random_generator = reset;
    }
    void setTrace (int trace_) override { trace = trace_; }
    int getTrace () const override { return trace; }
    void setSPRT (double sprt_eps_ = 0.005, double sprt_delta_ = 0.0025,
                  double avg_num_models_ = 1, double time_for_model_est_ = 5e2) override {
        sprt_eps = sprt_eps_; sprt_delta = sprt_delta_;
        avg_num_models = avg_num_models_; time_for_model_est = time_for_model_est_;
    }
    void setImageSize (int img1_width_, int img1_height_,
                       int img2_width_, int img2_height_) override {
        img1_width = img1_width_, img1_height = img1_height_,
        img2_width = img2_width_, img2_height = img2_height_;
    }
    NeighborSearchMethod getNeighborsSearch () const override {
        return neighborsType;
    }
    int getKNN () const override {
        return k_nearest_neighbors;
    }
    ErrorMetric getError () const override {
        return est_error;
    }
    EstimationMethod getEstimator () const override {
        return estimator;
    }
    inline void setDescriptor(const Mat &desc) override {
        desc.copyTo(descriptor);
    }
    inline const Mat& getDescriptor () const override {
        return descriptor;
    }
    inline Mat& getRefDescriptor () override {
        return descriptor;
    }
    int getSampleSize () const override {
        return sample_size;
    }
    int getMaxTimeMcs() const override {
        return max_time_mcs;
    }
    bool resetRandomGenerator () const override {
        return reset_random_generator;
    }
    int getMaxNumHypothesisToTestBeforeRejection() const override {
        return max_hypothesis_test_before_verification;
    }
    PolishingMethod getFinalPolisher () const override {
        return polisher;
    }
    int getLOThresholdMultiplier() const override {
        return lo_threshold_multiplier;
    }
    int getLOIterativeSampleSize() const override {
        return lo_iter_sample_size;
    }
    int getImage1Width () const override {
        return img1_width;
    }
    int getImage1Height () const override {
        return img1_height;
    }
    int getImage2Width () const override {
        return img2_width;
    }
    int getImage2Height () const override {
        return img2_height;
    }
    int getLOIterativeMaxIters() const override {
        return lo_iterative_iterations;
    }
    int getLOInnerMaxIters() const override {
        return lo_inner_iterations;
    }
    LocalOptimMethod getLO () const override {
        return lo;
    }
    ScoreMethod getScore () const override {
        return score;
    }
    int getMaxIters () const override {
        return max_iterations;
    }
    double getConfidence () const override {
        return confidence;
    }
    bool isTimeLimit () const override {
        return time_limit;
    }
    double getThreshold () const override {
        return threshold;
    }
    VerificationMethod getVerifier () const override {
        return verifier;
    }
    SamplingMethod getSampler () const override {
        return sampler;
    }
    int getMaxSampleSizeLO () const override {
        return lo_inner_iterations;
    }
    int getMaxSampleSizeLOiterative () const override {
        return lo_iter_sample_size;
    }
    double getSPRTdelta () const override {
        return sprt_delta;
    }
    double getSPRTepsilon () const override {
        return sprt_eps;
    }
    double getSPRTavgNumModels () const override {
        return avg_num_models;
    }
    int getCellSize () const override {
        return cell_size;
    }
    double getTimeForModelEstimation () const override {
        return time_for_model_est;
    }
    bool isSampleLimit () const override {
        return sample_size_limit;
    }
    double getRelaxCoef () const override {
        return relax_coef;
    }
    const std::vector<int> &getGridCellNumber () const override {
        return grid_cell_number;
    }
    bool isFundamental () const override {
        return estimator == EstimationMethod ::Fundamental ||
               estimator == EstimationMethod ::Fundamental8;
    }
    bool isHomography () const override {
        return estimator == EstimationMethod ::Homography || estimator == EstimationMethod ::HomographyQR;
    }
    bool isEssential () const override {
        return estimator == EstimationMethod ::Essential;
    }
    bool isPnP() const override {
        return estimator == EstimationMethod ::P3P || estimator == EstimationMethod ::P6P;
    }
};

Ptr<Model> Model::create(double threshold_, EstimationMethod estimator_, SamplingMethod sampler_,
                         double confidence_, int max_iterations_, ScoreMethod score_) {
    return makePtr<ModelImpl>(threshold_, estimator_, sampler_, confidence_,
                              max_iterations_, score_);
}
}}
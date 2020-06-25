// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_USAC_USAC_HPP
#define OPENCV_USAC_USAC_HPP

namespace cv { namespace usac {
// Abstract Error class
class Error : public Algorithm {
public:
    virtual void setModelParameters (const Mat &model) = 0;
    virtual float getError (int point_idx) const = 0;
};

// Symmetric Reprojected Error
class ReprojectedErrorSymmetric : public Error {
public:
    static Ptr<ReprojectedErrorSymmetric> create(const Mat &points);
};

// Forward Reprojected Error
class ReprojectedErrorForward : public Error {
public:
    static Ptr<ReprojectedErrorForward> create(const Mat &points);
};

/*
* Class for normalizing transformations of points.
*/
class NormTransform : public Algorithm {
public:
    virtual void getNormTransformation (Mat &norm_points, const std::vector<int> &sample,
                                        int sample_number, Mat &T1, Mat &T2) const = 0;
    static Ptr<NormTransform> create (const Mat &points);
};

/////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////// SOLVER ///////////////////////////////////////////
class MinimalSolver : public Algorithm {
public:
    /*
     * Estimate models from minimal sample
     * models.size() == output
     */
    virtual int estimate (const std::vector<int> &sample, std::vector<Mat> &models) = 0;
    /*
     * Get minimal sample size required for estimation.
     */
    virtual int getSampleSize() const = 0;
    /*
     * Get maximum number of possible solutions.
     */
    virtual int getMaxNumberOfSolutions () const = 0;
};

//-------------------------- HOMOGRAPHY MATRIX -----------------------
class HomographyMinimalSolver4ptsQR : public MinimalSolver {
public:
    static Ptr<HomographyMinimalSolver4ptsQR> create(const Mat &points_);
};

class HomographyMinimalSolver4ptsGEM : public MinimalSolver {
public:
    static Ptr<HomographyMinimalSolver4ptsGEM> create(const Mat &points_);
};

//////////////////////////////////////// NON MINIMAL SOLVER ///////////////////////////////////////
class NonMinimalSolver : public Algorithm {
public:
    /*
     * Estimate models from non minimal sample
     * models.size() == output
     */
    virtual int estimate (const std::vector<int> &sample, int sample_size,
                          std::vector<Mat> &models, const std::vector<double> &weights) = 0;

    /*
     * Get minimal sample size required for non-minimal estimation.
     */
    virtual int getMinimumRequiredSampleSize() const = 0;

    /*
     * Get maximum number of possible solutions.
     */
    virtual int getMaxNumberOfSolutions () const = 0;
};

class HomographyNonMinimalSolver : public NonMinimalSolver {
public:
    static Ptr<HomographyNonMinimalSolver> create(const Mat &points_);
};

//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////// DEGENERACY //////////////////////////////////
class Degeneracy : public Algorithm {
private:
    std::vector<Mat> recovered_models;
public:
    virtual ~Degeneracy() override = default;
    /*
     * Check if sample causes degenerate configurations.
     * For example, test if points are collinear.
     */
    inline virtual bool isSampleGood (const std::vector<int> &sample) const {
        return true;
    }
    /*
     * Check if model satisfies constraints.
     * For example, test if epipolar geometry satisfies oriented constraint.
     */
    inline virtual bool isModelValid (const Mat &model, const std::vector<int> &sample) const {
        return true;
    }
    /*
     * Check if model is degenerate.
     * Firstly test if sample is degenerate. Then check model itself.
     * For example, for fundamental matrix estimation checks if no more than 5 points lie on the dominant plane
     * and then check if model satisfies oriented constraint.
     * This method could be combination of two previous functions, although is more general.
     */
    virtual bool isModelDegenerate (const Mat &model, const std::vector<int> &sample) const {
        return false;
    }
    /*
     * Check if model satisfies rank constraint. For example, homography matrix must have rank 3; fundamental or
     * essential matrix must have rank 2.
     */
    virtual bool satisfyRankConstraint (const Mat &model) const {
        return true;
    }
    /*
     * Recover rank constraint.
     * Primarily for epipolar geometry estimation. If matrix is of rank 3 then do SVD to get rank 2.
     * Return: true if recovered successfully, false - otherwise.
     */
    virtual bool recoverRank (Mat &model) const {
        return true;
    }
    /*
     * Fix degenerate model.
     * For example, for Fundamental matrix estimation.
     * Return:
     * -1 model is degenerate failed to recover:
     * Otherwise: return number of recovered models. If output number is 0 then model is not degenerate.
     */
    virtual int recoverIfDegenerate (const std::vector<int> &sample, const Mat &best_model) {
        return 0;
    }
    virtual const std::vector<Mat> &getRecoveredModels() const {
        return recovered_models;
    }
    /*
     * Get maximum number of possible recovered models.
     * For example, for Fundamental matrix maximum 5 non-degenerate models could be computed.
     */
    virtual int getMaximumNumberOfRecoveredModels () const {
        return 0;
    }
};

class HomographyDegeneracy : public Degeneracy {
public:
    static Ptr<HomographyDegeneracy> create(const Mat &points_, int sample_size_);
};


/////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////// ESTIMATOR //////////////////////////////////
class Estimator : public Algorithm{
public:
    /*
     * Estimate models with minimal solvers.
     * Return number of valid solutions after estimation.
     * Return models accordingly to number of solutions.
     * Note, vector of models must allocated before.
     * Note, not all degenerate tests including in estimation.
     */
    virtual int
    estimateModels (const std::vector<int> &sample, std::vector<Mat> &models) const = 0;

    /*
     * Estimate model with non-minimal solver.
     * Return true if model was estimated successfully. Otherwise return false.
     * Return only one valid model.
     * Note, not all degenerate tests including in estimation.
     */
    // todo: think about weights reference. Maybe setter?
    virtual int
    estimateModelNonMinimalSample (const std::vector<int> &sample, int sample_size,
                       std::vector<Mat> &models, const std::vector<double> &weights) const = 0;

    /*
     * Get number of samples required for minimal estimation.
     */
    virtual int getMinimalSampleSize () const = 0;

    /*
     * Get minimal number of samples required for non-minimal estimation.
     */
    virtual int getNonMinimalSampleSize () const = 0;

    /*
     * Get maximum number of possible solutions of minimal estimation.
     */
    virtual int getMaxNumSolutions () const = 0;

    /*
     * Get maximum number of possible solutions of non-minimal estimation.
     */
    virtual int getMaxNumSolutionsNonMinimal () const = 0;
};

class HomographyEstimator : public Estimator {
public:
    static Ptr<HomographyEstimator> create (const Ptr<MinimalSolver> &min_solver_,
            const Ptr<NonMinimalSolver> &non_min_solver_, const Ptr<Degeneracy> &degeneracy_);
};

//////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////// SCORE ///////////////////////////////////////////
class Score {
public:
    int inlier_number;
    double score;
    Score () {
        inlier_number = 0;
        score = std::numeric_limits<double>::max();
    }
    Score (int inlier_number_, double score_) {
        inlier_number = inlier_number_;
        score = score_;
    }
    /*
     * Compare two scores. Objective is minimization of score. Lower is better.
     */
    inline bool better(const Score &score2) const {
        return score < score2.score;
    }
};

////////////////////////////////////////// QUALITY ///////////////////////////////////////////
class Quality : public Algorithm {
public:
    virtual ~Quality() override = default;
    /*
     * Calculates number of inliers and score for current model.
     * @score: of class Score contains inlier_number and score variables.
     * @model: Mat current model, e.g., H matrix.
     * @get_inliers: if true inliers will be stored to @inliers.
     * @inliers: array of inliers, contains inlier indices, which are stored from the beginning of array.
     * Note, all pointers must be allocated, @inliers should be allocated to max size (=points size)
     */
    virtual Score getScore (const Mat &model, double threshold, bool get_inliers, std::vector<int> &inliers) const = 0;
    virtual Score getScore (const Mat &model, bool get_inliers, std::vector<int> &inliers) const = 0;
    virtual Score getScore (const Mat &model) const = 0;

    // make sure than estimator is updated with the latest model.
    // make isInlier() also virtual for MAGSAC, because threshold is unknown.
    // for other quality classes inlier is point which error is less than threshold.
    virtual bool isInlier (int point_idx) const = 0;

    // set model to estimator
    virtual void setModel (const Mat &model) const = 0;

    // get inliers with initialized threshold
    virtual int getInliers (const Mat &model, std::vector<int> &inliers) const = 0;

    // get inliers for given threshold
    virtual int getInliers (const Mat &model, std::vector<int> &inliers, double thr) const = 0;

    /*
     * Set the best score, so evaluation of the model can terminate earlier
     */
    virtual void setBestScore (double best_score_) = 0;
    /*
     * @inliers_mask: true if point i is inlier, false - otherwise.
     */
    virtual int getInliers (const Mat &model, std::vector<bool> &inliers_mask) const = 0;
};

class RansacQuality : public Quality {
public:
    static Ptr<RansacQuality> create(int points_size_, double threshold_, const Ptr<Error> &error_);
};

class MsacQuality : public Quality {
public:
    static Ptr<MsacQuality> create(int points_size_, double threshold_, const Ptr<Error> &error_);
};

//////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////// MODEL VERIFIER ////////////////////////////////////
class ModelVerifier : public Algorithm {
public:
    virtual ~ModelVerifier() override = default;
    /*
     * Returns true if model is good, false - otherwise.
     */
    inline virtual bool isModelGood(const Mat &model) {
        return true;
    }
    /*
     * Return true if score was computed during evaluation.
     */
    inline virtual bool getScore(Score &score) const {
        return false;
    }
    /*
     * Reset ModelVerification to initial state.
     * Assume parameters of ModelVerification (e.g., points size) are not changed.
     * SPRT as sequential decision making requires to be reset.
     */
    virtual void reset () {}
};

struct SPRT_history {
    /*
     * delta:
     * The probability of a data point being consistent
     * with a ‘bad’ model is modeled as a probability of
     * a random event with Bernoulli distribution with parameter
     * δ : p(1|Hb) = δ.

     * epsilon:
     * The probability p(1|Hg) = ε
     * that any randomly chosen data point is consistent with a ‘good’ model
     * is approximated by the fraction of inliers ε among the data
     * points

     * A is the decision threshold, the only parameter of the Adapted SPRT
     */
    double epsilon, delta, A;
    // number of samples processed by test
    int tested_samples; // k
    SPRT_history () {
        tested_samples = 0;
    }
};

/*
* Matas, Jiri, and Ondrej Chum. "Randomized RANSAC with sequential probability ratio test."
* Tenth IEEE International Conference on Computer Vision (ICCV'05) Volume 1. Vol. 2. IEEE, 2005.
*/
class SPRT : public ModelVerifier {
public:
    /*
     * Return constant reference of vector of SPRT histories for SPRT termination.
     */
    virtual const std::vector<SPRT_history> &getSPRTvector () const = 0;
};

///////////////////////////////// SPRT VERIFIER UNIVERSAL /////////////////////////////////////////
class SPRTverifier : public SPRT {
public:
    static Ptr<SPRTverifier> create (RNG &rng, const Ptr<Quality> &quality_, int points_size_,
         int sample_size_, double prob_pt_of_good_model, double prob_pt_of_bad_model,
         double time_sample, double avg_num_models);
};

///////////////////////////////////// SPRT VERIFIER MSAC //////////////////////////////////////////
class SPRTScore : public SPRT {
public:
    static Ptr<SPRTScore> create (RNG &rng, const Ptr<Error> &err_, int points_size_, int sample_size_,
         double inlier_threshold_, double prob_pt_of_good_model, double prob_pt_of_bad_model,
         double time_sample, double avg_num_models, bool binary_score);
};

/////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////// TERMINATION ///////////////////////////////////////////
class TerminationCriteria : public Algorithm {
public:
    virtual bool terminate(int current_iteration) const = 0;
    /*
     * After calling this function begin_time is set.
     * This function is used in the beginning of RANSAC.
     */
    virtual void startMeasureTime() = 0;
    /*
     * Updates necessary conditions for terminate() function.
     */
    virtual void update(const Mat &model, int inlier_number) = 0;
    /*
     * Return predicted number of iterations required for RANSAC
     */
    virtual int getPredictedNumberIterations() const = 0;
    /*
     * Resets Termination to initial state. Assume parameters of Termination
     * (e.g., max iterations) are not changed.
     */
    virtual void reset() = 0;
};

//////////////////////////////// STANDARD TERMINATION ///////////////////////////////////////////
class StandardTerminationCriteria : public TerminationCriteria {
public:
    static Ptr<StandardTerminationCriteria> create(double confidence, int points_size_,
               int sample_size_, int max_iterations_, bool is_time_limit_,
               int max_time_mcs_ = std::numeric_limits<int>::max());
};

//////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////// UTILS ////////////////////////////////////////////////
class Utils : public Algorithm {
public:
    static void random_shuffle (RNG &rng, std::vector<int> &array);
};

class Math {
public:
    /*
     * @points Nx4 array: x1 y1 x2 y2
     * @sample Mx1 array
     */
    static bool haveCollinearPoints(const Mat &points, const std::vector<int> &sample,
            double threshold=1);
    static Mat getSkewSymmetric(const Mat &v_);
    static Mat cross(const Mat &a_, const Mat &b_);
    static int rank3x3 (const Mat &A_);
    static void eliminateUpperTriangluar (double * a, int m, int n);
};

///////////////////////////////////////// RANDOM GENERATOR /////////////////////////////////////
class RandomGenerator : public Algorithm {
public:
    virtual ~RandomGenerator() override = default;
    // interval is <0, max_range);
    virtual void resetGenerator (int max_range) = 0;
    virtual void generateUniqueRandomSet (std::vector<int> &sample) = 0;
    virtual void setSubsetSize (int subset_sz) = 0;
    virtual int getRandomNumber () = 0;
};

class UniformRandomGenerator : public RandomGenerator {
public:
    static Ptr<UniformRandomGenerator> create (RNG &rng);
    static Ptr<UniformRandomGenerator> create (RNG &rng, int max_range, int subset_size_);
    virtual void generateUniqueRandomSet (std::vector<int> &sample, int subset_size, int max_range) = 0;
    virtual void generateUniqueRandomSet (std::vector<int> &sample, int max_range) = 0;
};

//////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////// SAMPLER ///////////////////////////////////////
class Sampler : public Algorithm {
public:
    virtual ~Sampler() override = default;

    // set new sample size
    virtual void setNewSampleSize (int sample_size) = 0;

    // set new points size
    virtual void setNewPointsSize (int points_size) = 0;

    // set new sample size and points size
    virtual void setNew (int sample_size, int points_size) = 0;
    /*
     * Generate sample. Fill @sample with indices of points.
     */
    virtual void generateSample (std::vector<int> &sample) = 0;
    /*
     * Generate sample for given points size
     */
    virtual void generateSample (std::vector<int> &sample, int points_size) = 0;
    /*
     * Generate sample for given sample size and points size.
     */
    virtual void generateSample (std::vector<int> &sample, int sample_size, int points_size) = 0;

    virtual int getSampleSize () const = 0;
    /*
     * Reset Sampler to initial state. Assume that parameters of Sampler (e.g., sample size) are not changed.
     * Sampler as Uniform or NAPSAC does not require reset() because sampling is independent.
     * However, PROSAC and P-NAPSAC requires.
     */
    virtual void reset () = 0;
};

////////////////////////////////////// UNIFORM SAMPLER ////////////////////////////////////////////
/*
* Choose uniformly m (sample size) points from N (points size).
* Uses Fisher-Yates shuffle.
*/
class UniformSampler : public Sampler {
public:
    static Ptr<UniformSampler> create(RNG &rng, int sample_size_, int points_size_);
};

///////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////// LOCAL OPTIMIZATION ////////////////////////////////////////
class LocalOptimization : public Algorithm {
public:
    virtual ~LocalOptimization() override = default;
    /*
     * Refine so-far-the-best RANSAC model in local optimization step.
     * @best_model: so-far-the-best model
     * @new_model: output refined new model.
     * @new_model_score: score of @new_model.
     * Returns bool if model was refined successfully, false - otherwise
     */
    virtual bool refineModel (const Mat &best_model, const Score &best_model_score,
                              Mat &new_model, Score &new_model_score) = 0;

    /*
     * Reset LocalOptimization to initial state. Assume parameters of LO
     * are not changed (e.g., non minimal sample size).
     */
    virtual void reset () {}
};

//////////////////////////////////// INNER LO ///////////////////////////////////////
class InnerLocalOptimization : public LocalOptimization {
public:
    static Ptr<InnerLocalOptimization>
    create(const Ptr<Estimator> &estimator_, const Ptr<Quality> &quality_,
           const Ptr<Sampler> &lo_sampler_, const Ptr<Degeneracy>  &degeneracy_,
           int points_size, int lo_inner_iterations_=15);
};

///////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////// FINAL MODEL POLISHER //////////////////////////////////////
class FinalModelPolisher : public Algorithm {
public:
    virtual ~FinalModelPolisher() override = default;
    /*
     * Polish so-far-the-best RANSAC model in the end of RANSAC.
     * @model: input final RANSAC model.
     * @new_model: output polished model.
     * @new_score: socre of output model.
     * Return true if polishing was successful, false - otherwise.
     */
    virtual bool polishSoFarTheBestModel (const Mat &model, const Score &best_model_score,
            Mat &new_model, Score &new_model_score) {
        return false;
    }
};

///////////////////////////////////// LEAST SQUARES POLISHER //////////////////////////////////////
class LeastSquaresPolishing : public FinalModelPolisher {
public:
    static Ptr<LeastSquaresPolishing> create (const Ptr<Estimator> &estimator_,
        const Ptr<Quality> &quality_, const Ptr<Degeneracy> &degeneracy_,
        int points_size, int lsq_iterations_=2);
};

/////////////////////////////////// RANSAC OUTPUT ///////////////////////////////////
class RansacOutput : public Algorithm {
public:
    virtual ~RansacOutput() override = default;
    static Ptr<RansacOutput> create(const Mat &model_,
        const std::vector<bool> &inliers_mask_,
        int time_mcs_, double score_, int number_inliers_, int number_iterations_,
        int number_estimated_models_, int number_good_models_);

    /*
     * Return inliers' indices.
     * size of vector = number of inliers
     */
    virtual const std::vector<int > &getInliers() const = 0;
    /*
     * Return inliers mask. Vector of points size. 1-inlier, 0-outlier.
     */
    virtual const std::vector<bool> &getInliersMask() const = 0;

    /*
     * Return inliers' errors. Vector of points size.
     */
    virtual int getTimeMicroSeconds() const = 0;
    virtual int getTimeMicroSeconds1() const = 0;
    virtual int getTimeMilliSeconds2() const = 0;
    virtual int getTimeSeconds3() const = 0;
    virtual int getNumberOfInliers() const = 0;
    virtual int getNumberOfMainIterations() const = 0;
    virtual int getNumberOfGoodModels () const = 0;
    virtual int getNumberOfEstimatedModels () const = 0;
    virtual const Mat &getModel() const = 0;
};

////////////////////////////////////////////// MODEL /////////////////////////////////////////////
/*
* Homography - 4 points
* Fundamental - 7 points
* Essential - 5 points, Stewenius solver
*/
enum EstimationMethod  { Homography, HomographyQR, Fundamental, Fundamental8,
    Essential, Affine, P3P, P6P, Similarity };
enum SamplingMethod  { Uniform, ProgressiveNAPSAC, Napsac, Prosac, Evsac };
enum NeighborSearchMethod {Flann, Grid, RadiusSearch};
enum LocalOptimMethod {NullLO, InLORsc, ItLORsc, ItFLORsc, InItLORsc, InItFLORsc, GC, SIGMA};
enum ScoreMethod {RANSAC, MSAC, LMS, MLESAC, MAGSAC};
enum VerificationMethod { NullVerifier, SprtVerifier, TddVerifier };
enum PolishingMethod { NonePolisher, LSQPolisher, GCPolisher };
enum ErrorMetric {DIST_TO_LINE, SAMPSON_ERR, SGD_ERR, SYMM_REPR_ERR, FORW_REPR_ERR, RERPOJ};

class Model : public Algorithm {
public:
    virtual bool isFundamental () const = 0;
    virtual bool isHomography () const = 0;
    virtual bool isEssential () const = 0;
    virtual bool isPnP () const = 0;

    // getters
    virtual int getSampleSize () const = 0;
    virtual bool resetRandomGenerator () const = 0;
    virtual int getMaxNumHypothesisToTestBeforeRejection() const = 0;
    virtual PolishingMethod getFinalPolisher () const = 0;
    virtual LocalOptimMethod getLO () const = 0;
    virtual const Mat &getDescriptor () const = 0;
    virtual Mat &getRefDescriptor () = 0;

    virtual ErrorMetric getError () const = 0;
    virtual EstimationMethod getEstimator () const = 0;
    virtual ScoreMethod getScore () const = 0;
    virtual int getMaxIters () const = 0;
    virtual double getConfidence () const = 0;
    virtual bool isTimeLimit () const = 0;
    virtual double getThreshold () const = 0;
    virtual VerificationMethod getVerifier () const = 0;
    virtual SamplingMethod getSampler () const = 0;
    virtual int getMaxSampleSizeLO () const = 0;
    virtual double getTimeForModelEstimation () const = 0;
    virtual double getSPRTdelta () const = 0;
    virtual double getSPRTepsilon () const = 0;
    virtual double getSPRTavgNumModels () const = 0;
    virtual NeighborSearchMethod getNeighborsSearch () const = 0;
    virtual int getKNN () const = 0;
    virtual int getCellSize () const = 0;
    virtual bool isSampleLimit () const = 0;
    virtual double getRelaxCoef () const = 0;
    virtual int getMaxTimeMcs() const = 0;

    virtual int getLOThresholdMultiplier() const = 0;
    virtual int getLOIterativeSampleSize() const = 0;
    virtual int getLOIterativeMaxIters() const = 0;
    virtual int getLOInnerMaxIters() const = 0;
    virtual int getMaxSampleSizeLOiterative () const = 0;

    virtual const std::vector<int> &getGridCellNumber () const = 0;
    virtual int getImage1Width () const = 0;
    virtual int getImage1Height () const = 0;
    virtual int getImage2Width () const = 0;
    virtual int getImage2Height () const = 0;

    // setters
    virtual void setLocalOptimization (LocalOptimMethod lo_) = 0;
    virtual void setKNearestNeighhbors (int knn_) = 0;
    virtual void setNeighborsType (NeighborSearchMethod neighbors) = 0;
    virtual void setCellSize (int cell_size_) = 0;
    virtual void setResetRandomGenerator (bool reset) = 0;

    virtual void setVerifier (VerificationMethod verifier_) = 0;
    virtual void setPolisher (PolishingMethod polisher_) = 0;
    virtual void setError (ErrorMetric error_) = 0;

    // 0 - no trace, 1 - inlier mask, 2 - full trace (e.g., time, num iters etc)
    virtual void setTrace (int trace) = 0;
    virtual int getTrace () const = 0;

    virtual void setDescriptor(const Mat &desc) = 0;
    virtual void setSPRT (double sprt_eps_ = 0.005, double sprt_delta_ = 0.0025,
                          double avg_num_models_ = 1, double time_for_model_est_ = 5e2) = 0;
    virtual void setImageSize (int img1_width_, int img1_height_,
                               int img2_width_, int img2_height_) = 0;

    static Ptr<Model> create(double threshold_, EstimationMethod estimator_, SamplingMethod sampler_,
         double confidence_=0.95, int max_iterations_=5000, ScoreMethod score_ =ScoreMethod::RANSAC);
};

Mat findHomography(InputArray srcPoints, InputArray dstPoints, int method = 0,
                   double ransacReprojThreshold = 3, OutputArray mask = noArray(),
                   const int maxIters = 2000, const double confidence = 0.995);

}}

#endif //OPENCV_USAC_USAC_HPP

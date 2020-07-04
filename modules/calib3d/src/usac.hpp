// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_USAC_USAC_HPP
#define OPENCV_USAC_USAC_HPP

namespace cv { namespace usac {
// Abstract Error class
class Error : public Algorithm {
public:
    // set model to use getError() function
    virtual void setModelParameters (const Mat &model) = 0;
    // returns error of point wih @point_idx w.r.t. model
    virtual float getError (int point_idx) const = 0;
};

// Symmetric Reprojection Error for Homography
class ReprojectionErrorSymmetric : public Error {
public:
    static Ptr<ReprojectionErrorSymmetric> create(const Mat &points);
};

// Forward Reprojection Error for Homography
class ReprojectionErrorForward : public Error {
public:
    static Ptr<ReprojectionErrorForward> create(const Mat &points);
};

// Normalizing transformation of data points
class NormTransform : public Algorithm {
public:
    /*
     * @norm_points is output matrix of size pts_size x 4
     * @sample constains indices of points
     * @sample_number is number of used points in sample <0; sample_number)
     * @T1, T2 are output transformation matrices
     */
    virtual void getNormTransformation (Mat &norm_points, const std::vector<int> &sample,
                                        int sample_number, Mat &T1, Mat &T2) const = 0;
    static Ptr<NormTransform> create (const Mat &points);
};

/////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////// SOLVER ///////////////////////////////////////////
class MinimalSolver : public Algorithm {
public:
    // Estimate models from minimal sample. models.size() == number of found solutions
    virtual int estimate (const std::vector<int> &sample, std::vector<Mat> &models) const = 0;
    // return minimal sample size required for estimation.
    virtual int getSampleSize() const = 0;
    // return maximum number of possible solutions.
    virtual int getMaxNumberOfSolutions () const = 0;
};

//-------------------------- HOMOGRAPHY MATRIX -----------------------
class HomographyMinimalSolver4ptsGEM : public MinimalSolver {
public:
    static Ptr<HomographyMinimalSolver4ptsGEM> create(const Mat &points_);
};

//////////////////////////////////////// NON MINIMAL SOLVER ///////////////////////////////////////
class NonMinimalSolver : public Algorithm {
public:
    // Estimate models from non minimal sample. models.size() == number of found solutions
    virtual int estimate (const std::vector<int> &sample, int sample_size,
          std::vector<Mat> &models, const std::vector<double> &weights) const = 0;
    // return minimal sample size required for non-minimal estimation.
    virtual int getMinimumRequiredSampleSize() const = 0;
    // return maximum number of possible solutions.
    virtual int getMaxNumberOfSolutions () const = 0;
};

//-------------------------- HOMOGRAPHY MATRIX -----------------------
class HomographyNonMinimalSolver : public NonMinimalSolver {
public:
    static Ptr<HomographyNonMinimalSolver> create(const Mat &points_);
};

//////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////// SCORE ///////////////////////////////////////////
class Score {
public:
    int inlier_number;
    double score;
    Score () { // set worst case
        inlier_number = 0;
        score = std::numeric_limits<double>::max();
    }
    Score (int inlier_number_, double score_) { // copy constructor
        inlier_number = inlier_number_;
        score = score_;
    }
    // Compare two scores. Objective is minimization of score. Lower score is better.
    inline bool better(const Score &score2) const {
        return score < score2.score;
    }
};

////////////////////////////////////////// QUALITY ///////////////////////////////////////////
class Quality : public Algorithm {
public:
    virtual ~Quality() override = default;
    /*
     * Calculates number of inliers and score of the @model.
     * return Score with calculated inlier_number and score.
     * @model: Mat current model, e.g., H matrix.
     * @get_inliers: if true inliers will be stored to @inliers.
     * @inliers: vector containing inlier indices, which are stored from the beginning of array.
     * Note, @inliers must be of size of number of points.
     */
    virtual Score getScore (const Mat &model, double threshold, bool get_inliers,
                            std::vector<int> &inliers) const = 0;
    virtual Score getScore (const Mat &model,bool get_inliers,std::vector<int> &inliers) const = 0;
    virtual Score getScore (const Mat &model) const = 0;
    // return true if point with given @point_idx is inliers, false-otherwise
    virtual bool isInlier (int point_idx) const = 0;
    // set @model for function isInlier()
    virtual void setModel (const Mat &model) const = 0;
    // get @inliers of the @model. Assume threshold is given
    virtual int getInliers (const Mat &model, std::vector<int> &inliers) const = 0;
    // get @inliers of the @model for given threshold
    virtual int getInliers (const Mat &model, std::vector<int> &inliers, double thr) const = 0;
    // Set the best score, so evaluation of the model can terminate earlier
    virtual void setBestScore (double best_score_) = 0;
    // set @inliers_mask: true if point i is inlier, false - otherwise.
    virtual int getInliers (const Mat &model, std::vector<bool> &inliers_mask) const = 0;
    virtual Ptr<Quality> clone () const = 0;
};

// RANSAC (binary) quality
class RansacQuality : public Quality {
public:
    static Ptr<RansacQuality> create(int points_size_, double threshold_,const Ptr<Error> &error_);
};

// M-estimator quality - truncated Squared error
class MsacQuality : public Quality {
public:
    static Ptr<MsacQuality> create(int points_size_, double threshold_, const Ptr<Error> &error_);
};

//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////// DEGENERACY //////////////////////////////////
class Degeneracy : public Algorithm {
public:
    virtual ~Degeneracy() override = default;
    /*
     * Check if sample causes degenerate configurations.
     * For example, test if points are collinear.
     */
    virtual bool isSampleGood (const std::vector<int> &/*sample*/) const {
        return true;
    }
    /*
     * Check if model satisfies constraints.
     * For example, test if epipolar geometry satisfies oriented constraint.
     */
    virtual bool isModelValid (const Mat &/*model*/, const std::vector<int> &/*sample*/) const {
        return true;
    }
    /*
     * Fix degenerate model.
     * Return true if model is degenerate, false - otherwise
     */
    virtual bool recoverIfDegenerate (const std::vector<int> &/*sample*/, const Mat &/*best_model*/,
                          Mat &/*non_degenerate_model*/, Score &/*non_degenerate_model_score*/) {
        return false;
    }
    virtual Ptr<Degeneracy> clone() const = 0;
};

class HomographyDegeneracy : public Degeneracy {
public:
    static Ptr<HomographyDegeneracy> create(const Mat &points_);
};

/////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////// ESTIMATOR //////////////////////////////////
class Estimator : public Algorithm{
public:
    /*
     * Estimate models with minimal solver.
     * Return number of valid solutions after estimation.
     * Return models accordingly to number of solutions.
     * Note, vector of models must allocated before.
     * Note, not all degenerate tests are included in estimation.
     */
    virtual int
    estimateModels (const std::vector<int> &sample, std::vector<Mat> &models) const = 0;
    /*
     * Estimate model with non-minimal solver.
     * Return number of valid solutions after estimation.
     * Note, not all degenerate tests are included in estimation.
     */
    virtual int
    estimateModelNonMinimalSample (const std::vector<int> &sample, int sample_size,
                       std::vector<Mat> &models, const std::vector<double> &weights) const = 0;
    // return minimal sample size required for minimal estimation.
    virtual int getMinimalSampleSize () const = 0;
    // return minimal sample size required for non-minimal estimation.
    virtual int getNonMinimalSampleSize () const = 0;
    // return maximum number of possible solutions of minimal estimation.
    virtual int getMaxNumSolutions () const = 0;
    // return maximum number of possible solutions of non-minimal estimation.
    virtual int getMaxNumSolutionsNonMinimal () const = 0;
    virtual Ptr<Estimator> clone() const = 0;
};

class HomographyEstimator : public Estimator {
public:
    static Ptr<HomographyEstimator> create (const Ptr<MinimalSolver> &min_solver_,
            const Ptr<NonMinimalSolver> &non_min_solver_, const Ptr<Degeneracy> &degeneracy_);
};

//////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////// MODEL VERIFIER ////////////////////////////////////
class ModelVerifier : public Algorithm {
public:
    virtual ~ModelVerifier() override = default;
    // Return true if model is good, false - otherwise.
    virtual bool isModelGood(const Mat &model) = 0;
    // Return true if score was computed during evaluation.
    virtual bool getScore(Score &score) const = 0;
    virtual Ptr<ModelVerifier> clone () const = 0;
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

///////////////////////////////// SPRT VERIFIER /////////////////////////////////////////
/*
* Matas, Jiri, and Ondrej Chum. "Randomized RANSAC with sequential probability ratio test."
* Tenth IEEE International Conference on Computer Vision (ICCV'05) Volume 1. Vol. 2. IEEE, 2005.
*/
class SPRT : public ModelVerifier {
public:
    // return constant reference of vector of SPRT histories for SPRT termination.
    virtual const std::vector<SPRT_history> &getSPRTvector () const = 0;

    static Ptr<SPRT> create (RNG &rng, const Ptr<Error> &err_, int points_size_,
       double inlier_threshold_, double prob_pt_of_good_model,
       double prob_pt_of_bad_model, double time_sample, double avg_num_models, int score_type_);
};

/////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////// TERMINATION ///////////////////////////////////////////
class TerminationCriteria : public Algorithm {
public:
    // return true if RANSAC can terminate by given @current_iteration number
    virtual bool terminate(int current_iteration) const = 0;
    // termination starts measure time. Should be called in the beginning of RANSAC.
    virtual void startMeasureTime() = 0;
    // update termination object by given @model and @inlier number.
    virtual void update(const Mat &model, int inlier_number) = 0;
    // return predicted number of iterations required for RANSAC
    virtual int getPredictedNumberIterations() const = 0;
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
namespace Utils {
    // randomly shuffle vector.
    void random_shuffle (RNG &rng, std::vector<int> &array);
}
namespace Math {
    /*
     * @points Nx4 array: x1 y1 x2 y2
     * @sample Mx1 array
     * return true if any 3 out of M points in sample are collinear
     */
    bool haveCollinearPoints(const Mat &points, const std::vector<int> &sample,
            double threshold=1);
    // return skew symmetric matrix
    Mat getSkewSymmetric(const Mat &v_);
    // do cross product between two vectors
    Mat cross(const Mat &a_, const Mat &b_);
    // compute rank of 3x3 matrix
    int rank3x3 (const Mat &A_);
    // eliminate matrix with m rows and n columns to be upper triangular.
    void eliminateUpperTriangluar (std::vector<double> &a, int m, int n);
}

///////////////////////////////////////// RANDOM GENERATOR /////////////////////////////////////
class RandomGenerator : public Algorithm {
public:
    virtual ~RandomGenerator() override = default;
    // interval is <0, max_range);
    virtual void resetGenerator (int max_range) = 0;
    // return sample filled with random numbers
    virtual void generateUniqueRandomSet (std::vector<int> &sample) = 0;
    // fill @sample of size @subset_size with random numbers in range <0, @max_range)
    virtual void generateUniqueRandomSet (std::vector<int> &sample, int subset_size,
                                                                    int max_range) = 0;
    // fill @sample of size @sample.size() with random numbers in range <0, @max_range)
    virtual void generateUniqueRandomSet (std::vector<int> &sample, int max_range) = 0;
    // return subset=sample size
    virtual void setSubsetSize (int subset_sz) = 0;
    // return random number
    virtual int getRandomNumber () = 0;
};

class UniformRandomGenerator : public RandomGenerator {
public:
    static Ptr<UniformRandomGenerator> create (RNG &rng);
    static Ptr<UniformRandomGenerator> create (RNG &rng, int max_range, int subset_size_);
};

//////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////// SAMPLER ///////////////////////////////////////
class Sampler : public Algorithm {
public:
    virtual ~Sampler() override = default;
    // set new points size
    virtual void setNewPointsSize (int points_size) = 0;
    // generate sample. Fill @sample with indices of points.
    virtual void generateSample (std::vector<int> &sample) = 0;
    // generate sample for given points size
    virtual void generateSample (std::vector<int> &sample, int points_size) = 0;
    // return sample size
    virtual int getSampleSize () const = 0;
};

////////////////////////////////////// UNIFORM SAMPLER ////////////////////////////////////////////
class UniformSampler : public Sampler {
public:
    static Ptr<UniformSampler> create(RNG &rng, int sample_size_, int points_size_);
};

///////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////// LOCAL OPTIMIZATION /////////////////////////////////////////
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
    virtual Ptr<LocalOptimization> clone() const = 0;
};

//////////////////////////////////// INNER LO ///////////////////////////////////////
class InnerLocalOptimization : public LocalOptimization {
public:
    static Ptr<InnerLocalOptimization>
    create(const Ptr<Estimator> &estimator_, const Ptr<Quality> &quality_,
           const Ptr<Sampler> &lo_sampler_, int points_size, int lo_inner_iterations_=15);
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
     * @new_score: score of output model.
     * Return true if polishing was successful, false - otherwise.
     */
    virtual bool polishSoFarTheBestModel (const Mat &model, const Score &best_model_score,
            Mat &new_model, Score &new_model_score) = 0;
};

///////////////////////////////////// LEAST SQUARES POLISHER //////////////////////////////////////
class LeastSquaresPolishing : public FinalModelPolisher {
public:
    static Ptr<LeastSquaresPolishing> create (const Ptr<Estimator> &estimator_,
        const Ptr<Quality> &quality_, int points_size, int lsq_iterations_=2);
};

/////////////////////////////////// RANSAC OUTPUT ///////////////////////////////////
class RansacOutput : public Algorithm {
public:
    virtual ~RansacOutput() override = default;
    static Ptr<RansacOutput> create(const Mat &model_,
        const std::vector<bool> &inliers_mask_,
        int time_mcs_, double score_, int number_inliers_, int number_iterations_,
        int number_estimated_models_, int number_good_models_);

    // Return inliers' indices. size of vector = number of inliers
    virtual const std::vector<int > &getInliers() = 0;
    // Return inliers mask. Vector of points size. 1-inlier, 0-outlier.
    virtual const std::vector<bool> &getInliersMask() const = 0;
    virtual int getTimeMicroSeconds() const = 0;
    virtual int getTimeMicroSeconds1() const = 0;
    virtual int getTimeMilliSeconds2() const = 0;
    virtual int getTimeSeconds3() const = 0;
    virtual int getNumberOfInliers() const = 0;
    virtual int getNumberOfMainIterations() const = 0;
    virtual int getNumberOfGoodModels () const = 0;
    virtual int getNumberOfEstimatedModels () const = 0;
    virtual const Mat &getModel() const = 0;
    virtual Ptr<RansacOutput> clone() const = 0;
};

////////////////////////////////////////////// MODEL /////////////////////////////////////////////
/*
* Homography - 4 points
* Fundamental - 7 points
* Essential - 5 points, Stewenius solver
*/
enum EstimationMethod  { Homography, Fundamental, Fundamental8,
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

    virtual int getSamplerLengthPNAPSAC () const = 0;
    virtual int getFinalLSQIterations () const = 0;
    virtual int getDegreesOfFreedom () const = 0;
    virtual double getSigmaQuantile () const = 0;
    virtual double getUpperIncompleteOfSigmaQuantile () const = 0;
    virtual double getLowerIncompleteOfSigmaQuantile () const = 0;
    virtual double getC () const = 0;
    virtual double getMaximumThreshold () const = 0;
    virtual double getGraphCutSpatialCoherenceTerm () const = 0;
    virtual int getLOSampleSize () const = 0;
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

    virtual void setSPRT (double sprt_eps_ = 0.005, double sprt_delta_ = 0.0025,
                          double avg_num_models_ = 1, double time_for_model_est_ = 5e2) = 0;
    virtual void setImageSize (int img1_width_, int img1_height_,
                               int img2_width_, int img2_height_) = 0;

    static Ptr<Model> create(double threshold_, EstimationMethod estimator_, SamplingMethod sampler_,
         double confidence_=0.95, int max_iterations_=5000, ScoreMethod score_ =ScoreMethod::MSAC);
};

int mergePoints (const Mat &pts1, const Mat &pts2, Mat &pts);

Mat findHomography(InputArray srcPoints, InputArray dstPoints, int method = 0,
                   double ransacReprojThreshold = 3, OutputArray mask = noArray(),
                   const int maxIters = 2000, const double confidence = 0.995);
}}

#endif //OPENCV_USAC_USAC_HPP

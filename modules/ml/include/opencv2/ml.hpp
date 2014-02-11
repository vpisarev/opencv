/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef __OPENCV_ML_HPP__
#define __OPENCV_ML_HPP__

#ifdef __cplusplus
#  include "opencv2/core.hpp"
#endif

#ifdef __cplusplus

#include <map>
#include <iostream>

// Apple defines a check() macro somewhere in the debug headers
// that interferes with a method definiton in this header
#undef check

namespace cv
{

namespace ml
{

/* Variable type */
enum
{
    VAR_NUMERICAL    =0,
    VAR_ORDERED      =0,
    VAR_CATEGORICAL  =1
};

enum
{
    TEST_ERROR = 0,
    TRAIN_ERROR = 1
};

class CV_EXPORTS_W_MAP ParamGrid
{
public:
    ParamGrid();
    ParamGrid(double _minVal, double _maxVal, double _logStep);

    CV_PROP_RW double minVal;
    CV_PROP_RW double maxVal;
    CV_PROP_RW double logStep;
};


class CV_EXPORTS TrainData
{
public:
    class CV_EXPORTS Params
    {
    public:
        Params();
        Params(int headerLineCount, int responseIdx, char delimiter, char missch, const String& varType);

        char delimiter;
        char missch;

        int headerLines;
        int responseIdx;
        String varTypeSpec;
    };

    virtual ~TrainData();

    virtual bool getTFlag() const = 0;
    virtual Mat getSamples() const = 0;
    virtual Mat getResponses() = 0;
    virtual Mat getMissing() const = 0;
    virtual Mat getVarIdx() const = 0;
    virtual Mat getVarType() const = 0;
    virtual Mat getTrainSampleIdx() const = 0;
    virtual Mat getTestSampleIdx() const = 0;

    virtual Mat getNormCatResponses() const = 0;
    virtual Mat getClassLabels() const = 0;
    virtual Mat getClassCounters() const = 0;
    
    virtual Params getParams() const = 0;

    virtual void setTrainTestSplit(int count, bool shuffle=true) = 0;
    virtual void setTrainTestSplitRatio(float ratio, bool shuffle=true) = 0;
};

CV_EXPORTS Ptr<TrainData> loadDataFromCSV(const String& filename, const TrainData::Params& params);
CV_EXPORTS Ptr<TrainData> createTrainData(InputArray samples, bool tflag, InputArray responses,
                                          InputArray varIdx, InputArray sampleIdx,
                                          InputArray varType, InputArray missing);


class CV_EXPORTS_W StatModel : public Algorithm
{
public:
    virtual ~StatModel();
    virtual void clear();

    virtual int getVarCount() const;
    virtual int getSampleCount() const;

    virtual bool isTrained() const;
    virtual bool isRegression() const;

    virtual bool train( InputArray trainData, bool tflag, InputArray responses,
                        InputArray varIdx, InputArray sampleIdx, InputArray varType,
                        InputArray missing, bool update=false );
    virtual float calcError( const Ptr<TrainData>& data, bool test, OutputArray resp ) const;

    virtual String defaultModelName() const;
};

/****************************************************************************************\
*                                 Normal Bayes Classifier                                *
\****************************************************************************************/

/* The structure, representing the grid range of statmodel parameters.
   It is used for optimizing statmodel accuracy by varying model parameters,
   the accuracy estimate being computed by cross-validation.
   The grid is logarithmic, so <step> must be greater then 1. */

class CV_EXPORTS_W NormalBayesClassifier : public StatModel
{
public:
    virtual ~NormalBayesClassifier();
    virtual float predict( InputArray inputs, OutputArray outputs, bool rawOutput=false ) const;
    virtual float predictProb( InputArray inputs, OutputArray outputs, OutputArray outputProbs ) const;
};

CV_EXPORTS_W Ptr<NormalBayesClassifier> createNormalBayesClassifier(InputArray trainData, InputArray responses,
                                                                    InputArray varIdx, InputArray sampleIdx);

/****************************************************************************************\
*                          K-Nearest Neighbour Classifier                                *
\****************************************************************************************/

// k Nearest Neighbors
class CV_EXPORTS_W KNearest : public StatModel
{
public:
    virtual float findNearest( InputArray samples, int k, OutputArray results,
                               OutputArray neighbors, OutputArray neighborResponses,
                               OutputArray dist ) const;
    int getMaxK() const;
    virtual bool isRegression() const;
};

CV_EXPORTS_W Ptr<KNearest> createKNearest(InputArray trainData, InputArray responses,
                                          InputArray sampleIdx,
                                          bool isRegression=false, int maxK=32);

/****************************************************************************************\
*                                   Support Vector Machines                              *
\****************************************************************************************/

// SVM model
class CV_EXPORTS_W SVM : public StatModel
{
public:
    class CV_EXPORTS_W_MAP Params
    {
    public:
        Params();
        Params( int svm_type, int kernel_type,
                double degree, double gamma, double coef0,
                double Cvalue, double nu, double p,
                const Mat& classWeights, TermCriteria termCrit );

        CV_PROP_RW int         svmType;
        CV_PROP_RW int         kernelType;
        CV_PROP_RW double      gamma, coef0, degree;

        CV_PROP_RW double      C;  // for CV_SVM_C_SVC, CV_SVM_EPS_SVR and CV_SVM_NU_SVR
        CV_PROP_RW double      nu; // for CV_SVM_NU_SVC, CV_SVM_ONE_CLASS, and CV_SVM_NU_SVR
        CV_PROP_RW double      p; // for CV_SVM_EPS_SVR
        CV_PROP_RW Mat         classWeights; // for CV_SVM_C_SVC
        CV_PROP_RW TermCriteria termCrit; // termination criteria
    };

    class CV_EXPORTS Kernel : public Algorithm
    {
    public:
        virtual ~Kernel();
        virtual int getType() const;
        virtual void calc( int vcount, int n, const float** vecs, const float* another, float* results ) = 0;
    };

    // SVM type
    enum { C_SVC=100, NU_SVC=101, ONE_CLASS=102, EPS_SVR=103, NU_SVR=104 };

    // SVM kernel type
    enum { LINEAR=0, POLY=1, RBF=2, SIGMOID=3, CHI2=4, INTER=5 };

    // SVM params type
    enum { C=0, GAMMA=1, P=2, NU=3, COEF=4, DEGREE=5 };

    virtual ~SVM();

    CV_WRAP virtual float predict( InputArray samples, OutputArray results, bool returnDFVal=false ) const;

    CV_WRAP virtual Mat getSupportVectors() const;
    virtual Params getParams() const;
    virtual void getDecisionFunction(int i, OutputArray alpha, OutputArray svidx) const;

    static ParamGrid getDefaultGrid( int param_id );
};

CV_EXPORTS Ptr<SVM::Kernel> createStandardSVMKernel(const SVM::Params& params);

CV_EXPORTS_W Ptr<SVM> createSVM( InputArray trainData, InputArray responses,
                                 InputArray varIdx, InputArray sampleIdx,
                                 const SVM::Params& params,
                                 const Ptr<SVM::Kernel>& kernel=Ptr<SVM::Kernel>() );

CV_EXPORTS_W Ptr<SVM> createSVMAuto( InputArray trainData, InputArray responses,
                                     InputArray varIdx, InputArray sampleIdx,
                                     const SVM::Params& params, int kFold = 10,
                                     ParamGrid Cgrid = SVM::getDefaultGrid(SVM::C),
                                     ParamGrid gammaGrid  = SVM::getDefaultGrid(SVM::GAMMA),
                                     ParamGrid pGrid      = SVM::getDefaultGrid(SVM::P),
                                     ParamGrid nuGrid     = SVM::getDefaultGrid(SVM::NU),
                                     ParamGrid coeffGrid  = SVM::getDefaultGrid(SVM::COEF),
                                     ParamGrid degreeGrid = SVM::getDefaultGrid(SVM::DEGREE),
                                     bool balanced=false);

/****************************************************************************************\
*                              Expectation - Maximization                                *
\****************************************************************************************/
class CV_EXPORTS_W EM : public Algorithm
{
public:
    // Type of covariation matrices
    enum {COV_MAT_SPHERICAL=0, COV_MAT_DIAGONAL=1, COV_MAT_GENERIC=2, COV_MAT_DEFAULT=COV_MAT_DIAGONAL};

    // Default parameters
    enum {DEFAULT_NCLUSTERS=5, DEFAULT_MAX_ITERS=100};

    // The initial step
    enum {START_E_STEP=1, START_M_STEP=2, START_AUTO_STEP=0};

    class CV_EXPORTS_W_MAP Params
    {
    public:
        explicit Params(int nclusters=DEFAULT_NCLUSTERS, int covMatType=EM::COV_MAT_DIAGONAL,
                        const TermCriteria& termCrit=TermCriteria(TermCriteria::COUNT+TermCriteria::EPS,
                                                                  EM::DEFAULT_MAX_ITERS, 1e-6));
        int nclusters;
        int covMatType;
        TermCriteria termCrit;
    };

    virtual Mat getWeights() const;
    virtual Mat getMeans() const;
    virtual void getCovs(std::vector<Mat>& covs) const;

    CV_WRAP virtual Vec2d predict(InputArray sample, OutputArray probs=noArray()) const;
};

CV_EXPORTS_W Ptr<EM> createEM(InputArray samples,
                              OutputArray logLikelihoods=noArray(),
                              OutputArray labels=noArray(),
                              OutputArray probs=noArray(),
                              const EM::Params& params=EM::Params());

CV_EXPORTS_W Ptr<EM> createEM_startWithE(InputArray samples, InputArray means0,
                                         InputArray covs0=noArray(),
                                         InputArray weights0=noArray(),
                                         OutputArray logLikelihoods=noArray(),
                                         OutputArray labels=noArray(),
                                         OutputArray probs=noArray(),
                                         const EM::Params& params=EM::Params());

CV_EXPORTS_W Ptr<EM> createEM_startWithM(InputArray samples, InputArray probs0,
                                         OutputArray logLikelihoods=noArray(),
                                         OutputArray labels=noArray(),
                                         OutputArray probs=noArray(),
                                         const EM::Params& params=EM::Params());

/****************************************************************************************\
*                                      Decision Tree                                     *
\****************************************************************************************/

class CV_EXPORTS_W DTree : public StatModel
{
public:
    class CV_EXPORTS_W_MAP Params
    {
    public:
        Params();
        Params( int maxDepth, int minSampleCount,
               float regressionAccuracy, bool useSurrogates,
               int maxCategories, int CVFolds,
               bool use1SERule, bool truncatePrunedTree,
               const Mat& priors );

        CV_PROP_RW int   maxCategories;
        CV_PROP_RW int   maxDepth;
        CV_PROP_RW int   minSampleCount;
        CV_PROP_RW int   CVFolds;
        CV_PROP_RW bool  useSurrogates;
        CV_PROP_RW bool  use1SERule;
        CV_PROP_RW bool  truncatePrunedTree;
        CV_PROP_RW float regressionAccuracy;
        CV_PROP_RW Mat priors;
    };

    class CV_EXPORTS_W Data
    {
    public:
        virtual ~Data();
        virtual void getVectors( const Mat& subsampleIdx,
                                 float* values, uchar* missing,
                                 float* responses, bool getClassIdx=false );
        virtual int getNumClasses() const;
        virtual int getVarType(int vi) const;
        virtual int getWorkVarCount() const;
    };

    class CV_EXPORTS Node
    {
    public:
        int classIdx;
        double value;

        int parent;
        int left;
        int right;

        int split;
    };

    class CV_EXPORTS Split
    {
    public:
        int varIdx;
        int condensedIdx;
        int inversed;
        float quality;
        int next;
        float c;
    };

    virtual ~DTree();

    virtual float predict( InputArray samples, InputArray missingDataMask,
                           OutputArray results, bool preprocessedInput=false,
                           bool rawOutput=false ) const;

    CV_WRAP virtual Mat getVarImportance();

    virtual int getRoot() const;
    virtual int getPrunedTreeIdx() const;
    virtual Ptr<Data> getTrainData() const;
    virtual const std::vector<Node>& getNodes();
    virtual const std::vector<Split>& getSplits();
    virtual const std::vector<int>& getSubsets();
    virtual int subsetSize() const;
};

CV_EXPORTS_W Ptr<DTree::Data> createDTreeTrainData(InputArray trainData, bool tflag,
                                                   InputArray responses, InputArray varIdx,
                                                   InputArray sampleIdx, InputArray varType,
                                                   InputArray missingDataMask,
                                                   const DTree::Params& params,
                                                   bool shared=false, bool addLabels=false);

CV_EXPORTS_W Ptr<DTree> createDTree(InputArray samples, bool tflag, InputArray responses,
                                    InputArray varIdx, InputArray sampleIdx,
                                    InputArray missingDataMask,
                                    const DTree::Params& params);

CV_EXPORTS_W Ptr<DTree> createDTree(const Ptr<DTree::Data>& trainData,
                                    InputArray subsampleIdx);

/****************************************************************************************\
*                                   Random Trees Classifier                              *
\****************************************************************************************/

class CV_EXPORTS_W RTrees : public StatModel
{
public:
    class CV_EXPORTS_W_MAP Params : public DTree::Params
    {
    public:
        Params();
        Params( int maxDepth, int minSampleCount,
                float regressionAccuracy, bool useSurrogates,
                int maxCategories, const float* priors,
                bool calcVarImportance, int nactiveVars, int maxNTrees,
                float forestAccuracy, int termcritType );

        CV_PROP_RW bool calcVarImportance; // true <=> RF processes variable importance
        CV_PROP_RW int nactiveVars;
        CV_PROP_RW TermCriteria termCrit;
    };

    virtual ~RTrees();
    CV_WRAP virtual Mat getVarImportance();
    CV_WRAP virtual float getTrainError();
    CV_WRAP Mat getActiveVarMask() const;

    RNG& getRNG();

    const std::vector<Ptr<DTree> >& getTrees() const;
};


CV_EXPORTS_W Ptr<RTrees> createRTrees(InputArray samples, bool tflag, InputArray responses,
                                      InputArray varIdx, InputArray sampleIdx,
                                      InputArray missingDataMask,
                                      const RTrees::Params& params);

/****************************************************************************************\
*                                   Boosted tree classifier                              *
\****************************************************************************************/

class CV_EXPORTS_W Boost : public StatModel
{
public:
    class CV_EXPORTS_W_MAP Params : public DTree::Params
    {
    public:
        CV_PROP_RW int boostType;
        CV_PROP_RW int weakCount;
        CV_PROP_RW int splitCriteria;
        CV_PROP_RW double weightTrimRate;

        Params();
        Params( int boostType, int weakCount, double weightTrimRate,
                int maxDepth, bool useSurrogates, const Mat& priors );
    };

    // Boosting type
    enum { DISCRETE=0, REAL=1, LOGIT=2, GENTLE=3 };

    // Splitting criteria
    enum { DEFAULT=0, GINI=1, MISCLASS=3, SQERR=4 };

    virtual ~Boost();
    virtual float predict( InputArray samples, InputArray missing,
                           OutputArray weakResponse, Range range,
                           bool rawMode=false, bool returnSum=false ) const;

    CV_WRAP virtual void prune( Range range );

    virtual const Mat getActiveVars(bool absoluteIdx=true);

    const std::vector<Ptr<DTree> >& getTrees() const;
    const Params& getParams() const;
    const Ptr<DTree::Data>& getData() const;
};

CV_EXPORTS Ptr<Boost> createBoost(InputArray trainData, bool tflag,
                                  InputArray response, InputArray varIdx,
                                  InputArray sampleIdx, InputArray varType,
                                  InputArray missing,
                                  const Boost::Params& params = Boost::Params());

/****************************************************************************************\
*                                   Gradient Boosted Trees                               *
\****************************************************************************************/

class CV_EXPORTS_W GBTrees : public StatModel
{
public:
    struct CV_EXPORTS_W_MAP Params : public DTree::Params
    {
        CV_PROP_RW int weakCount;
        CV_PROP_RW int lossFunctionType;
        CV_PROP_RW float subsamplePortion;
        CV_PROP_RW float shrinkage;

        Params();
        Params( int lossFunctionType, int weakCount, float shrinkage,
                float subsamplePortion, int maxDepth, bool useSurrogates );
    };

    enum {SQUARED_LOSS=0, ABSOLUTE_LOSS, HUBER_LOSS=3, DEVIANCE_LOSS};
    virtual ~GBTrees();

    virtual float predictSerial( InputArray samples, InputArray missing,
                                 OutputArray weakResponses, int k=-1) const;

    virtual float predict( InputArray samples, InputArray missing,
                           OutputArray weakResponses, int k=-1) const;
};

CV_EXPORTS_W Ptr<GBTrees> createGBTrees(InputArray trainData, int tflag,
                                        InputArray responses, InputArray varIdx,
                                        InputArray sampleIdx, InputArray varType,
                                        InputArray missing,
                                        const GBTrees::Params& params=GBTrees::Params());

/****************************************************************************************\
*                              Artificial Neural Networks (ANN)                          *
\****************************************************************************************/

/////////////////////////////////// Multi-Layer Perceptrons //////////////////////////////

class CV_EXPORTS_W ANN_MLP : public StatModel
{
public:
    struct CV_EXPORTS_W_MAP Params
    {
        Params();
        Params( TermCriteria termCrit, int trainMethod, double param1, double param2=0 );

        enum { BACKPROP=0, RPROP=1 };

        CV_PROP_RW TermCriteria termCrit;
        CV_PROP_RW int trainMethod;

        // backpropagation parameters
        CV_PROP_RW double bpDWScale, bpMomentScale;

        // rprop parameters
        CV_PROP_RW double rpDW0, rpDWPlus, rpDWMinus, rpDWMin, rpDWMax;
    };

    virtual ~ANN_MLP();
    virtual float predict( InputArray inputs, OutputArray outputs ) const;

    // possible activation functions
    enum { IDENTITY = 0, SIGMOID_SYM = 1, GAUSSIAN = 2 };

    // available training flags
    enum { UPDATE_WEIGHTS = 1, NO_INPUT_SCALE = 2, NO_OUTPUT_SCALE = 4 };

    virtual Mat getLayerSizes() const;
    virtual Mat getWeights(int layerIdx);
};

CV_EXPORTS_W Ptr<ANN_MLP> createANN_MLP(InputArray layerSizes,
                                        InputArray inputs, InputArray outputs,
                                        InputArray sampleWeights, InputArray sampleIdx,
                                        ANN_MLP::Params params, int flags,
                                        int activateFunc=ANN_MLP::SIGMOID_SYM,
                                        double fparam1=0, double fparam2=0);

/****************************************************************************************\
*                           Auxilary functions declarations                              *
\****************************************************************************************/

/* Generates <sample> from multivariate normal distribution, where <mean> - is an
   average row vector, <cov> - symmetric covariation matrix */
CV_EXPORTS void randMVNormal( InputArray mean, InputArray cov, int nsamples, OutputArray sample, RNG& rng);

/* Generates sample from gaussian mixture distribution */
CV_EXPORTS void randGaussMixture( InputArray means, InputArray covs, InputArray weights,
                                  int nsamples, OutputArray samples, OutputArray sampClasses );

/* creates test set */
CV_EXPORTS void createConcentricSpheresTestSet( int nsamples, int nfeatures, int nclasses,
                                                OutputArray samples, OutputArray responses);

}
}

#endif // __cplusplus
#endif // __OPENCV_ML_HPP__

/* End of file. */

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
// Copyright (C) 2014, Itseez Inc, all rights reserved.
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

#include "precomp.hpp"

namespace cv { namespace ml {

static inline double
log_ratio( double val )
{
    const double eps = 1e-5;
    val = std::max( val, eps );
    val = std::min( val, 1. - eps );
    return log( val/(1. - val) );
}


Boost::~Boost() {}

Boost::Params::Params()
{
    boostType = Boost::REAL;
    weakCount = 100;
    weightTrimRate = 0.95;
    CVFolds = 0;
    maxDepth = 1;
}


Boost::Params::Params( int _boostType, int _weak_count,
                       double _weightTrimRate, int _maxDepth,
                       bool _use_surrogates, const Mat& _priors )
{
    boostType = _boostType;
    weakCount = _weak_count;
    weightTrimRate = _weightTrimRate;
    CVFolds = 0;
    maxDepth = _maxDepth;
    useSurrogates = _use_surrogates;
    priors = _priors;
}


class DTreesImplForBoost : public DTreesImpl
{
public:
    DTreesImplForBoost() {}
    
    void setBParams(const Boost::Params& p)
    {
        bparams = p;
    }

    Boost::Params getBParams() const
    {
        return bparams;
    }

    void clear()
    {
        DTreesImpl::clear();
    }

    void startTraining( const Ptr<TrainData>& trainData, int flags )
    {
        DTreesImpl::startTraining(trainData, flags);

        if( bparams.boostType != Boost::DISCRETE )
        {
            isClassifier = false;
            int i, n = (int)w->cat_responses.size();
            w->ord_responses.resize(n);

            double a = -1, b = 1;
            if( bparams.boostType == Boost::REAL )
                a = 0;
            else if( bparams.boostType == Boost::LOGIT )
            {
                sumResult.assign(w->sidx.size(), 0.);
                a = -2, b = 2;
            }

            for( i = 0; i < n; i++ )
                w->ord_responses[i] = w->cat_responses[i] > 0 ? b : a;
        }
    }

    void endTraining()
    {
        DTreesImpl::endTraining();
        vector<double> e1, e2;
        std::swap(boostWeights, e1);
        std::swap(sumResult, e2);
    }

    void scaleTree( int root, double scale )
    {
        int nidx = root, pidx = 0;
        Node *node = 0;

        // traverse the tree and save all the nodes in depth-first order
        for(;;)
        {
            for(;;)
            {
                node = &nodes[nidx];
                node->value *= scale;
                if( node->left < 0 )
                    break;
                nidx = node->left;
            }

            for( pidx = node->parent; pidx >= 0 && nodes[pidx].right == nidx;
                 nidx = pidx, pidx = nodes[pidx].parent )
                ;
            
            if( pidx < 0 )
                break;
            
            nidx = nodes[pidx].right;
        }
    }

    void calcValue( int nidx, const vector<int>& _sidx )
    {
        DTreesImpl::calcValue(nidx, _sidx);
        if( bparams.boostType == Boost::REAL )
        {
            WNode* node = &w->wnodes[nidx];
            double p = node->value;
            node->value = 0.5*log_ratio(p);
        }
    }

    bool train( const Ptr<TrainData>& trainData, int flags )
    {
        Params dp(bparams.maxDepth, bparams.minSampleCount, bparams.regressionAccuracy,
                  bparams.useSurrogates, bparams.maxCategories, 0,
                  false, false, bparams.priors);
        setParams(dp);
        startTraining(trainData, flags);
        int treeidx, ntrees = bparams.weakCount >= 0 ? bparams.weakCount : 10000;
        vector<int> sidx = w->sidx;

        for( treeidx = 0; treeidx < ntrees; treeidx++ )
        {
            int root = addTree( sidx );
            if( root < 0 )
                return false;
            updateWeightsAndTrim( treeidx, sidx );
        }
        endTraining();
        return true;
    }

    void updateWeightsAndTrim( int treeidx, vector<int>& sidx )
    {
        int i, j, n = (int)w->sidx.size();
        int nallvars = w->data->getNAllVars();
        double sumw = 0.;
        cv::AutoBuffer<double> buf(n*3);
        double* result = buf;
        float* psamples = (float*)w->samples.ptr<float>();
        size_t sstep0 = w->samples.step/sizeof(psamples[0]), sstep1 = 1;
        int predictFlags = bparams.boostType == Boost::DISCRETE ? (PREDICT_MAX_VOTE | RAW_OUTPUT) : PREDICT_SUM;

        if( w->data->getLayout() == ROW_SAMPLE )
            std::swap(sstep0, sstep1);

        for( i = 0; i < n; i++ )
        {
            Mat sample(nallvars, 1, CV_32F, psamples + i*sstep0, sstep1*sizeof(psamples[0]));
            result[i] = predictTrees(Range(treeidx, treeidx+1), sample, predictFlags);
        }

        // now update weights and other parameters for each type of boosting
        if( bparams.boostType == Boost::DISCRETE )
        {
            // Discrete AdaBoost:
            //   weak_eval[i] (=f(x_i)) is in {-1,1}
            //   err = sum(w_i*(f(x_i) != y_i))/sum(w_i)
            //   C = log((1-err)/err)
            //   w_i *= exp(C*(f(x_i) != y_i))
            double err = 0.;

            for( i = 0; i < n; i++ )
            {
                double wval = boostWeights[i];
                sumw += wval;
                err += wval*(result[i] != w->cat_responses[w->sidx[i]]);
            }

            if( sumw != 0 )
                err /= sumw;
            double C = -log_ratio( err );
            double scale = std::exp(C);

            sumw = 0;
            for( i = 0; i < n; i++ )
            {
                double wval = boostWeights[i];
                if( result[i] != w->cat_responses[w->sidx[i]] )
                    wval *= scale;
                sumw += wval;
                boostWeights[i] = wval;
            }

            scaleTree(roots[treeidx], C);
        }
        else if( bparams.boostType == Boost::REAL || bparams.boostType == Boost::GENTLE )
        {
            // Real AdaBoost:
            //   weak_eval[i] = f(x_i) = 0.5*log(p(x_i)/(1-p(x_i))), p(x_i)=P(y=1|x_i)
            //   w_i *= exp(-y_i*f(x_i))

            // Gentle AdaBoost:
            //   weak_eval[i] = f(x_i) in [-1,1]
            //   w_i *= exp(-y_i*f(x_i))
            for( i = 0; i < n; i++ )
            {
                double wval = boostWeights[i]*std::exp(-result[i]*w->ord_responses[w->sidx[i]]);
                sumw += wval;
                boostWeights[i] = wval;
            }
        }
        else if( bparams.boostType == Boost::LOGIT )
        {
            // LogitBoost:
            //   weak_eval[i] = f(x_i) in [-z_max,z_max]
            //   sum_response = F(x_i).
            //   F(x_i) += 0.5*f(x_i)
            //   p(x_i) = exp(F(x_i))/(exp(F(x_i)) + exp(-F(x_i))=1/(1+exp(-2*F(x_i)))
            //   reuse weak_eval: weak_eval[i] <- p(x_i)
            //   w_i = p(x_i)*1(1 - p(x_i))
            //   z_i = ((y_i+1)/2 - p(x_i))/(p(x_i)*(1 - p(x_i)))
            //   store z_i to the data->data_root as the new target responses
            const double lb_weight_thresh = FLT_EPSILON;
            const double lb_z_max = 10.;

            for( i = 0; i < n; i++ )
            {
                sumResult[i] += 0.5*result[i];
                double p = 1./(1 + std::exp(-2*sumResult[i]));
                double wval = std::max( p*(1 - p), lb_weight_thresh ), z;
                boostWeights[i] = wval;
                sumw += wval;
                j = w->sidx[i];
                if( w->ord_responses[j] > 0 )
                {
                    z = 1./p;
                    w->ord_responses[j] = std::min(z, lb_z_max);
                }
                else
                {
                    z = 1./(1-p);
                    w->ord_responses[j] = -std::min(z, lb_z_max);
                }
            }
        }
        else
            CV_Error(CV_StsNotImplemented, "Unknown boosting type");
        
        // renormalize weights
        if( sumw > FLT_EPSILON )
        {
            sumw = 1./sumw;
            for( i = 0; i < n; ++i )
                boostWeights[i] *= sumw;
        }

        if( bparams.weightTrimRate <= 0. || bparams.weightTrimRate >= 1. )
            return;

        std::copy(boostWeights.begin(), boostWeights.end(), result);
        std::sort(result, result + n);

        // as weight trimming occurs immediately after updating the weights,
        // where they are renormalized, we assume that the weight sum = 1.
        sumw = 1. - bparams.weightTrimRate;

        for( i = 0; i < n; i++ )
        {
            double wval = result[i];
            if( sumw <= 0 )
                break;
            sumw -= wval;
        }

        double threshold = i < n ? result[i] : DBL_MAX;
        sidx.clear();

        for( i = 0; i < n; i++ )
        {
            if( boostWeights[i] >= threshold )
                sidx.push_back(w->sidx[i]);
        }
    }

    float predictTrees( const Range& range, const Mat& sample, int flags0 ) const
    {
        int flags = (flags0 & ~PREDICT_MASK) | PREDICT_SUM;
        float val = DTreesImpl::predictTrees(range, sample, flags);
        if( flags != flags0 )
        {
            int ival = (int)(val > 0);
            if( !(flags0 & RAW_OUTPUT) )
                ival = classLabels[ival];
            val = (float)ival;
        }
        return val;
    }

    void writeTrainingParams( FileStorage& fs ) const
    {
        fs << "boosting_type" <<
        (bparams.boostType == Boost::DISCRETE ? "DiscreteAdaboost" :
        bparams.boostType == Boost::REAL ? "RealAdaboost" :
        bparams.boostType == Boost::LOGIT ? "LogitBoost" :
        bparams.boostType == Boost::GENTLE ? "GentleAdaboost" : "Unknown");

        DTreesImpl::writeTrainingParams(fs);
        fs << "weight_trimming_rate" << bparams.weightTrimRate;
    }

    void write( FileStorage& fs ) const
    {
        if( roots.empty() )
            CV_Error( CV_StsBadArg, "RTrees have not been trained" );

        writeParams(fs);

        int k, ntrees = (int)roots.size();

        fs << "ntrees" << ntrees
        << "trees" << "[";

        for( k = 0; k < ntrees; k++ )
        {
            fs << "{";
            writeTree(fs, roots[k]);
            fs << "}";
        }

        fs << "]";
    }

    void readParams( const FileNode& fn )
    {
        DTreesImpl::readParams(fn);
        bparams.maxDepth = params0.maxDepth;
        bparams.minSampleCount = params0.minSampleCount;
        bparams.regressionAccuracy = params0.regressionAccuracy;
        bparams.useSurrogates = params0.useSurrogates;
        bparams.maxCategories = params0.maxCategories;
        bparams.priors = params0.priors;

        FileNode tparams_node = fn["training_params"];
        String bts = (String)tparams_node["boosting_type"];
        bparams.boostType = (bts == "DiscreteAdaboost" ? Boost::DISCRETE :
                             bts == "RealAdaboost" ? Boost::REAL :
                             bts == "LogitBoost" ? Boost::LOGIT :
                             bts == "GentleAdaboost" ? Boost::GENTLE : -1);
        bparams.weightTrimRate = (double)tparams_node["weight_trimming_rate"];
    }

    void read( const FileNode& fn )
    {
        clear();

        int ntrees = (int)fn["ntrees"];
        readParams(fn);

        FileNode trees_node = fn["trees"];
        FileNodeIterator it = trees_node.begin();
        CV_Assert( ntrees == (int)trees_node.size() );
        
        for( int treeidx = 0; treeidx < ntrees; treeidx++, ++it )
        {
            FileNode nfn = (*it)["nodes"];
            readTree(nfn);
        }
    }
    
    Boost::Params bparams;
    vector<double> boostWeights;
    vector<double> sumResult;
};


class BoostImpl : public Boost
{
public:
    BoostImpl() {}

    bool train( const Ptr<TrainData>& trainData, int flags )
    {
        return impl.train(trainData, flags);
    }

    float predict( InputArray samples, OutputArray results, int flags ) const
    {
        return impl.predict(samples, results, flags);
    }

    void write( FileStorage& fs ) const
    {
        impl.write(fs);
    }

    void read( const FileNode& fn )
    {
        impl.read(fn);
    }

    void setParams(const Params& p) { impl.setBParams(p); }
    Params getParams() const { return impl.getBParams(); }

    const vector<int>& getRoots() const { return impl.getRoots(); }
    const vector<Node>& getNodes() const { return impl.getNodes(); }
    const vector<Split>& getSplits() const { return impl.getSplits(); }
    const vector<int>& getSubsets() const { return impl.getSubsets(); }

    DTreesImplForBoost impl;
};


Ptr<Boost> createBoost(const Ptr<TrainData>& trainData, const Boost::Params& params)
{
    Ptr<BoostImpl> p = makePtr<BoostImpl>();
    p->setParams(params);
    if( !p->train(trainData, 0) )
        p.release();
    return p;
}

}}

/* End of file. */

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
#include <ctype.h>

namespace cv {
namespace ml {

using std::vector;

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

DTrees::Params::Params()
{
    maxDepth = INT_MAX;
    minSampleCount = 10;
    regressionAccuracy = 0.01f;
    useSurrogates = false;
    maxCategories = 10;
    CVFolds = 10;
    use1SERule = true;
    truncatePrunedTree = true;
    priors = Mat();
}

DTrees::Params::Params( int _maxDepth, int _minSampleCount,
                        double _regressionAccuracy, bool _useSurrogates,
                        int _maxCategories, int _CVFolds,
                        bool _use1SERule, bool _truncatePrunedTree,
                        const Mat& _priors )
{
    maxDepth = _maxDepth;
    minSampleCount = _minSampleCount;
    regressionAccuracy = (float)_regressionAccuracy;
    useSurrogates = _useSurrogates;
    maxCategories = _maxCategories;
    CVFolds = _CVFolds;
    use1SERule = _use1SERule;
    truncatePrunedTree = _truncatePrunedTree;
    priors = _priors;
}

DTrees::Node::Node()
{
    classIdx = 0;
    value = 0;
    parent = left = right = split = -1;
}

DTrees::Split::Split()
{
    varIdx = 0;
    condensedIdx = 0;
    inversed = false;
    quality = 0.f;
    next = -1;
    c = 0.f;
    subsetOfs = 0;
}


DTreesImpl::WorkData::WorkData(const Ptr<TrainData>& _data)
{
    data = _data;
    vector<int> subsampleIdx;
    Mat sidx0 = _data->getTrainSampleIdx();
    if( !sidx0.empty() )
        sidx0.copyTo(sidx);
    else
    {
        int n = _data->getNSamples();
        setRangeVector(sidx, n);
    }
    Mat vidx0 = _data->getVarIdx();
    if( !vidx0.empty() )
        vidx0.copyTo(vidx);
    else
        setRangeVector(vidx, _data->getNAllVars());

    maxSubsetSize = 1 /* TODO */;
}

int DTreesImpl::WorkData::getNumValid(int, const vector<int>& _sidx) const
{
    return (int)_sidx.size();
}

void DTreesImpl::WorkData::getOrdVarData( int vi, const vector<int>& _sidx, float* values )
{
    int i, n = (int)_sidx.size();
    const int* s = n > 0 ? &_sidx[0] : 0;

    size_t step = samples.step/samples.elemSize();
    size_t sstep = layout == ROW_SAMPLE ? step : 1;
    size_t vstep = layout == ROW_SAMPLE ? 1 : step;

    const float* src = samples.ptr<float>() + vi*vstep;
    for( i = 0; i < n; i++ )
        values[i] = src[s[i]*sstep];
}

void DTreesImpl::WorkData::getCatVarData( int vi, const vector<int>& _sidx, int* values )
{
    int i, n = (int)_sidx.size();
    const int* s = n > 0 ? &_sidx[0] : 0;

    size_t step = samples.step/samples.elemSize();
    size_t sstep = layout == ROW_SAMPLE ? step : 1;
    size_t vstep = layout == ROW_SAMPLE ? 1 : step;

    const float* src = samples.ptr<float>() + vi*vstep;
    for( i = 0; i < n; i++ )
        values[i] = src[s[i]*sstep];
}

DTreesImpl::DTreesImpl() {}
DTreesImpl::~DTreesImpl() {}
void DTreesImpl::clear() {}

void DTreesImpl::startTraining( const Ptr<TrainData>& trainData, int )
{
    clear();
    w = makePtr<WorkData>(trainData);
}

void DTreesImpl::endTraining()
{
    w.release();
}

bool DTreesImpl::train( const Ptr<TrainData>& trainData, int flags )
{
    startTraining(trainData, flags);
    bool ok = addTree( w->sidx ) >= 0;
    w.release();
    endTraining();
    return ok;
}

const vector<int>& DTreesImpl::getActiveVars()
{
    return w->vidx;
}

int DTreesImpl::addTree(const vector<int>& sidx )
{
    size_t n = (params.maxDepth > 0 ? (1 << params.maxDepth) : 1024) + w->wnodes.size();

    w->wnodes.reserve(n);
    w->wsplits.reserve(n);
    w->wsubsets.reserve(n*w->maxSubsetSize);

    int cv_n = params.CVFolds;

    if( cv_n > 0 )
    {
        w->cv_Tn.resize(n*cv_n);
        w->cv_node_error.resize(n*cv_n);
        w->cv_node_risk.resize(n*cv_n);
    }

    w->cls_count.resize(classLabels.size());

    // build the tree recursively
    int root = addNodeAndTrySplit(-1, sidx);

    int pruned_tree_idx = pruneCV(root);

    /*void CvDTree::free_prune_data(bool _cut_tree)
    {
        CvDTreeNode* node = root;

        for(;;)
        {
            CvDTreeNode* parent;
            for(;;)
            {
                // do not call cvSetRemoveByPtr( cv_heap, node->cv_Tn )
                // as we will clear the whole cross-validation heap at the end
                node->cv_Tn = 0;
                node->cv_node_error = node->cv_node_risk = 0;
                if( !node->left )
                    break;
                node = node->left;
            }

            for( parent = node->parent; parent && parent->right == node;
                node = parent, parent = parent->parent )
            {
                if( _cut_tree && parent->Tn <= pruned_tree_idx )
                {
                    data->free_node( parent->left );
                    data->free_node( parent->right );
                    parent->left = parent->right = 0;
                }
            }
            
            if( !parent )
                break;
            
            node = parent->right;
        }
        
        if( data->cv_heap )
            cvClearSet( data->cv_heap );
    }*/
    roots.push_back(root);
    return root;
}

void DTreesImpl::setParams(const Params& _params)
{
    params0 = params = _params;
    if( params.maxCategories < 2 )
        CV_Error( CV_StsOutOfRange, "params.max_categories should be >= 2" );
    params.maxCategories = std::min( params.maxCategories, 15 );

    if( params.maxDepth < 0 )
        CV_Error( CV_StsOutOfRange, "params.max_depth should be >= 0" );
    params.maxDepth = std::min( params.maxDepth, 25 );

    params.minSampleCount = std::max(params.minSampleCount, 1);

    if( params.CVFolds < 0 )
        CV_Error( CV_StsOutOfRange,
                 "params.CVFolds should be =0 (the tree is not pruned) "
                 "or n>0 (tree is pruned using n-fold cross-validation)" );

    if( params.CVFolds == 1 )
        params.CVFolds = 0;
    
    if( params.regressionAccuracy < 0 )
        CV_Error( CV_StsOutOfRange, "params.regression_accuracy should be >= 0" );
}

int DTreesImpl::addNodeAndTrySplit( int parent, const vector<int>& sidx )
{
    w->wnodes.push_back(WNode());
    int nidx = (int)(w->wnodes.size() - 1);
    WNode& node = w->wnodes.back();

    node.parent = parent;
    node.depth = parent >= 0 ? w->wnodes[parent].depth + 1 : 0;

    int nfolds = params.CVFolds;

    if( nfolds > 0 )
    {
        w->cv_Tn.resize((nidx+1)*nfolds);
        w->cv_node_error.resize((nidx+1)*nfolds);
        w->cv_node_risk.resize((nidx+1)*nfolds);
    }

    int n = node.sample_count = (int)sidx.size();
    bool can_split = true;
    vector<int> sleft, sright;

    calcValue( nidx, sidx );

    if( n <= params.minSampleCount || node.depth >= params.maxDepth )
        can_split = false;
    else if( isClassifier )
    {
        int nz = countNonZero(w->cls_count);
        if( nz <= 1 )
            can_split = false;
    }
    else
    {
        if( sqrt(node.node_risk)/n < params.regressionAccuracy )
            can_split = false;
    }

    if( can_split )
        node.split = findBestSplit( sidx );

    if( node.split >= 0 )
    {
        calcDir( node.split, sidx, sleft, sright );
        if( params.useSurrogates )
            CV_Error( CV_StsNotImplemented, "surrogate splits are not implemented yet");

        node.left = addNodeAndTrySplit( nidx, sleft );
        node.right = addNodeAndTrySplit( nidx, sright );
    }
    return nidx;
}

int DTreesImpl::findBestSplit( const vector<int>& _sidx )
{
    const vector<int>& activeVars = getActiveVars();
    int splitidx = -1;
    int vi, nv = (int)activeVars.size();
    AutoBuffer<int> buf(w->maxSubsetSize*2);
    int *subset = buf, *best_subset = subset + w->maxSubsetSize;
    int best_mi = 0;
    WSplit split, best_split;
    best_split.quality = 0.;

    for( vi = 0; vi < nv; vi++ )
    {
        int mi = catCount[vi];
        if( mi <= 0 )
        {
            if( isClassifier )
                split = findSplitOrdClass(vi, _sidx, best_split.quality);
            else
                split = findSplitOrdReg(vi, _sidx, best_split.quality);
        }
        else
        {
            if( isClassifier )
                split = findSplitCatClass(vi, _sidx, best_split.quality, subset);
            else
                split = findSplitCatReg(vi, _sidx, best_split.quality, subset);
        }
        if( split.quality > best_split.quality )
        {
            best_split = split;
            best_mi = mi;
            std::swap(subset, best_subset);
        }
    }

    if( best_split.quality > 0 )
    {
        if( best_mi > 0 )
        {
            int i, prevsz = (int)w->wsubsets.size(), ssize = (best_mi + 31)/32;
            w->wsubsets.resize(prevsz + ssize);
            for( i = 0; i < ssize; i++ )
                w->wsubsets[prevsz + i] = best_subset[i];
            best_split.subset_ofs = prevsz;
        }
        w->wsplits.push_back(best_split);
        splitidx = (int)(w->wsplits.size()-1);
    }

    return splitidx;
}

void DTreesImpl::calcValue( int nidx, const vector<int>& _sidx )
{
    WNode* node = &w->wnodes[nidx];
    int i, j, k, n = (int)_sidx.size(), cv_n = params.CVFolds;
    int m = (int)classLabels.size();
    int nvars = getVarCount();

    cv::AutoBuffer<double> buf(std::max(m, 3)*std::max(cv_n,1));

    if( cv_n > 0 )
    {
        size_t sz = w->cv_Tn.size();
        w->cv_Tn.resize(sz + cv_n);
        w->cv_node_risk.resize(sz + cv_n);
        w->cv_node_error.resize(sz + cv_n);
    }

    if( isClassifier )
    {
        // in case of classification tree:
        //  * node value is the label of the class that has the largest weight in the node.
        //  * node risk is the weighted number of misclassified samples,
        //  * j-th cross-validation fold value and risk are calculated as above,
        //    but using the samples with cv_labels(*)!=j.
        //  * j-th cross-validation fold error is calculated as the weighted number of
        //    misclassified samples with cv_labels(*)==j.

        // compute the number of instances of each class
        double* cv_cls_count = buf;
        double* cls_count = &w->cls_count[0];

        double max_val = -1, total_weight = 0;
        int max_k = -1;

        for( k = 0; k < m; k++ )
            cls_count[k] = 0;

        if( cv_n == 0 )
        {
            for( i = 0; i < n; i++ )
            {
                int idx = _sidx[i];
                cls_count[w->cat_responses[idx]] += w->sample_weights[idx];
            }
        }
        else
        {
            for( j = 0; j < cv_n; j++ )
                for( k = 0; k < m; k++ )
                    cv_cls_count[j*m + k] = 0;

            for( i = 0; i < n; i++ )
            {
                int idx = _sidx[i];
                j = w->cv_labels[idx]; k = w->cat_responses[idx];
                cv_cls_count[j*m + k] += w->sample_weights[idx];
            }

            for( j = 0; j < cv_n; j++ )
                for( k = 0; k < m; k++ )
                    cls_count[k] += cv_cls_count[j*m + k];
        }

        for( k = 0; k < m; k++ )
        {
            double val = cls_count[k];
            total_weight += val;
            if( max_val < val )
            {
                max_val = val;
                max_k = k;
            }
        }

        node->class_idx = max_k;
        node->value = catMap[catOfs[nvars] + max_k];
        node->node_risk = total_weight - max_val;

        for( j = 0; j < cv_n; j++ )
        {
            double sum_k = 0, sum = 0, max_val_k = 0;
            max_val = -1; max_k = -1;

            for( k = 0; k < m; k++ )
            {
                double val_k = cv_cls_count[j*m + k];
                double val = cls_count[k] - val_k;
                sum_k += val_k;
                sum += val;
                if( max_val < val )
                {
                    max_val = val;
                    max_val_k = val_k;
                    max_k = k;
                }
            }

            w->cv_Tn[nidx*cv_n + j] = INT_MAX;
            w->cv_node_risk[nidx*cv_n + j] = sum - max_val;
            w->cv_node_error[nidx*cv_n + j] = sum_k - max_val_k;
        }
    }
    else
    {
        // in case of regression tree:
        //  * node value is 1/n*sum_i(Y_i), where Y_i is i-th response,
        //    n is the number of samples in the node.
        //  * node risk is the sum of squared errors: sum_i((Y_i - <node_value>)^2)
        //  * j-th cross-validation fold value and risk are calculated as above,
        //    but using the samples with cv_labels(*)!=j.
        //  * j-th cross-validation fold error is calculated
        //    using samples with cv_labels(*)==j as the test subset:
        //    error_j = sum_(i,cv_labels(i)==j)((Y_i - <node_value_j>)^2),
        //    where node_value_j is the node value calculated
        //    as described in the previous bullet, and summation is done
        //    over the samples with cv_labels(*)==j.
        double sum = 0, sum2 = 0;

        if( cv_n == 0 )
        {
            for( i = 0; i < n; i++ )
            {
                int idx = _sidx[i];
                double t = w->ord_responses[idx]*w->sample_weights[idx];
                sum += t;
                sum2 += t*t;
            }
        }
        else
        {
            double *cv_sum = buf, *cv_sum2 = cv_sum + cv_n;
            int* cv_count = (int*)(cv_sum2 + cv_n);

            for( j = 0; j < cv_n; j++ )
            {
                cv_sum[j] = cv_sum2[j] = 0.;
                cv_count[j] = 0;
            }

            for( i = 0; i < n; i++ )
            {
                int idx = _sidx[i];
                j = w->cv_labels[idx];
                double t = w->ord_responses[idx];
                cv_sum[j] += t;
                cv_sum2[j] += t*t;
                cv_count[j]++;
            }
            
            for( j = 0; j < cv_n; j++ )
            {
                sum += cv_sum[j];
                sum2 += cv_sum2[j];
            }

            for( j = 0; j < cv_n; j++ )
            {
                double s = sum - cv_sum[j], si = sum - s;
                double s2 = sum2 - cv_sum2[j], s2i = sum2 - s2;
                int c = cv_count[j], ci = n - c;
                double r = si/std::max(ci, 1);
                w->cv_node_risk[nidx*cv_n + j] = s2i - r*r*ci;
                w->cv_node_error[nidx*cv_n + j] = s2 - 2*r*s + c*r*r;
                w->cv_Tn[nidx*cv_n + j] = INT_MAX;
            }
        }
        
        node->node_risk = sum2 - (sum/n)*sum;
        node->value = sum/n;
    }
}

DTreesImpl::WSplit DTreesImpl::findSplitOrdClass( int vi, const vector<int>& _sidx, double initQuality )
{
    const double epsilon = FLT_EPSILON*2;
    int n = (int)_sidx.size();
    int m = (int)classLabels.size();

    cv::AutoBuffer<uchar> buf(n*(sizeof(float) + sizeof(int)) + m*2*sizeof(double));
    const int* sidx = &_sidx[0];
    const int* responses = &w->cat_responses[0];
    const double* weights = &w->sample_weights[0];
    double* lcw = (double*)(uchar*)buf;
    double* rcw = lcw + m;
    float* values = (float*)(rcw + m);
    int* sorted_idx = (int*)(values + n);
    int i, best_i = -1;
    double best_val = initQuality;

    for( i = 0; i < m; i++ )
        lcw[i] = rcw[i] = 0.;

    w->getOrdVarData( vi, _sidx, values );

    for( i = 0; i < n; i++ )
    {
        sorted_idx[i] = i;
        int j = sidx[i];
        rcw[responses[j]] += weights[j];
    }

    std::sort(sorted_idx, sorted_idx + n, cmp_lt_idx<float>(values));

    double L = 0, R = 0, lsum2 = 0, rsum2 = 0;
    for( i = 0; i < m; i++ )
    {
        double wval = rcw[i];
        R += wval;
        rsum2 += wval*wval;
    }

    for( i = 0; i < n - 1; i++ )
    {
        int curr = sorted_idx[i];
        int next = sorted_idx[i+1];
        int j = sidx[curr];
        double wval = weights[j], w2 = wval*wval;
        L += wval; R -= wval;
        int idx = responses[j];
        double lv = lcw[idx], rv = rcw[idx];
        lsum2 += 2*lv*wval + w2;
        rsum2 -= 2*rv*wval - w2;
        lcw[idx] = lv + wval; rcw[idx] = rv - wval;

        if( values[curr] + epsilon < values[next] )
        {
            double val = (lsum2*R + rsum2*L)/(L*R);
            if( best_val < val )
            {
                best_val = val;
                best_i = i;
            }
        }
    }

    WSplit split;
    if( best_i >= 0 )
    {
        split.var_idx = vi;
        split.c = (values[sorted_idx[best_i]] + values[sorted_idx[best_i+1]])*0.5f;
        split.inversed = 0;
        split.quality = (float)best_val;
    }
    return split;
}

// simple k-means, slightly modified to take into account the "weight" (L1-norm) of each vector.
void DTreesImpl::clusterCategories( const double* vectors, int n, int m, double* csums, int k, int* labels )
{
    int iters = 0, max_iters = 100;
    int i, j, idx;
    cv::AutoBuffer<double> buf(n + k);
    double *v_weights = buf, *c_weights = buf + n;
    bool modified = true;
    RNG r(-1);

    // assign labels randomly
    for( i = 0; i < n; i++ )
    {
        int sum = 0;
        const double* v = vectors + i*m;
        labels[i] = i < k ? i : r.uniform(0, k);

        // compute weight of each vector
        for( j = 0; j < m; j++ )
            sum += v[j];
        v_weights[i] = sum ? 1./sum : 0.;
    }

    for( i = 0; i < n; i++ )
    {
        int i1 = r.uniform(0, n);
        int i2 = r.uniform(0, n);
        std::swap( labels[i1], labels[i2] );
    }

    for( iters = 0; iters <= max_iters; iters++ )
    {
        // calculate csums
        for( i = 0; i < k; i++ )
        {
            for( j = 0; j < m; j++ )
                csums[i*m + j] = 0;
        }

        for( i = 0; i < n; i++ )
        {
            const double* v = vectors + i*m;
            double* s = csums + labels[i]*m;
            for( j = 0; j < m; j++ )
                s[j] += v[j];
        }

        // exit the loop here, when we have up-to-date csums
        if( iters == max_iters || !modified )
            break;

        modified = false;

        // calculate weight of each cluster
        for( i = 0; i < k; i++ )
        {
            const double* s = csums + i*m;
            double sum = 0;
            for( j = 0; j < m; j++ )
                sum += s[j];
            c_weights[i] = sum ? 1./sum : 0;
        }

        // now for each vector determine the closest cluster
        for( i = 0; i < n; i++ )
        {
            const double* v = vectors + i*m;
            double alpha = v_weights[i];
            double min_dist2 = DBL_MAX;
            int min_idx = -1;

            for( idx = 0; idx < k; idx++ )
            {
                const double* s = csums + idx*m;
                double dist2 = 0., beta = c_weights[idx];
                for( j = 0; j < m; j++ )
                {
                    double t = v[j]*alpha - s[j]*beta;
                    dist2 += t*t;
                }
                if( min_dist2 > dist2 )
                {
                    min_dist2 = dist2;
                    min_idx = idx;
                }
            }
            
            if( min_idx != labels[i] )
                modified = true;
            labels[i] = min_idx;
        }
    }
}

DTreesImpl::WSplit DTreesImpl::findSplitCatClass( int vi, const vector<int>& _sidx, double initQuality, int* subset )
{
    int _mi = catCount[vi], mi = _mi;
    int n = (int)_sidx.size();
    int m = (int)classLabels.size();

    int base_size = m*(3 + mi) + mi + 1;
    if( m > 2 && mi > params.maxCategories )
        base_size += m*std::min(params.maxCategories, n) + mi;
    else
        base_size += mi;
    AutoBuffer<double> buf(base_size + n);

    double* lc = (double*)buf;
    double* rc = lc + m;
    double* _cjk = rc + m*2, *cjk = _cjk;
    double* c_weights = cjk + m*mi;

    int* labels = (int*)(buf + base_size);
    w->getCatVarData(vi, _sidx, labels);
    const int* responses = &w->cat_responses[0];
    const double* weights = &w->sample_weights[0];

    int* cluster_labels = 0;
    double** dbl_ptr = 0;
    int i, j, k, idx;
    double L = 0, R = 0;
    double best_val = initQuality;
    int prevcode = 0, best_subset = -1, subset_i, subset_n, subtract = 0;

    // init array of counters:
    // c_{jk} - number of samples that have vi-th input variable = j and response = k.
    for( j = -1; j < mi; j++ )
        for( k = 0; k < m; k++ )
            cjk[j*m + k] = 0;

    for( i = 0; i < n; i++ )
    {
        idx = _sidx[i];
        j = labels[i];
        k = responses[i];
        cjk[j*m + k] += weights[idx];
    }

    if( m > 2 )
    {
        if( mi > params.maxCategories )
        {
            mi = std::min(params.maxCategories, n);
            cjk = c_weights + _mi;
            cluster_labels = (int*)(cjk + m*mi);
            clusterCategories( _cjk, _mi, m, cjk, mi, cluster_labels );
        }
        subset_i = 1;
        subset_n = 1 << mi;
    }
    else
    {
        assert( m == 2 );
        dbl_ptr = (double**)(c_weights + _mi);
        for( j = 0; j < mi; j++ )
            dbl_ptr[j] = cjk + j*2 + 1;
        std::sort(dbl_ptr, dbl_ptr + mi, cmp_lt_ptr<double>());
        subset_i = 0;
        subset_n = mi;
    }

    for( k = 0; k < m; k++ )
    {
        double sum = 0;
        for( j = 0; j < mi; j++ )
            sum += cjk[j*m + k];
        rc[k] = sum;
        lc[k] = 0;
    }

    for( j = 0; j < mi; j++ )
    {
        double sum = 0;
        for( k = 0; k < m; k++ )
            sum += cjk[j*m + k];
        c_weights[j] = sum;
        R += c_weights[j];
    }

    for( ; subset_i < subset_n; subset_i++ )
    {
        double lsum2 = 0, rsum2 = 0;

        if( m == 2 )
            idx = (int)(dbl_ptr[subset_i] - cjk)/2;
        else
        {
            int graycode = (subset_i>>1)^subset_i;
            int diff = graycode ^ prevcode;

            // determine index of the changed bit.
            Cv32suf u;
            idx = diff >= (1 << 16) ? 16 : 0;
            u.f = (float)(((diff >> 16) | diff) & 65535);
            idx += (u.i >> 23) - 127;
            subtract = graycode < prevcode;
            prevcode = graycode;
        }

        double* crow = cjk + idx*m;
        double weight = c_weights[idx];
        if( weight < FLT_EPSILON )
            continue;

        if( !subtract )
        {
            for( k = 0; k < m; k++ )
            {
                double t = crow[k];
                double lval = lc[k] + t;
                double rval = rc[k] - t;
                lsum2 += lval*lval;
                rsum2 += rval*rval;
                lc[k] = lval; rc[k] = rval;
            }
            L += weight;
            R -= weight;
        }
        else
        {
            for( k = 0; k < m; k++ )
            {
                double t = crow[k];
                double lval = lc[k] - t;
                double rval = rc[k] + t;
                lsum2 += lval*lval;
                rsum2 += rval*rval;
                lc[k] = lval; rc[k] = rval;
            }
            L -= weight;
            R += weight;
        }

        if( L > FLT_EPSILON && R > FLT_EPSILON )
        {
            double val = (lsum2*R + rsum2*L)/(L*R);
            if( best_val < val )
            {
                best_val = val;
                best_subset = subset_i;
            }
        }
    }

    WSplit split;
    if( best_subset >= 0 )
    {
        split.var_idx = vi;
        split.quality = (float)best_val;
        memset( subset, 0, (_mi + 31)/32 * sizeof(int) );
        if( m == 2 )
        {
            for( i = 0; i <= best_subset; i++ )
            {
                idx = (int)(dbl_ptr[i] - cjk) >> 1;
                subset[idx >> 5] |= 1 << (idx & 31);
            }
        }
        else
        {
            for( i = 0; i < _mi; i++ )
            {
                idx = cluster_labels ? cluster_labels[i] : i;
                if( best_subset & (1 << idx) )
                    subset[i >> 5] |= 1 << (i & 31);
            }
        }
    }
    return split;
}

DTreesImpl::WSplit DTreesImpl::findSplitOrdReg( int vi, const vector<int>& _sidx, double initQuality )
{
    const float epsilon = FLT_EPSILON*2;
    const double* weights = &w->sample_weights[0];
    int n = (int)_sidx.size();

    AutoBuffer<uchar> buf(n*(sizeof(int) + sizeof(float)));

    float* values = (float*)(uchar*)buf;
    int* sorted_idx = (int*)(values + n);
    w->getOrdVarData(vi, _sidx, values);
    const double* responses = &w->ord_responses[0];

    int i, j, best_i = -1;
    double L = 0, R = 0;
    double best_val = initQuality, lsum = 0, rsum = 0;

    for( i = 0; i < n; i++ )
    {
        sorted_idx[i] = i;
        j = _sidx[i];
        R += weights[j];
        rsum += weights[j]*responses[j];
    }

    std::sort(sorted_idx, sorted_idx + n, cmp_lt_idx<float>(values));

    // find the optimal split
    for( i = 0; i < n - 1; i++ )
    {
        int curr = sorted_idx[i];
        int next = sorted_idx[i+1];
        j = _sidx[curr];
        double wval = weights[j];
        double t = responses[j]*wval;
        L += wval; R -= wval;
        lsum += t; rsum -= t;

        if( values[curr] + epsilon < values[next] )
        {
            double val = (lsum*lsum*R + rsum*rsum*L)/(L*R);
            if( best_val < val )
            {
                best_val = val;
                best_i = i;
            }
        }
    }

    WSplit split;
    if( best_i >= 0 )
    {
        split.var_idx = vi;
        split.c = (values[sorted_idx[best_i]] + values[sorted_idx[best_i+1]])*0.5f;
        split.inversed = 0;
        split.quality = (float)best_val;
    }
    return split;
}

DTreesImpl::WSplit DTreesImpl::findSplitCatReg( int vi, const vector<int>& _sidx, double initQuality, int* subset )
{
    const double* weights = &w->sample_weights[0];
    const double* responses = &w->ord_responses[0];
    int n = (int)_sidx.size();
    int mi = catCount[vi];

    AutoBuffer<double> buf(3*mi + 3 + n);
    double* sum = buf;
    double* counts = sum + mi + 1;
    double** sum_ptr = (double**)(counts + mi);
    int* cat_labels = (int*)(sum_ptr + mi);

    w->getCatVarData(vi, _sidx, cat_labels);

    double L = 0, R = 0, best_val = initQuality, lsum = 0, rsum = 0;
    int i, j, best_subset = -1, subset_i;

    for( i = -1; i < mi; i++ )
        sum[i] = counts[i] = 0;

    // calculate sum response and weight of each category of the input var
    for( i = 0; i < n; i++ )
    {
        int idx = cat_labels[i];
        j = _sidx[i];
        double wval = weights[j];
        sum[idx] += responses[j]*wval;
        counts[idx] += wval;
    }

    // calculate average response in each category
    for( i = 0; i < mi; i++ )
    {
        R += counts[i];
        rsum += sum[i];
        sum[i] = fabs(counts[i]) > DBL_EPSILON ? sum[i]/counts[i] : 0;
        sum_ptr[i] = sum + i;
    }

    std::sort(sum_ptr, sum_ptr + mi, cmp_lt_ptr<double>());

    // revert back to unnormalized sums
    // (there should be a very little loss in accuracy)
    for( i = 0; i < mi; i++ )
        sum[i] *= counts[i];
    
    for( subset_i = 0; subset_i < mi-1; subset_i++ )
    {
        int idx = (int)(sum_ptr[subset_i] - sum);
        double ni = counts[idx];
        
        if( ni > FLT_EPSILON )
        {
            double s = sum[idx];
            lsum += s; L += ni;
            rsum -= s; R -= ni;
            
            if( L > FLT_EPSILON && R > FLT_EPSILON )
            {
                double val = (lsum*lsum*R + rsum*rsum*L)/(L*R);
                if( best_val < val )
                {
                    best_val = val;
                    best_subset = subset_i;
                }
            }
        }
    }
    
    WSplit split;
    if( best_subset >= 0 )
    {
        split.var_idx = vi;
        split.quality = (float)best_val;
        memset( subset, 0, (mi + 31)/32 * sizeof(int));
        for( i = 0; i <= best_subset; i++ )
        {
            int idx = (int)(sum_ptr[i] - sum);
            subset[idx >> 5] |= 1 << (idx & 31);
        }
    }
    return split;
}

void DTreesImpl::calcDir( int splitidx, const vector<int>& _sidx, vector<int>& _sleft, vector<int>& _sright )
{
    WSplit split = w->wsplits[splitidx];
    int i, j, n = (int)_sidx.size(), vi = split.var_idx;
    _sleft.reserve(n);
    _sright.reserve(n);

    AutoBuffer<float> buf(n);
    int mi = catCount[vi];

    if( mi <= 0 ) // split on an ordered variable
    {
        float c = split.c;
        float* values = buf;
        w->getOrdVarData(vi, _sidx, values);

        for( i = 0; i < n; i++ )
        {
            j = _sidx[i];
            if( values[i] <= c )
                _sleft.push_back(j);
            else
                _sright.push_back(j);
        }
    }
    else
    {
        const int* subset = &w->wsubsets[split.subset_ofs];
        int* cat_labels = (int*)(float*)buf;
        w->getCatVarData(vi, _sidx, cat_labels);

        for( i = 0; i < n; i++ )
        {
            j = _sidx[i];
            unsigned u = cat_labels[i];
            if( CV_DTREE_CAT_DIR(u, subset) < 0 )
                _sleft.push_back(j);
            else
                _sright.push_back(j);
        }
    }
}

int DTreesImpl::pruneCV( int root )
{
    vector<double> ab;

    // 1. build tree sequence for each cv fold, calculate error_{Tj,beta_k}.
    // 2. choose the best tree index (if need, apply 1SE rule).
    // 3. store the best index and cut the branches.

    int ti, tree_count = 0, j, cv_n = params.CVFolds, n = w->wnodes[root].sample_count;
    // currently, 1SE for regression is not implemented
    bool use_1se = params.use1SERule != 0 && isClassifier;
    double min_err = 0, min_err_se = 0;
    int min_idx = -1;

    // build the main tree sequence, calculate alpha's
    for(;;tree_count++)
    {
        double min_alpha = updateTreeRNC(root, tree_count, -1);
        if( cutTree(root, tree_count, -1, min_alpha) )
            break;

        ab.push_back(min_alpha);
    }

    if( tree_count > 0 )
    {
        ab[0] = 0.;

        for( ti = 1; ti < tree_count-1; ti++ )
            ab[ti] = std::sqrt(ab[ti]*ab[ti+1]);
        ab[tree_count-1] = DBL_MAX*0.5;

        Mat err_jk(cv_n, tree_count, CV_64F);

        for( j = 0; j < cv_n; j++ )
        {
            int tj = 0, tk = 0;
            for( ; tj < tree_count; tj++ )
            {
                double min_alpha = updateTreeRNC(root, tj, j);
                if( cutTree(root, tj, j, min_alpha) )
                    min_alpha = DBL_MAX;

                for( ; tk < tree_count; tk++ )
                {
                    if( ab[tk] > min_alpha )
                        break;
                    err_jk.at<double>(j, tk) = w->wnodes[root].tree_error;
                }
            }
        }

        for( ti = 0; ti < tree_count; ti++ )
        {
            double sum_err = 0;
            for( j = 0; j < cv_n; j++ )
                sum_err += err_jk.at<double>(j, ti);
            if( ti == 0 || sum_err < min_err )
            {
                min_err = sum_err;
                min_idx = ti;
                if( use_1se )
                    min_err_se = sqrt( sum_err*(n - sum_err) );
            }
            else if( sum_err < min_err + min_err_se )
                min_idx = ti;
        }
    }

    return min_idx;
}

double DTreesImpl::updateTreeRNC( int root, double T, int fold )
{
    int nidx = root, pidx = -1, cv_n = params.CVFolds;
    double min_alpha = DBL_MAX;

    for(;;)
    {
        WNode *node = 0, *parent = 0;

        for(;;)
        {
            node = &w->wnodes[nidx];
            double t = fold >= 0 ? w->cv_Tn[nidx*cv_n + fold] : node->Tn;
            if( t <= T || node->left < 0 )
            {
                node->complexity = 1;
                node->tree_risk = node->node_risk;
                node->tree_error = 0.;
                if( fold >= 0 )
                {
                    node->tree_risk = w->cv_node_risk[nidx*cv_n + fold];
                    node->tree_error = w->cv_node_error[nidx*cv_n + fold];
                }
                break;
            }
            nidx = node->left;
        }

        for( pidx = node->parent; pidx >= 0 && w->wnodes[pidx].right == nidx;
             nidx = pidx, pidx = w->wnodes[pidx].parent )
        {
            node = &w->wnodes[nidx];
            parent = &w->wnodes[pidx];
            parent->complexity += node->complexity;
            parent->tree_risk += node->tree_risk;
            parent->tree_error += node->tree_error;

            parent->alpha = ((fold >= 0 ? w->cv_node_risk[pidx*cv_n + fold] : parent->node_risk)
                             - parent->tree_risk)/(parent->complexity - 1);
            min_alpha = std::min( min_alpha, parent->alpha );
        }

        if( pidx < 0 )
            break;

        node = &w->wnodes[nidx];
        parent = &w->wnodes[pidx];
        parent->complexity = node->complexity;
        parent->tree_risk = node->tree_risk;
        parent->tree_error = node->tree_error;
        nidx = parent->right;
    }

    return min_alpha;
}

bool DTreesImpl::cutTree( int root, double T, int fold, double min_alpha )
{
    int cv_n = params.CVFolds, nidx = root, pidx = -1;
    WNode* node = &w->wnodes[root];
    if( node->left < 0 )
        return true;

    for(;;)
    {
        for(;;)
        {
            node = &w->wnodes[nidx];
            double t = fold >= 0 ? w->cv_Tn[nidx*cv_n + fold] : node->Tn;
            if( t <= T || node->left < 0 )
                break;
            if( node->alpha <= min_alpha + FLT_EPSILON )
            {
                if( fold >= 0 )
                    w->cv_Tn[nidx*cv_n + fold] = T;
                else
                    node->Tn = T;
                if( nidx == root )
                    return true;
                break;
            }
            nidx = node->left;
        }
        
        for( pidx = node->parent; pidx >= 0 && w->wnodes[pidx].right == nidx;
             nidx = pidx, pidx = w->wnodes[pidx].parent )
            ;
        
        if( pidx < 0 )
            break;
        
        nidx = w->wnodes[pidx].right;
    }
    
    return false;
}

float DTreesImpl::predictTrees( const Range& range, const Mat& sample, int flags ) const
{
    CV_Assert( sample.type() == CV_32F );

    int predictType = flags & PREDICT_MASK;
    int i, ncats = (int)catOfs.size(), nclasses = (int)classLabels.size();
    AutoBuffer<double> buf(nclasses + ncats + 1);
    double* votes = buf;
    int* catbuf = (int*)(votes + nclasses);
    const int* vidx = (flags & (COMPRESSED_INPUT|PREPROCESSED_INPUT)) == 0 && !varIdx.empty() ? &varIdx[0] : 0;
    const int* vtype = &varType[0];
    const int* cofs = !catOfs.empty() ? &catOfs[0] : 0;
    const int* cmap = !catMap.empty() ? &catMap[0] : 0;
    const float* psample = sample.ptr<float>();
    const double* tw = !treeWeights.empty() ? &treeWeights[0] : 0;
    size_t sstep = sample.isContinuous() ? 1 : sample.step/sizeof(float);
    double sum = 0.;
    int lastClassIdx = -1;

    for( i = 0; i < ncats; i++ )
        catbuf[i] = -1;

    if( predictType == PREDICT_AUTO )
    {
        predictType = !isClassifier || (classLabels.size() == 2 && (flags & RAW_OUTPUT) != 0) ?
            PREDICT_SUM : PREDICT_MAX_VOTE;
    }

    if( predictType == PREDICT_MAX_VOTE )
    {
        for( i = 0; i < nclasses; i++ )
            votes[i] = 0;
    }

    for( int ridx = range.start; ridx < range.end; ridx++ )
    {
        int nidx = roots[ridx], prev = nidx, c = 0;
        const Node* node;

        while( nidx >= 0 )
        {
            prev = nidx;
            node = &nodes[nidx];
            const Split& split = splits[node->split];
            int vi = vidx ? vidx[split.varIdx] : split.varIdx;
            int ci = vtype[vi];
            float val = psample[vi*sstep];
            if( ci < 0 )
                nidx = val <= split.c ? node->left : node->right;
            else
            {
                if( flags & PREPROCESSED_INPUT )
                    c = cvRound(val);
                else
                {
                    c = catbuf[ci];
                    if( c < 0 )
                    {
                        int a = c = cofs[ci*2];
                        int b = cofs[ci*2+1];

                        int ival = cvRound(val);
                        if( ival != val )
                            CV_Error( CV_StsBadArg,
                                     "one of input categorical variable is not an integer" );

                        while( a < b )
                        {
                            c = (a + b) >> 1;
                            if( ival < cmap[c] )
                                b = c;
                            else if( ival > cmap[c] )
                                a = c+1;
                            else
                                break;
                        }

                        if( c < 0 || ival != cmap[c] )
                            continue;

                        catbuf[ci] = c -= cofs[ci*2];
                    }
                    const int* subset = &subsets[split.subsetOfs];
                    unsigned u = c;
                    nidx = CV_DTREE_CAT_DIR(u, subset) < 0 ? node->left : node->right;
                }
            }
        }

        double wval = tw ? tw[ridx] : 1.;
        if( predictType == PREDICT_SUM )
            sum += wval*nodes[prev].value;
        else
        {
            lastClassIdx = nodes[prev].classIdx;
            votes[lastClassIdx] += wval;
        }
    }

    if( predictType == PREDICT_MAX_VOTE )
    {
        int best_idx = lastClassIdx;
        if( range.end - range.start > 1 )
        {
            best_idx = 0;
            for( i = 1; i < nclasses; i++ )
                if( votes[best_idx] < votes[i] )
                    best_idx = i;
        }
        sum = (flags & RAW_OUTPUT) ? (float)best_idx : classLabels[best_idx];
    }

    return (float)sum;
}

void DTreesImpl::writeTrainingParams(FileStorage& fs) const
{
    fs << "use_surrogates" << (params0.useSurrogates ? 1 : 0);

    if( isClassifier )
    {
        fs << "max_categories" << params0.maxCategories;
    }
    else
    {
        fs << "regression_accuracy" << params0.regressionAccuracy;
    }

    fs << "max_depth" << params0.maxDepth;
    fs << "min_sample_count" << params0.minSampleCount;
    fs << "cross_validation_folds" << params0.CVFolds;

    if( params0.CVFolds > 1 )
        fs << "use_1se_rule" << (params0.use1SERule ? 1 : 0);

    if( !params0.priors.empty() )
        fs << "priors" << params0.priors;
}

void DTreesImpl::writeParams(FileStorage& fs) const
{
    fs << "is_classifier" << isClassifier;
    fs << "var_all" << varType.size();
    fs << "var_count" << getVarCount();

    int ord_var_count = 0, cat_var_count = 0;
    int i, n = (int)varType.size();
    for( i = 0; i < n; i++ )
        if( varType[i] < 0 )
            ord_var_count++;
        else
            cat_var_count++;
    fs << "ord_var_count" << ord_var_count;
    fs << "cat_var_count" << cat_var_count;

    fs << "training_params" << "{";
    writeTrainingParams(fs);

    fs << "}";

    if( !varIdx.empty() )
        fs << "var_idx" << varIdx;

    fs << "var_type" << "[:";

    n = (int)varType.size();
    for( i = 0; i < n; i++ )
        fs << (int)(varType[i] >= 0);

    fs << "]";

    if( !catOfs.empty() )
        fs << "cat_ofs" << catOfs;
    if( !catMap.empty() )
        fs << "cat_map" << catMap;
}

void DTreesImpl::writeSplit( FileStorage& fs, int splitidx ) const
{
    const Split& split = splits[splitidx];

    fs << "{:";

    fs << "var" << split.varIdx;
    fs << "quality" << split.quality;

    int ci = varType[split.varIdx];
    if( ci >= 0 ) // split on a categorical var
    {
        int i, n = catCount[ci], to_right = 0;
        const int* subset = &subsets[split.subsetOfs];
        for( i = 0; i < n; i++ )
            to_right += CV_DTREE_CAT_DIR(i, subset) > 0;

        // ad-hoc rule when to use inverse categorical split notation
        // to achieve more compact and clear representation
        int default_dir = to_right <= 1 || to_right <= std::min(3, n/2) || to_right <= n/3 ? -1 : 1;

        fs << (default_dir*(split.inversed ? -1 : 1) > 0 ? "in" : "not_in") << "[:";

        for( i = 0; i < n; i++ )
        {
            int dir = CV_DTREE_CAT_DIR(i, subset);
            if( dir*default_dir < 0 )
                fs << i;
        }

        fs << "]";
    }
    else
        fs << (!split.inversed ? "le" : "gt") << split.c;

    fs << "}";
}

void DTreesImpl::writeNode( FileStorage& fs, int nidx, int depth ) const
{
    const Node& node = nodes[nidx];
    fs << "{";
    fs << "depth" << depth;
    fs << "value" << node.value;

    if( isClassifier )
        fs << "norm_class_idx" << node.classIdx;

    if( node.split >= 0 )
    {
        fs << "splits" << "[";

        for( int splitidx = node.split; splitidx >= 0; splitidx = splits[splitidx].next )
            writeSplit( fs, splitidx );

        fs << "]";
    }

    fs << "}";
}

void DTreesImpl::writeTree( FileStorage& fs, int root ) const
{
    fs << "nodes" << "[";

    int nidx = root, pidx = 0, depth = 0;
    const Node *node = 0;

    // traverse the tree and save all the nodes in depth-first order
    for(;;)
    {
        for(;;)
        {
            writeNode( fs, nidx, depth );
            node = &nodes[nidx];
            if( node->left < 0 )
                break;
            nidx = node->left;
            depth++;
        }

        for( pidx = node->parent; pidx >= 0 && nodes[pidx].right == nidx;
             nidx = pidx, pidx = nodes[pidx].parent )
            depth--;

        if( pidx < 0 )
            break;

        nidx = nodes[pidx].right;
    }

    fs << "]";
}

void DTreesImpl::write( FileStorage& fs ) const
{
    writeParams(fs);
    writeTree(fs, roots[0]);
}

void DTreesImpl::readParams( const FileNode& fn )
{
    isClassifier = (int)fn["is_classifier"];
    /*int var_all = (int)fn["var_all"];
    int var_count = (int)fn["var_count"];
    int cat_var_count = (int)fn["cat_var_count"];
    int ord_var_count = (int)fn["ord_var_count"];*/

    FileNode tparams_node = fn["training_params"];

    params0 = Params();

    if( !tparams_node.empty() ) // training parameters are not necessary
    {
        params0.useSurrogates = (int)tparams_node["use_surrogates"] != 0;

        if( isClassifier )
        {
            params0.maxCategories = (int)tparams_node["max_categories"];
        }
        else
        {
            params0.regressionAccuracy = (float)tparams_node["regression_accuracy"];
        }

        params0.maxDepth = (int)tparams_node["max_depth"];
        params0.minSampleCount = (int)tparams_node["min_sample_count"];
        params0.CVFolds = (int)tparams_node["cross_validation_folds"];

        if( params0.CVFolds > 1 )
        {
            params.use1SERule = (int)tparams_node["use_1se_rule"] != 0;
        }

        tparams_node["priors"] >> params0.priors;
    }

    fn["var_idx"] >> varIdx;
    fn["var_type"] >> varType;
    // TODO: init var type

    fn["cat_ofs"] >> catOfs;
    fn["cat_map"] >> catMap;

    setParams(params0);
}

int DTreesImpl::readSplit( const FileNode& fn )
{
    Split split;

    int vi = (int)fn["var"];
    CV_Assert( 0 <= vi && vi <= (int)varType.size() );

    int ci = varType[vi];
    if( ci >= 0 ) // split on categorical var
    {
        int i, val, nb = (catCount[ci] + 31)/32;
        split.subsetOfs = (int)subsets.size();
        for( i = 0; i < nb; i++ )
            subsets.push_back(0);
        int* subset = &subsets[split.subsetOfs];
        FileNode fns = fn["in"];
        if( fns.empty() )
        {
            fns = fn["not_in"];
            split.inversed = true;
        }

        if( fns.isInt() )
        {
            val = (int)fns;
            subset[val >> 5] |= 1 << (val & 31);
        }
        else
        {
            FileNodeIterator it = fns.begin();
            int n = (int)fns.size();
            for( i = 0; i < n; i++, ++it )
            {
                val = (int)*it;
                subset[val >> 5] |= 1 << (val & 31);
            }
        }

        // for categorical splits we do not use inversed splits,
        // instead we inverse the variable set in the split
        if( split.inversed )
            for( i = 0; i < nb; i++ )
                subset[i] ^= -1;
    }
    else
    {
        FileNode cmpNode = fn["le"];
        if( !cmpNode.empty() )
        {
            cmpNode = fn["gt"];
            split.inversed = true;
        }
        split.c = (float)cmpNode;
    }
    
    split.quality = (float)fn["quality"];
    splits.push_back(split);

    return (int)(splits.size() - 1);
}

int DTreesImpl::readNode( const FileNode& fn )
{
    Node node;
    node.value = (double)fn["value"];

    if( isClassifier )
        node.classIdx = (int)fn["norm_class_idx"];

    FileNode sfn = fn["splits"];
    if( !sfn.empty() )
    {
        int i, n = (int)sfn.size(), prevsplit = -1;
        FileNodeIterator it = sfn.begin();

        for( i = 0; i < n; i++, ++it )
        {
            int splitidx = readSplit(*it);
            if( splitidx < 0 )
                break;
            if( prevsplit < 0 )
                node.split = splitidx;
            else
                splits[prevsplit].next = splitidx;
            prevsplit = splitidx;
        }
    }
    nodes.push_back(node);
    return (int)(nodes.size() - 1);
}

int DTreesImpl::readTree( const FileNode& fn )
{
    int i, n = (int)fn.size(), root = -1, pidx = -1;
    FileNodeIterator it = fn.begin();

    for( i = 0; i < n; i++, ++it )
    {
        int nidx = readNode(*it);
        if( nidx < 0 )
            break;
        Node& node = nodes[nidx];
        node.parent = pidx;
        if( pidx < 0 )
            root = nidx;
        else
        {
            Node& parent = nodes[pidx];
            if( parent.left < 0 )
                parent.left = nidx;
            else
                parent.right = nidx;
        }
        if( node.split >= 0 )
            pidx = nidx;
        else
        {
            while( pidx >= 0 && nodes[pidx].right >= 0 )
                pidx = nodes[pidx].parent;
        }
    }
    return root;
}

void DTreesImpl::read( const FileNode& fn )
{
    clear();
    readParams(fn);

    FileNode fnodes = fn["nodes"];
    CV_Assert( !fnodes.empty() );
    int root = readTree(fnodes);
    roots.push_back(root);
}

Mat DTreesImpl::getVarImportance() { return Mat(); }

Ptr<DTrees> createDTree(const Ptr<TrainData>& trainData, const DTrees::Params& params)
{
    Ptr<DTreesImpl> p = makePtr<DTreesImpl>();
    p->setParams(params);
    if( !p->train(trainData, 0) )
        p.release();
    return p;
}
    
}
}

/* End of file. */

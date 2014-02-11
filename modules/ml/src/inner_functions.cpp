/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
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
//   * The name of Intel Corporation may not be used to endorse or promote products
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

StatModel::~StatModel() {}
void StatModel::clear() {}

int StatModel::getVarCount() const { return 0; }
int StatModel::getSampleCount() const { return 0; }

bool StatModel::isTrained() const { return false; }
bool StatModel::isRegression() const { return false; }

bool StatModel::train( InputArray, bool, InputArray,
           InputArray, InputArray, InputArray, InputArray, bool )
{
    CV_Error(CV_StsNotImplemented, "");
    return false;
}

float StatModel::calcError( const Ptr<TrainData>&, bool, OutputArray ) const
{
    return FLT_MAX;
}

String StatModel::defaultModelName() const { return ""; }

/* Calculates upper triangular matrix S, where A is a symmetrical matrix A=S'*S */
static void Cholesky( const Mat& A, Mat& S )
{
    CV_Assert(A.type() == CV_32F);

    int dim = A.rows;
    S.create(dim, dim, CV_32F);

    int i, j, k;

    for( i = 0; i < dim; i++ )
    {
        for( j = 0; j < i; j++ )
            S.at<float>(i,j) = 0.f;

        float sum = 0.f;
        for( k = 0; k < i; k++ )
        {
            float val = S.at<float>(k,i);
            sum += val*val;
        }

        S.at<float>(i,i) = std::sqrt(std::max(A.at<float>(i,i) - sum, 0.f));
        float ival = 1.f/S.at<float>(i, i);

        for( j = i + 1; j < dim; j++ )
        {
            sum = 0;
            for( k = 0; k < i; k++ )
                sum += S.at<float>(k, i) * S.at<float>(k, j);

            S.at<float>(i, j) = (A.at<float>(i, j) - sum)*ival;
        }
    }
}

/* Generates <sample> from multivariate normal distribution, where <mean> - is an
   average row vector, <cov> - symmetric covariation matrix */
void randMVNormal( CvMat* mean, CvMat* cov, CvMat* sample, CvRNG* rng )
{
    int dim = sample->cols;
    int amount = sample->rows;

    CvRNG state = rng ? *rng : cvRNG( cvGetTickCount() );
    cvRandArr(&state, sample, CV_RAND_NORMAL, cvScalarAll(0), cvScalarAll(1) );

    CvMat* utmat = cvCreateMat(dim, dim, sample->type);
    CvMat* vect = cvCreateMatHeader(1, dim, sample->type);

    cvChol(cov, utmat);

    int i;
    for( i = 0; i < amount; i++ )
    {
        cvGetRow(sample, vect, i);
        cvMatMulAdd(vect, utmat, mean, vect);
    }

    cvReleaseMat(&vect);
    cvReleaseMat(&utmat);
}


/* Generates <sample> of <amount> points from a discrete variate xi,
   where Pr{xi = k} == probs[k], 0 < k < len - 1. */
static void cvRandSeries( float probs[], int len, int sample[], int amount )
{
    CvMat* univals = cvCreateMat(1, amount, CV_32FC1);
    float* knots = (float*)cvAlloc( len * sizeof(float) );

    int i, j;

    CvRNG state = cvRNG(-1);
    cvRandArr(&state, univals, CV_RAND_UNI, cvScalarAll(0), cvScalarAll(1) );

    knots[0] = probs[0];
    for( i = 1; i < len; i++ )
        knots[i] = knots[i - 1] + probs[i];

    for( i = 0; i < amount; i++ )
        for( j = 0; j < len; j++ )
        {
            if ( CV_MAT_ELEM(*univals, float, 0, i) <= knots[j] )
            {
                sample[i] = j;
                break;
            }
        }

    cvFree(&knots);
}

/* Generates <sample> from gaussian mixture distribution */
CV_IMPL void cvRandGaussMixture( CvMat* means[],
                                 CvMat* covs[],
                                 float weights[],
                                 int clsnum,
                                 CvMat* sample,
                                 CvMat* sampClasses )
{
    int dim = sample->cols;
    int amount = sample->rows;

    int i, clss;

    int* sample_clsnum = (int*)cvAlloc( amount * sizeof(int) );
    CvMat** utmats = (CvMat**)cvAlloc( clsnum * sizeof(CvMat*) );
    CvMat* vect = cvCreateMatHeader(1, dim, CV_32FC1);

    CvMat* classes;
    if( sampClasses )
        classes = sampClasses;
    else
        classes = cvCreateMat(1, amount, CV_32FC1);

    CvRNG state = cvRNG(-1);
    cvRandArr(&state, sample, CV_RAND_NORMAL, cvScalarAll(0), cvScalarAll(1));

    cvRandSeries(weights, clsnum, sample_clsnum, amount);

    for( i = 0; i < clsnum; i++ )
    {
        utmats[i] = cvCreateMat(dim, dim, CV_32FC1);
        cvChol(covs[i], utmats[i]);
    }

    for( i = 0; i < amount; i++ )
    {
        CV_MAT_ELEM(*classes, float, 0, i) = (float)sample_clsnum[i];
        cvGetRow(sample, vect, i);
        clss = sample_clsnum[i];
        cvMatMulAdd(vect, utmats[clss], means[clss], vect);
    }

    if( !sampClasses )
        cvReleaseMat(&classes);
    for( i = 0; i < clsnum; i++ )
        cvReleaseMat(&utmats[i]);
    cvFree(&utmats);
    cvFree(&sample_clsnum);
    cvReleaseMat(&vect);
}


CvMat* icvGenerateRandomClusterCenters ( int seed, const CvMat* data,
                                         int num_of_clusters, CvMat* _centers )
{
    CvMat* centers = _centers;

    CvRNG rng;
    CvMat data_comp, centers_comp;
    CvPoint minLoc, maxLoc; // Not used, just for function "cvMinMaxLoc"
    double minVal, maxVal;
    int i;
    int dim = data ? data->cols : 0;

    if( ICV_IS_MAT_OF_TYPE(data, CV_32FC1) )
    {
        if( _centers && !ICV_IS_MAT_OF_TYPE (_centers, CV_32FC1) )
        {
            CV_ERROR(CV_StsBadArg,"");
        }
        else if( !_centers )
            CV_CALL(centers = cvCreateMat (num_of_clusters, dim, CV_32FC1));
    }
    else if( ICV_IS_MAT_OF_TYPE(data, CV_64FC1) )
    {
        if( _centers && !ICV_IS_MAT_OF_TYPE (_centers, CV_64FC1) )
        {
            CV_ERROR(CV_StsBadArg,"");
        }
        else if( !_centers )
            CV_CALL(centers = cvCreateMat (num_of_clusters, dim, CV_64FC1));
    }
    else
        CV_ERROR (CV_StsBadArg,"");

    if( num_of_clusters < 1 )
        CV_ERROR (CV_StsBadArg,"");

    rng = cvRNG(seed);
    for (i = 0; i < dim; i++)
    {
        CV_CALL(cvGetCol (data, &data_comp, i));
        CV_CALL(cvMinMaxLoc (&data_comp, &minVal, &maxVal, &minLoc, &maxLoc));
        CV_CALL(cvGetCol (centers, &centers_comp, i));
        CV_CALL(cvRandArr (&rng, &centers_comp, CV_RAND_UNI, cvScalarAll(minVal), cvScalarAll(maxVal)));
    }

    //__END__;

    if( (cvGetErrStatus () < 0) || (centers != _centers) )
        cvReleaseMat (&centers);

    return _centers ? _centers : centers;
} // end of icvGenerateRandomClusterCenters


const float**
cvGetTrainSamples( const CvMat* train_data, int tflag,
                   const CvMat* var_idx, const CvMat* sample_idx,
                   int* _var_count, int* _sample_count,
                   bool always_copy_data )
{
    float** samples = 0;

    int i, j, var_count, sample_count, s_step, v_step;
    bool copy_data;
    const float* data;
    const int *s_idx, *v_idx;

    if( !CV_IS_MAT(train_data) )
        CV_ERROR( CV_StsBadArg, "Invalid or NULL training data matrix" );

    var_count = var_idx ? var_idx->cols + var_idx->rows - 1 :
                tflag == CV_ROW_SAMPLE ? train_data->cols : train_data->rows;
    sample_count = sample_idx ? sample_idx->cols + sample_idx->rows - 1 :
                   tflag == CV_ROW_SAMPLE ? train_data->rows : train_data->cols;

    if( _var_count )
        *_var_count = var_count;

    if( _sample_count )
        *_sample_count = sample_count;

    copy_data = tflag != CV_ROW_SAMPLE || var_idx || always_copy_data;

    CV_CALL( samples = (float**)cvAlloc(sample_count*sizeof(samples[0]) +
                (copy_data ? 1 : 0)*var_count*sample_count*sizeof(samples[0][0])) );
    data = train_data->data.fl;
    s_step = train_data->step / sizeof(samples[0][0]);
    v_step = 1;
    s_idx = sample_idx ? sample_idx->data.i : 0;
    v_idx = var_idx ? var_idx->data.i : 0;

    if( !copy_data )
    {
        for( i = 0; i < sample_count; i++ )
            samples[i] = (float*)(data + (s_idx ? s_idx[i] : i)*s_step);
    }
    else
    {
        samples[0] = (float*)(samples + sample_count);
        if( tflag != CV_ROW_SAMPLE )
            CV_SWAP( s_step, v_step, i );

        for( i = 0; i < sample_count; i++ )
        {
            float* dst = samples[i] = samples[0] + i*var_count;
            const float* src = data + (s_idx ? s_idx[i] : i)*s_step;

            if( !v_idx )
                for( j = 0; j < var_count; j++ )
                    dst[j] = src[j*v_step];
            else
                for( j = 0; j < var_count; j++ )
                    dst[j] = src[v_idx[j]*v_step];
        }
    }

    return (const float**)samples;
}

/* End of file */

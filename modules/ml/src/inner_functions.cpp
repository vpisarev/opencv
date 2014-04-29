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

bool StatModel::train( const Ptr<TrainData>&, int )
{
    CV_Error(CV_StsNotImplemented, "");
    return false;
}

float StatModel::calcError( const Ptr<TrainData>&, bool, OutputArray ) const
{
    return FLT_MAX;
}

String StatModel::defaultModelName() const { return "statmodel"; }

void StatModel::save(const String& filename) const
{
    FileStorage fs(filename, FileStorage::WRITE);
    fs << defaultModelName() << "{";
    write(fs);
    fs << "}";
}

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
void randMVNormal( InputArray _mean, InputArray _cov, int nsamples, OutputArray _samples )
{
    Mat mean = _mean.getMat(), cov = _cov.getMat();
    int dim = (int)mean.total();

    _samples.create(nsamples, dim, CV_32F);
    Mat samples = _samples.getMat();
    randu(samples, 0., 1.);

    Mat utmat;
    Cholesky(cov, utmat);
    int flags = mean.cols == 1 ? 0 : GEMM_3_T;

    for( int i = 0; i < nsamples; i++ )
    {
        Mat sample = samples.row(i);
        gemm(sample, utmat, 1, mean, 1, sample, flags);
    }
}

}}

/* End of file */

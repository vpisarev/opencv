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

#include <stdarg.h>
#include <ctype.h>

/****************************************************************************************\
                                COPYRIGHT NOTICE
                                ----------------

  The code has been derived from libsvm library (version 2.6)
  (http://www.csie.ntu.edu.tw/~cjlin/libsvm).

  Here is the orignal copyright:
------------------------------------------------------------------------------------------
    Copyright (c) 2000-2003 Chih-Chung Chang and Chih-Jen Lin
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions
    are met:

    1. Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.

    3. Neither name of copyright holders nor the names of its contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.


    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
    A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR
    CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
    EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
    PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
    LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
    NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
\****************************************************************************************/

namespace cv { namespace ml {

#define CV_SVM_MIN_CACHE_SIZE  (40 << 20)  /* 40Mb */

typedef float Qfloat;
#define QFLOAT_TYPE CV_32F

// Param Grid
static void checkParamGrid(const ParamGrid& pg)
{
    if( pg.minVal > pg.maxVal )
        CV_Error( CV_StsBadArg, "Lower bound of the grid must be less then the upper one" );
    if( pg.minVal < DBL_EPSILON )
        CV_Error( CV_StsBadArg, "Lower bound of the grid must be positive" );
    if( pg.logStep < 1. + FLT_EPSILON )
        CV_Error( CV_StsBadArg, "Grid step must greater then 1" );
}

// SVM training parameters
SVM::Params::Params()
{
    svmType = SVM::C_SVC;
    kernelType = SVM::RBF;
    degree = 0;
    gamma = 1;
    coef0 = 0;
    C = 1;
    nu = 0;
    p = 0;
    termCrit = TermCriteria( CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 1000, FLT_EPSILON );
}


SVM::Params::Params( int _svmType, int _kernelType,
                     double _degree, double _gamma, double _coef0,
                     double _Con, double _nu, double _p,
                     const Mat& _classWeights, TermCriteria _termCrit )
{
    svmType = _svmType;
    kernelType = _kernelType;
    degree = _degree;
    gamma = _gamma;
    coef0 = _coef0;
    C = _Con;
    nu = _nu;
    p = _p;
    classWeights = _classWeights;
    termCrit = _termCrit;
}


/////////////////////////////////////// SVM kernel ///////////////////////////////////////
class SVMKernelImpl : public SVM::Kernel
{
public:
    SVMKernelImpl()
    {
    }

    SVMKernelImpl( const SVM::Params& _params )
    {
        params = _params;
    }

    virtual ~SVMKernelImpl()
    {
    }

    void calc_non_rbf_base( int vcount, int var_count, const float** vecs,
                            const float* another, Qfloat* results,
                            double alpha, double beta )
    {
        int j, k;
        for( j = 0; j < vcount; j++ )
        {
            const float* sample = vecs[j];
            double s = 0;
            for( k = 0; k <= var_count - 4; k += 4 )
                s += sample[k]*another[k] + sample[k+1]*another[k+1] +
                sample[k+2]*another[k+2] + sample[k+3]*another[k+3];
            for( ; k < var_count; k++ )
                s += sample[k]*another[k];
            results[j] = (Qfloat)(s*alpha + beta);
        }
    }

    void calc_linear( int vcount, int var_count, const float** vecs,
                      const float* another, Qfloat* results )
    {
        calc_non_rbf_base( vcount, var_count, vecs, another, results, 1, 0 );
    }

    void calc_poly( int vcount, int var_count, const float** vecs,
                    const float* another, Qfloat* results )
    {
        Mat R( 1, vcount, QFLOAT_TYPE, results );
        calc_non_rbf_base( vcount, var_count, vecs, another, results, params.gamma, params.coef0 );
        if( vcount > 0 )
            cv::pow( R, params.degree, R );
    }

    void calc_sigmoid( int vcount, int var_count, const float** vecs,
                       const float* another, Qfloat* results )
    {
        int j;
        calc_non_rbf_base( vcount, var_count, vecs, another, results,
                          -2*params.gamma, -2*params.coef0 );
        // TODO: speedup this
        for( j = 0; j < vcount; j++ )
        {
            Qfloat t = results[j];
            double e = exp(-fabs(t));
            if( t > 0 )
                results[j] = (Qfloat)((1. - e)/(1. + e));
            else
                results[j] = (Qfloat)((e - 1.)/(e + 1.));
        }
    }


    void calc_rbf( int vcount, int var_count, const float** vecs,
                   const float* another, Qfloat* results )
    {
        Mat R( 1, vcount, QFLOAT_TYPE, results );
        double gamma = -params.gamma;
        int j, k;

        for( j = 0; j < vcount; j++ )
        {
            const float* sample = vecs[j];
            double s = 0;

            for( k = 0; k <= var_count - 4; k += 4 )
            {
                double t0 = sample[k] - another[k];
                double t1 = sample[k+1] - another[k+1];

                s += t0*t0 + t1*t1;

                t0 = sample[k+2] - another[k+2];
                t1 = sample[k+3] - another[k+3];

                s += t0*t0 + t1*t1;
            }

            for( ; k < var_count; k++ )
            {
                double t0 = sample[k] - another[k];
                s += t0*t0;
            }
            results[j] = (Qfloat)(s*gamma);
        }

        if( vcount > 0 )
            cvExp( &R, &R );
    }

    /// Histogram intersection kernel
    void calc_intersec( int vcount, int var_count, const float** vecs,
                        const float* another, Qfloat* results )
    {
        int j, k;
        for( j = 0; j < vcount; j++ )
        {
            const float* sample = vecs[j];
            double s = 0;
            for( k = 0; k <= var_count - 4; k += 4 )
                s += std::min(sample[k],another[k]) + std::min(sample[k+1],another[k+1]) +
                std::min(sample[k+2],another[k+2]) + std::min(sample[k+3],another[k+3]);
            for( ; k < var_count; k++ )
                s += std::min(sample[k],another[k]);
            results[j] = (Qfloat)(s);
        }
    }

    /// Exponential chi2 kernel
    void calc_chi2( int vcount, int var_count, const float** vecs,
                    const float* another, Qfloat* results )
    {
        Mat R( 1, vcount, QFLOAT_TYPE, results );
        double gamma = -params.gamma;
        int j, k;
        for( j = 0; j < vcount; j++ )
        {
            const float* sample = vecs[j];
            double chi2 = 0;
            for(k = 0 ; k < var_count; k++ )
            {
                double d = sample[k]-another[k];
                double devisor = sample[k]+another[k];
                /// if devisor == 0, the Chi2 distance would be zero,
                // but calculation would rise an error because of deviding by zero
                if (devisor != 0)
                {
                    chi2 += d*d/devisor;
                }
            }
            results[j] = (Qfloat) (gamma*chi2);
        }
        if( vcount > 0 )
            exp( R, R );
    }
    
    void calc( int vcount, int var_count, const float** vecs,
               const float* another, Qfloat* results )
    {
        switch( params.kernelType )
        {
        case SVM::RBF:
            calc_rbf(vcount, var_count, vecs, another, results);
            break;
        case SVM::POLY:
            calc_poly(vcount, var_count, vecs, another, results);
            break;
        case SVM::SIGMOID:
            calc_sigmoid(vcount, var_count, vecs, another, results);
            break;
        case SVM::CHI2:
            calc_chi2(vcount, var_count, vecs, another, results);
            break;
        case SVM::INTER:
            calc_intersec(vcount, var_count, vecs, another, results);
            break;
        default:
            CV_Error(CV_StsBadArg, "Unknown kernel type");
        }
        const Qfloat max_val = (Qfloat)(FLT_MAX*1e-3);
        for( int j = 0; j < vcount; j++ )
        {
            if( results[j] > max_val )
                results[j] = max_val;
        }
    }

    SVM::Params params;
};


class SVMImpl : public SVM
{
    virtual ~SVMImpl() {}

    virtual ParamGrid getDefaultGrid( int param_id ) const
    {
        ParamGrid grid;
        if( param_id == SVM::C )
        {
            grid.minVal = 0.1;
            grid.maxVal = 500;
            grid.logStep = 5; // total iterations = 5
        }
        else if( param_id == SVM::GAMMA )
        {
            grid.minVal = 1e-5;
            grid.maxVal = 0.6;
            grid.logStep = 15; // total iterations = 4
        }
        else if( param_id == SVM::P )
        {
            grid.minVal = 0.01;
            grid.maxVal = 100;
            grid.logStep = 7; // total iterations = 4
        }
        else if( param_id == SVM::NU )
        {
            grid.minVal = 0.01;
            grid.maxVal = 0.2;
            grid.logStep = 3; // total iterations = 3
        }
        else if( param_id == SVM::COEF )
        {
            grid.minVal = 0.1;
            grid.maxVal = 300;
            grid.logStep = 14; // total iterations = 3
        }
        else if( param_id == SVM::DEGREE )
        {
            grid.minVal = 0.01;
            grid.maxVal = 4;
            grid.logStep = 7; // total iterations = 3
        }
        else
            cvError( CV_StsBadArg, "SVM::getDefaultGrid", "Invalid type of parameter "
                     "(use one of SVM::C, SVM::GAMMA et al.)", __FILE__, __LINE__ );
        return grid;
    }

};


// Generalized SMO+SVMlight algorithm
// Solves:
//
//  min [0.5(\alpha^T Q \alpha) + b^T \alpha]
//
//      y^T \alpha = \delta
//      y_i = +1 or -1
//      0 <= alpha_i <= Cp for y_i = 1
//      0 <= alpha_i <= Cn for y_i = -1
//
// Given:
//
//  Q, b, y, Cp, Cn, and an initial feasible point \alpha
//  l is the size of vectors and matrices
//  eps is the stopping criterion
//
// solution will be put in \alpha, objective value will be put in obj
//

void CvSVMSolver::clear()
{
    G = 0;
    alpha = 0;
    y = 0;
    b = 0;
    buf[0] = buf[1] = 0;
    cvReleaseMemStorage( &storage );
    kernel = 0;
    select_working_set_func = 0;
    calc_rho_func = 0;

    rows = 0;
    samples = 0;
    get_row_func = 0;
}


CvSVMSolver::CvSVMSolver()
{
    storage = 0;
    clear();
}


CvSVMSolver::~CvSVMSolver()
{
    clear();
}


CvSVMSolver::CvSVMSolver( int _sample_count, int _var_count, const float** _samples, schar* _y,
                int _alpha_count, double* _alpha, double _Cp, double _Cn,
                CvMemStorage* _storage, CvSVMKernel* _kernel, GetRow _get_row,
                SelectWorkingSet _select_working_set, CalcRho _calc_rho )
{
    storage = 0;
    create( _sample_count, _var_count, _samples, _y, _alpha_count, _alpha, _Cp, _Cn,
            _storage, _kernel, _get_row, _select_working_set, _calc_rho );
}


bool CvSVMSolver::create( int _sample_count, int _var_count, const float** _samples, schar* _y,
                int _alpha_count, double* _alpha, double _Cp, double _Cn,
                CvMemStorage* _storage, CvSVMKernel* _kernel, GetRow _get_row,
                SelectWorkingSet _select_working_set, CalcRho _calc_rho )
{
    bool ok = false;
    int i, svmType;

    CV_FUNCNAME( "CvSVMSolver::create" );

    __BEGIN__;

    int rows_hdr_size;

    clear();

    sample_count = _sample_count;
    var_count = _var_count;
    samples = _samples;
    y = _y;
    alpha_count = _alpha_count;
    alpha = _alpha;
    kernel = _kernel;

    C[0] = _Cn;
    C[1] = _Cp;
    eps = kernel->params->termCrit.epsilon;
    max_iter = kernel->params->termCrit.max_iter;
    storage = cvCreateChildMemStorage( _storage );

    b = (double*)cvMemStorageAlloc( storage, alpha_count*sizeof(b[0]));
    alpha_status = (schar*)cvMemStorageAlloc( storage, alpha_count*sizeof(alpha_status[0]));
    G = (double*)cvMemStorageAlloc( storage, alpha_count*sizeof(G[0]));
    for( i = 0; i < 2; i++ )
        buf[i] = (Qfloat*)cvMemStorageAlloc( storage, sample_count*2*sizeof(buf[i][0]) );
    svmType = kernel->params->svmType;

    select_working_set_func = _select_working_set;
    if( !select_working_set_func )
        select_working_set_func = svmType == CvSVM::NU_SVC || svmType == CvSVM::NU_SVR ?
        &CvSVMSolver::select_working_set_nu_svm : &CvSVMSolver::select_working_set;

    calc_rho_func = _calc_rho;
    if( !calc_rho_func )
        calc_rho_func = svmType == CvSVM::NU_SVC || svmType == CvSVM::NU_SVR ?
            &CvSVMSolver::calc_rho_nu_svm : &CvSVMSolver::calc_rho;

    get_row_func = _get_row;
    if( !get_row_func )
        get_row_func = params->svmType == CvSVM::EPS_SVR ||
                       params->svmType == CvSVM::NU_SVR ? &CvSVMSolver::get_row_svr :
                       params->svmType == CvSVM::C_SVC ||
                       params->svmType == CvSVM::NU_SVC ? &CvSVMSolver::get_row_svc :
                       &CvSVMSolver::get_row_one_class;

    cache_line_size = sample_count*sizeof(Qfloat);
    // cache size = max(num_of_samples^2*sizeof(Qfloat)*0.25, 64Kb)
    // (assuming that for large training sets ~25% of Q matrix is used)
    cache_size = MAX( cache_line_size*sample_count/4, CV_SVM_MIN_CACHE_SIZE );

    // the size of Q matrix row headers
    rows_hdr_size = sample_count*sizeof(rows[0]);
    if( rows_hdr_size > storage->block_size )
        CV_ERROR( CV_StsOutOfRange, "Too small storage block size" );

    lru_list.prev = lru_list.next = &lru_list;
    rows = (CvSVMKernelRow*)cvMemStorageAlloc( storage, rows_hdr_size );
    memset( rows, 0, rows_hdr_size );

    ok = true;

    __END__;

    return ok;
}


float* CvSVMSolver::get_row_base( int i, bool* _existed )
{
    int i1 = i < sample_count ? i : i - sample_count;
    CvSVMKernelRow* row = rows + i1;
    bool existed = row->data != 0;
    Qfloat* data;

    if( existed || cache_size <= 0 )
    {
        CvSVMKernelRow* del_row = existed ? row : lru_list.prev;
        data = del_row->data;
        assert( data != 0 );

        // delete row from the LRU list
        del_row->data = 0;
        del_row->prev->next = del_row->next;
        del_row->next->prev = del_row->prev;
    }
    else
    {
        data = (Qfloat*)cvMemStorageAlloc( storage, cache_line_size );
        cache_size -= cache_line_size;
    }

    // insert row into the LRU list
    row->data = data;
    row->prev = &lru_list;
    row->next = lru_list.next;
    row->prev->next = row->next->prev = row;

    if( !existed )
    {
        kernel->calc( sample_count, var_count, samples, samples[i1], row->data );
    }

    if( _existed )
        *_existed = existed;

    return row->data;
}


float* CvSVMSolver::get_row_svc( int i, float* row, float*, bool existed )
{
    if( !existed )
    {
        const schar* _y = y;
        int j, len = sample_count;
        assert( _y && i < sample_count );

        if( _y[i] > 0 )
        {
            for( j = 0; j < len; j++ )
                row[j] = _y[j]*row[j];
        }
        else
        {
            for( j = 0; j < len; j++ )
                row[j] = -_y[j]*row[j];
        }
    }
    return row;
}


float* CvSVMSolver::get_row_one_class( int, float* row, float*, bool )
{
    return row;
}


float* CvSVMSolver::get_row_svr( int i, float* row, float* dst, bool )
{
    int j, len = sample_count;
    Qfloat* dst_pos = dst;
    Qfloat* dst_neg = dst + len;
    if( i >= len )
    {
        Qfloat* temp;
        CV_SWAP( dst_pos, dst_neg, temp );
    }

    for( j = 0; j < len; j++ )
    {
        Qfloat t = row[j];
        dst_pos[j] = t;
        dst_neg[j] = -t;
    }
    return dst;
}



float* CvSVMSolver::get_row( int i, float* dst )
{
    bool existed = false;
    float* row = get_row_base( i, &existed );
    return (this->*get_row_func)( i, row, dst, existed );
}


#undef is_upper_bound
#define is_upper_bound(i) (alpha_status[i] > 0)

#undef is_lower_bound
#define is_lower_bound(i) (alpha_status[i] < 0)

#undef is_free
#define is_free(i) (alpha_status[i] == 0)

#undef get_C
#define get_C(i) (C[y[i]>0])

#undef update_alpha_status
#define update_alpha_status(i) \
    alpha_status[i] = (schar)(alpha[i] >= get_C(i) ? 1 : alpha[i] <= 0 ? -1 : 0)

#undef reconstruct_gradient
#define reconstruct_gradient() /* empty for now */


bool CvSVMSolver::solve_generic( CvSVMSolutionInfo& si )
{
    int iter = 0;
    int i, j, k;

    // 1. initialize gradient and alpha status
    for( i = 0; i < alpha_count; i++ )
    {
        update_alpha_status(i);
        G[i] = b[i];
        if( fabs(G[i]) > 1e200 )
            return false;
    }

    for( i = 0; i < alpha_count; i++ )
    {
        if( !is_lower_bound(i) )
        {
            const Qfloat *Q_i = get_row( i, buf[0] );
            double alpha_i = alpha[i];

            for( j = 0; j < alpha_count; j++ )
                G[j] += alpha_i*Q_i[j];
        }
    }

    // 2. optimization loop
    for(;;)
    {
        const Qfloat *Q_i, *Q_j;
        double C_i, C_j;
        double old_alpha_i, old_alpha_j, alpha_i, alpha_j;
        double delta_alpha_i, delta_alpha_j;

#ifdef _DEBUG
        for( i = 0; i < alpha_count; i++ )
        {
            if( fabs(G[i]) > 1e+300 )
                return false;

            if( fabs(alpha[i]) > 1e16 )
                return false;
        }
#endif

        if( (this->*select_working_set_func)( i, j ) != 0 || iter++ >= max_iter )
            break;

        Q_i = get_row( i, buf[0] );
        Q_j = get_row( j, buf[1] );

        C_i = get_C(i);
        C_j = get_C(j);

        alpha_i = old_alpha_i = alpha[i];
        alpha_j = old_alpha_j = alpha[j];

        if( y[i] != y[j] )
        {
            double denom = Q_i[i]+Q_j[j]+2*Q_i[j];
            double delta = (-G[i]-G[j])/MAX(fabs(denom),FLT_EPSILON);
            double diff = alpha_i - alpha_j;
            alpha_i += delta;
            alpha_j += delta;

            if( diff > 0 && alpha_j < 0 )
            {
                alpha_j = 0;
                alpha_i = diff;
            }
            else if( diff <= 0 && alpha_i < 0 )
            {
                alpha_i = 0;
                alpha_j = -diff;
            }

            if( diff > C_i - C_j && alpha_i > C_i )
            {
                alpha_i = C_i;
                alpha_j = C_i - diff;
            }
            else if( diff <= C_i - C_j && alpha_j > C_j )
            {
                alpha_j = C_j;
                alpha_i = C_j + diff;
            }
        }
        else
        {
            double denom = Q_i[i]+Q_j[j]-2*Q_i[j];
            double delta = (G[i]-G[j])/MAX(fabs(denom),FLT_EPSILON);
            double sum = alpha_i + alpha_j;
            alpha_i -= delta;
            alpha_j += delta;

            if( sum > C_i && alpha_i > C_i )
            {
                alpha_i = C_i;
                alpha_j = sum - C_i;
            }
            else if( sum <= C_i && alpha_j < 0)
            {
                alpha_j = 0;
                alpha_i = sum;
            }

            if( sum > C_j && alpha_j > C_j )
            {
                alpha_j = C_j;
                alpha_i = sum - C_j;
            }
            else if( sum <= C_j && alpha_i < 0 )
            {
                alpha_i = 0;
                alpha_j = sum;
            }
        }

        // update alpha
        alpha[i] = alpha_i;
        alpha[j] = alpha_j;
        update_alpha_status(i);
        update_alpha_status(j);

        // update G
        delta_alpha_i = alpha_i - old_alpha_i;
        delta_alpha_j = alpha_j - old_alpha_j;

        for( k = 0; k < alpha_count; k++ )
            G[k] += Q_i[k]*delta_alpha_i + Q_j[k]*delta_alpha_j;
    }

    // calculate rho
    (this->*calc_rho_func)( si.rho, si.r );

    // calculate objective value
    for( i = 0, si.obj = 0; i < alpha_count; i++ )
        si.obj += alpha[i] * (G[i] + b[i]);

    si.obj *= 0.5;

    si.upper_bound_p = C[1];
    si.upper_bound_n = C[0];

    return true;
}


// return 1 if already optimal, return 0 otherwise
bool
CvSVMSolver::select_working_set( int& out_i, int& out_j )
{
    // return i,j which maximize -grad(f)^T d , under constraint
    // if alpha_i == C, d != +1
    // if alpha_i == 0, d != -1
    double Gmax1 = -DBL_MAX;        // max { -grad(f)_i * d | y_i*d = +1 }
    int Gmax1_idx = -1;

    double Gmax2 = -DBL_MAX;        // max { -grad(f)_i * d | y_i*d = -1 }
    int Gmax2_idx = -1;

    int i;

    for( i = 0; i < alpha_count; i++ )
    {
        double t;

        if( y[i] > 0 )    // y = +1
        {
            if( !is_upper_bound(i) && (t = -G[i]) > Gmax1 )  // d = +1
            {
                Gmax1 = t;
                Gmax1_idx = i;
            }
            if( !is_lower_bound(i) && (t = G[i]) > Gmax2 )  // d = -1
            {
                Gmax2 = t;
                Gmax2_idx = i;
            }
        }
        else        // y = -1
        {
            if( !is_upper_bound(i) && (t = -G[i]) > Gmax2 )  // d = +1
            {
                Gmax2 = t;
                Gmax2_idx = i;
            }
            if( !is_lower_bound(i) && (t = G[i]) > Gmax1 )  // d = -1
            {
                Gmax1 = t;
                Gmax1_idx = i;
            }
        }
    }

    out_i = Gmax1_idx;
    out_j = Gmax2_idx;

    return Gmax1 + Gmax2 < eps;
}


void
CvSVMSolver::calc_rho( double& rho, double& r )
{
    int i, nr_free = 0;
    double ub = DBL_MAX, lb = -DBL_MAX, sum_free = 0;

    for( i = 0; i < alpha_count; i++ )
    {
        double yG = y[i]*G[i];

        if( is_lower_bound(i) )
        {
            if( y[i] > 0 )
                ub = MIN(ub,yG);
            else
                lb = MAX(lb,yG);
        }
        else if( is_upper_bound(i) )
        {
            if( y[i] < 0)
                ub = MIN(ub,yG);
            else
                lb = MAX(lb,yG);
        }
        else
        {
            ++nr_free;
            sum_free += yG;
        }
    }

    rho = nr_free > 0 ? sum_free/nr_free : (ub + lb)*0.5;
    r = 0;
}


bool
CvSVMSolver::select_working_set_nu_svm( int& out_i, int& out_j )
{
    // return i,j which maximize -grad(f)^T d , under constraint
    // if alpha_i == C, d != +1
    // if alpha_i == 0, d != -1
    double Gmax1 = -DBL_MAX;    // max { -grad(f)_i * d | y_i = +1, d = +1 }
    int Gmax1_idx = -1;

    double Gmax2 = -DBL_MAX;    // max { -grad(f)_i * d | y_i = +1, d = -1 }
    int Gmax2_idx = -1;

    double Gmax3 = -DBL_MAX;    // max { -grad(f)_i * d | y_i = -1, d = +1 }
    int Gmax3_idx = -1;

    double Gmax4 = -DBL_MAX;    // max { -grad(f)_i * d | y_i = -1, d = -1 }
    int Gmax4_idx = -1;

    int i;

    for( i = 0; i < alpha_count; i++ )
    {
        double t;

        if( y[i] > 0 )    // y == +1
        {
            if( !is_upper_bound(i) && (t = -G[i]) > Gmax1 )  // d = +1
            {
                Gmax1 = t;
                Gmax1_idx = i;
            }
            if( !is_lower_bound(i) && (t = G[i]) > Gmax2 )  // d = -1
            {
                Gmax2 = t;
                Gmax2_idx = i;
            }
        }
        else        // y == -1
        {
            if( !is_upper_bound(i) && (t = -G[i]) > Gmax3 )  // d = +1
            {
                Gmax3 = t;
                Gmax3_idx = i;
            }
            if( !is_lower_bound(i) && (t = G[i]) > Gmax4 )  // d = -1
            {
                Gmax4 = t;
                Gmax4_idx = i;
            }
        }
    }

    if( MAX(Gmax1 + Gmax2, Gmax3 + Gmax4) < eps )
        return 1;

    if( Gmax1 + Gmax2 > Gmax3 + Gmax4 )
    {
        out_i = Gmax1_idx;
        out_j = Gmax2_idx;
    }
    else
    {
        out_i = Gmax3_idx;
        out_j = Gmax4_idx;
    }
    return 0;
}


void
CvSVMSolver::calc_rho_nu_svm( double& rho, double& r )
{
    int nr_free1 = 0, nr_free2 = 0;
    double ub1 = DBL_MAX, ub2 = DBL_MAX;
    double lb1 = -DBL_MAX, lb2 = -DBL_MAX;
    double sum_free1 = 0, sum_free2 = 0;
    double r1, r2;

    int i;

    for( i = 0; i < alpha_count; i++ )
    {
        double G_i = G[i];
        if( y[i] > 0 )
        {
            if( is_lower_bound(i) )
                ub1 = MIN( ub1, G_i );
            else if( is_upper_bound(i) )
                lb1 = MAX( lb1, G_i );
            else
            {
                ++nr_free1;
                sum_free1 += G_i;
            }
        }
        else
        {
            if( is_lower_bound(i) )
                ub2 = MIN( ub2, G_i );
            else if( is_upper_bound(i) )
                lb2 = MAX( lb2, G_i );
            else
            {
                ++nr_free2;
                sum_free2 += G_i;
            }
        }
    }

    r1 = nr_free1 > 0 ? sum_free1/nr_free1 : (ub1 + lb1)*0.5;
    r2 = nr_free2 > 0 ? sum_free2/nr_free2 : (ub2 + lb2)*0.5;

    rho = (r1 - r2)*0.5;
    r = (r1 + r2)*0.5;
}


/*
///////////////////////// construct and solve various formulations ///////////////////////
*/

bool CvSVMSolver::solve_c_svc( int _sample_count, int _var_count, const float** _samples, schar* _y,
                               double _Cp, double _Cn, CvMemStorage* _storage,
                               CvSVMKernel* _kernel, double* _alpha, CvSVMSolutionInfo& _si )
{
    int i;

    if( !create( _sample_count, _var_count, _samples, _y, _sample_count,
                 _alpha, _Cp, _Cn, _storage, _kernel, &CvSVMSolver::get_row_svc,
                 &CvSVMSolver::select_working_set, &CvSVMSolver::calc_rho ))
        return false;

    for( i = 0; i < sample_count; i++ )
    {
        alpha[i] = 0;
        b[i] = -1;
    }

    if( !solve_generic( _si ))
        return false;

    for( i = 0; i < sample_count; i++ )
        alpha[i] *= y[i];

    return true;
}


bool CvSVMSolver::solve_nu_svc( int _sample_count, int _var_count, const float** _samples, schar* _y,
                                CvMemStorage* _storage, CvSVMKernel* _kernel,
                                double* _alpha, CvSVMSolutionInfo& _si )
{
    int i;
    double sum_pos, sum_neg, inv_r;

    if( !create( _sample_count, _var_count, _samples, _y, _sample_count,
                 _alpha, 1., 1., _storage, _kernel, &CvSVMSolver::get_row_svc,
                 &CvSVMSolver::select_working_set_nu_svm, &CvSVMSolver::calc_rho_nu_svm ))
        return false;

    sum_pos = kernel->params->nu * sample_count * 0.5;
    sum_neg = kernel->params->nu * sample_count * 0.5;

    for( i = 0; i < sample_count; i++ )
    {
        if( y[i] > 0 )
        {
            alpha[i] = MIN(1.0, sum_pos);
            sum_pos -= alpha[i];
        }
        else
        {
            alpha[i] = MIN(1.0, sum_neg);
            sum_neg -= alpha[i];
        }
        b[i] = 0;
    }

    if( !solve_generic( _si ))
        return false;

    inv_r = 1./_si.r;

    for( i = 0; i < sample_count; i++ )
        alpha[i] *= y[i]*inv_r;

    _si.rho *= inv_r;
    _si.obj *= (inv_r*inv_r);
    _si.upper_bound_p = inv_r;
    _si.upper_bound_n = inv_r;

    return true;
}


bool CvSVMSolver::solve_one_class( int _sample_count, int _var_count, const float** _samples,
                                   CvMemStorage* _storage, CvSVMKernel* _kernel,
                                   double* _alpha, CvSVMSolutionInfo& _si )
{
    int i, n;
    double nu = _kernel->params->nu;

    if( !create( _sample_count, _var_count, _samples, 0, _sample_count,
                 _alpha, 1., 1., _storage, _kernel, &CvSVMSolver::get_row_one_class,
                 &CvSVMSolver::select_working_set, &CvSVMSolver::calc_rho ))
        return false;

    y = (schar*)cvMemStorageAlloc( storage, sample_count*sizeof(y[0]) );
    n = cvRound( nu*sample_count );

    for( i = 0; i < sample_count; i++ )
    {
        y[i] = 1;
        b[i] = 0;
        alpha[i] = i < n ? 1 : 0;
    }

    if( n < sample_count )
        alpha[n] = nu * sample_count - n;
    else
        alpha[n-1] = nu * sample_count - (n-1);

    return solve_generic(_si);
}


bool CvSVMSolver::solve_eps_svr( int _sample_count, int _var_count, const float** _samples,
                                 const float* _y, CvMemStorage* _storage,
                                 CvSVMKernel* _kernel, double* _alpha, CvSVMSolutionInfo& _si )
{
    int i;
    double p = _kernel->params->p, kernel_param_c = _kernel->params->C;

    if( !create( _sample_count, _var_count, _samples, 0,
                 _sample_count*2, 0, kernel_param_c, kernel_param_c, _storage, _kernel, &CvSVMSolver::get_row_svr,
                 &CvSVMSolver::select_working_set, &CvSVMSolver::calc_rho ))
        return false;

    y = (schar*)cvMemStorageAlloc( storage, sample_count*2*sizeof(y[0]) );
    alpha = (double*)cvMemStorageAlloc( storage, alpha_count*sizeof(alpha[0]) );

    for( i = 0; i < sample_count; i++ )
    {
        alpha[i] = 0;
        b[i] = p - _y[i];
        y[i] = 1;

        alpha[i+sample_count] = 0;
        b[i+sample_count] = p + _y[i];
        y[i+sample_count] = -1;
    }

    if( !solve_generic( _si ))
        return false;

    for( i = 0; i < sample_count; i++ )
        _alpha[i] = alpha[i] - alpha[i+sample_count];

    return true;
}


bool CvSVMSolver::solve_nu_svr( int _sample_count, int _var_count, const float** _samples,
                                const float* _y, CvMemStorage* _storage,
                                CvSVMKernel* _kernel, double* _alpha, CvSVMSolutionInfo& _si )
{
    int i;
    double kernel_param_c = _kernel->params->C, sum;

    if( !create( _sample_count, _var_count, _samples, 0,
                 _sample_count*2, 0, 1., 1., _storage, _kernel, &CvSVMSolver::get_row_svr,
                 &CvSVMSolver::select_working_set_nu_svm, &CvSVMSolver::calc_rho_nu_svm ))
        return false;

    y = (schar*)cvMemStorageAlloc( storage, sample_count*2*sizeof(y[0]) );
    alpha = (double*)cvMemStorageAlloc( storage, alpha_count*sizeof(alpha[0]) );
    sum = kernel_param_c * _kernel->params->nu * sample_count * 0.5;

    for( i = 0; i < sample_count; i++ )
    {
        alpha[i] = alpha[i + sample_count] = MIN(sum, kernel_param_c);
        sum -= alpha[i];

        b[i] = -_y[i];
        y[i] = 1;

        b[i + sample_count] = _y[i];
        y[i + sample_count] = -1;
    }

    if( !solve_generic( _si ))
        return false;

    for( i = 0; i < sample_count; i++ )
        _alpha[i] = alpha[i] - alpha[i+sample_count];

    return true;
}


//////////////////////////////////////////////////////////////////////////////////////////

CvSVM::CvSVM()
{
    decision_func = 0;
    class_labels = 0;
    class_weights = 0;
    storage = 0;
    var_idx = 0;
    kernel = 0;
    solver = 0;
    default_model_name = "my_svm";

    clear();
}


CvSVM::~CvSVM()
{
    clear();
}


void CvSVM::clear()
{
    cvFree( &decision_func );
    cvReleaseMat( &class_labels );
    cvReleaseMat( &class_weights );
    cvReleaseMemStorage( &storage );
    cvReleaseMat( &var_idx );
    delete kernel;
    delete solver;
    kernel = 0;
    solver = 0;
    var_all = 0;
    sv = 0;
    sv_total = 0;
}


CvSVM::CvSVM( const CvMat* _train_data, const CvMat* _responses,
    const CvMat* _var_idx, const CvMat* _sample_idx, CvSVMParams _params )
{
    decision_func = 0;
    class_labels = 0;
    class_weights = 0;
    storage = 0;
    var_idx = 0;
    kernel = 0;
    solver = 0;
    default_model_name = "my_svm";

    train( _train_data, _responses, _var_idx, _sample_idx, _params );
}


int CvSVM::get_support_vector_count() const
{
    return sv_total;
}


const float* CvSVM::get_support_vector(int i) const
{
    return sv && (unsigned)i < (unsigned)sv_total ? sv[i] : 0;
}

bool CvSVM::set_params( const CvSVMParams& _params )
{
    bool ok = false;

    CV_FUNCNAME( "CvSVM::set_params" );

    __BEGIN__;

    int kernelType, svmType;

    params = _params;

    kernelType = params.kernelType;
    svmType = params.svmType;

    if( kernelType != LINEAR && kernelType != POLY &&
        kernelType != SIGMOID && kernelType != RBF &&
        kernelType != INTER && kernelType != CHI2)
        CV_ERROR( CV_StsBadArg, "Unknown/unsupported kernel type" );

    if( kernelType == LINEAR )
        params.gamma = 1;
    else if( params.gamma <= 0 )
        CV_ERROR( CV_StsOutOfRange, "gamma parameter of the kernel must be positive" );

    if( kernelType != SIGMOID && kernelType != POLY )
        params.coef0 = 0;
    else if( params.coef0 < 0 )
        CV_ERROR( CV_StsOutOfRange, "The kernel parameter <coef0> must be positive or zero" );

    if( kernelType != POLY )
        params.degree = 0;
    else if( params.degree <= 0 )
        CV_ERROR( CV_StsOutOfRange, "The kernel parameter <degree> must be positive" );

    if( svmType != C_SVC && svmType != NU_SVC &&
        svmType != ONE_CLASS && svmType != EPS_SVR &&
        svmType != NU_SVR )
        CV_ERROR( CV_StsBadArg, "Unknown/unsupported SVM type" );

    if( svmType == ONE_CLASS || svmType == NU_SVC )
        params.C = 0;
    else if( params.C <= 0 )
        CV_ERROR( CV_StsOutOfRange, "The parameter C must be positive" );

    if( svmType == C_SVC || svmType == EPS_SVR )
        params.nu = 0;
    else if( params.nu <= 0 || params.nu >= 1 )
        CV_ERROR( CV_StsOutOfRange, "The parameter nu must be between 0 and 1" );

    if( svmType != EPS_SVR )
        params.p = 0;
    else if( params.p <= 0 )
        CV_ERROR( CV_StsOutOfRange, "The parameter p must be positive" );

    if( svmType != C_SVC )
        params.class_weights = 0;

    params.termCrit = cvCheckTermCriteria( params.termCrit, DBL_EPSILON, INT_MAX );
    params.termCrit.epsilon = MAX( params.termCrit.epsilon, DBL_EPSILON );
    ok = true;

    __END__;

    return ok;
}



void CvSVM::create_kernel()
{
    kernel = new CvSVMKernel(&params,0);
}


void CvSVM::create_solver( )
{
    solver = new CvSVMSolver;
}


// switching function
bool CvSVM::train1( int sample_count, int var_count, const float** samples,
                    const void* _responses, double Cp, double Cn,
                    CvMemStorage* _storage, double* alpha, double& rho )
{
    bool ok = false;

    //CV_FUNCNAME( "CvSVM::train1" );

    __BEGIN__;

    CvSVMSolutionInfo si;
    int svmType = params.svmType;

    si.rho = 0;

    ok = svmType == C_SVC ? solver->solve_c_svc( sample_count, var_count, samples, (schar*)_responses,
                                                  Cp, Cn, _storage, kernel, alpha, si ) :
         svmType == NU_SVC ? solver->solve_nu_svc( sample_count, var_count, samples, (schar*)_responses,
                                                    _storage, kernel, alpha, si ) :
         svmType == ONE_CLASS ? solver->solve_one_class( sample_count, var_count, samples,
                                                          _storage, kernel, alpha, si ) :
         svmType == EPS_SVR ? solver->solve_eps_svr( sample_count, var_count, samples, (float*)_responses,
                                                      _storage, kernel, alpha, si ) :
         svmType == NU_SVR ? solver->solve_nu_svr( sample_count, var_count, samples, (float*)_responses,
                                                    _storage, kernel, alpha, si ) : false;

    rho = si.rho;

    __END__;

    return ok;
}


bool CvSVM::do_train( int svmType, int sample_count, int var_count, const float** samples,
                    const CvMat* responses, CvMemStorage* temp_storage, double* alpha )
{
    bool ok = false;

    CV_FUNCNAME( "CvSVM::do_train" );

    __BEGIN__;

    CvSVMDecisionFunc* df = 0;
    const int sample_size = var_count*sizeof(samples[0][0]);
    int i, j, k;

    cvClearMemStorage( storage );

    if( svmType == ONE_CLASS || svmType == EPS_SVR || svmType == NU_SVR )
    {
        int sv_count = 0;

        CV_CALL( decision_func = df =
            (CvSVMDecisionFunc*)cvAlloc( sizeof(df[0]) ));

        df->rho = 0;
        if( !train1( sample_count, var_count, samples, svmType == ONE_CLASS ? 0 :
            responses->data.i, 0, 0, temp_storage, alpha, df->rho ))
            EXIT;

        for( i = 0; i < sample_count; i++ )
            sv_count += fabs(alpha[i]) > 0;

        CV_Assert(sv_count != 0);

        sv_total = df->sv_count = sv_count;
        CV_CALL( df->alpha = (double*)cvMemStorageAlloc( storage, sv_count*sizeof(df->alpha[0])) );
        CV_CALL( sv = (float**)cvMemStorageAlloc( storage, sv_count*sizeof(sv[0])));

        for( i = k = 0; i < sample_count; i++ )
        {
            if( fabs(alpha[i]) > 0 )
            {
                CV_CALL( sv[k] = (float*)cvMemStorageAlloc( storage, sample_size ));
                memcpy( sv[k], samples[i], sample_size );
                df->alpha[k++] = alpha[i];
            }
        }
    }
    else
    {
        int class_count = class_labels->cols;
        int* sv_tab = 0;
        const float** temp_samples = 0;
        int* class_ranges = 0;
        schar* temp_y = 0;
        assert( svmType == CvSVM::C_SVC || svmType == CvSVM::NU_SVC );

        if( svmType == CvSVM::C_SVC && params.class_weights )
        {
            const CvMat* cw = params.class_weights;

            if( !CV_IS_MAT(cw) || (cw->cols != 1 && cw->rows != 1) ||
                cw->rows + cw->cols - 1 != class_count ||
                (CV_MAT_TYPE(cw->type) != CV_32FC1 && CV_MAT_TYPE(cw->type) != CV_64FC1) )
                CV_ERROR( CV_StsBadArg, "params.class_weights must be 1d floating-point vector "
                    "containing as many elements as the number of classes" );

            CV_CALL( class_weights = cvCreateMat( cw->rows, cw->cols, CV_64F ));
            CV_CALL( cvConvert( cw, class_weights ));
            CV_CALL( cvScale( class_weights, class_weights, params.C ));
        }

        CV_CALL( decision_func = df = (CvSVMDecisionFunc*)cvAlloc(
            (class_count*(class_count-1)/2)*sizeof(df[0])));

        CV_CALL( sv_tab = (int*)cvMemStorageAlloc( temp_storage, sample_count*sizeof(sv_tab[0]) ));
        memset( sv_tab, 0, sample_count*sizeof(sv_tab[0]) );
        CV_CALL( class_ranges = (int*)cvMemStorageAlloc( temp_storage,
                            (class_count + 1)*sizeof(class_ranges[0])));
        CV_CALL( temp_samples = (const float**)cvMemStorageAlloc( temp_storage,
                            sample_count*sizeof(temp_samples[0])));
        CV_CALL( temp_y = (schar*)cvMemStorageAlloc( temp_storage, sample_count));

        class_ranges[class_count] = 0;
        cvSortSamplesByClasses( samples, responses, class_ranges, 0 );
        //check that while cross-validation there were the samples from all the classes
        if( class_ranges[class_count] <= 0 )
            CV_ERROR( CV_StsBadArg, "While cross-validation one or more of the classes have "
            "been fell out of the sample. Try to enlarge <CvSVMParams::k_fold>" );

        if( svmType == NU_SVC )
        {
            // check if nu is feasible
            for(i = 0; i < class_count; i++ )
            {
                int ci = class_ranges[i+1] - class_ranges[i];
                for( j = i+1; j< class_count; j++ )
                {
                    int cj = class_ranges[j+1] - class_ranges[j];
                    if( params.nu*(ci + cj)*0.5 > MIN( ci, cj ) )
                    {
                        // !!!TODO!!! add some diagnostic
                        EXIT; // exit immediately; will release the model and return NULL pointer
                    }
                }
            }
        }

        // train n*(n-1)/2 classifiers
        for( i = 0; i < class_count; i++ )
        {
            for( j = i+1; j < class_count; j++, df++ )
            {
                int si = class_ranges[i], ci = class_ranges[i+1] - si;
                int sj = class_ranges[j], cj = class_ranges[j+1] - sj;
                double Cp = params.C, Cn = Cp;
                int k1 = 0, sv_count = 0;

                for( k = 0; k < ci; k++ )
                {
                    temp_samples[k] = samples[si + k];
                    temp_y[k] = 1;
                }

                for( k = 0; k < cj; k++ )
                {
                    temp_samples[ci + k] = samples[sj + k];
                    temp_y[ci + k] = -1;
                }

                if( class_weights )
                {
                    Cp = class_weights->data.db[i];
                    Cn = class_weights->data.db[j];
                }

                if( !train1( ci + cj, var_count, temp_samples, temp_y,
                             Cp, Cn, temp_storage, alpha, df->rho ))
                    EXIT;

                for( k = 0; k < ci + cj; k++ )
                    sv_count += fabs(alpha[k]) > 0;

                df->sv_count = sv_count;

                CV_CALL( df->alpha = (double*)cvMemStorageAlloc( temp_storage,
                                                sv_count*sizeof(df->alpha[0])));
                CV_CALL( df->sv_index = (int*)cvMemStorageAlloc( temp_storage,
                                                sv_count*sizeof(df->sv_index[0])));

                for( k = 0; k < ci; k++ )
                {
                    if( fabs(alpha[k]) > 0 )
                    {
                        sv_tab[si + k] = 1;
                        df->sv_index[k1] = si + k;
                        df->alpha[k1++] = alpha[k];
                    }
                }

                for( k = 0; k < cj; k++ )
                {
                    if( fabs(alpha[ci + k]) > 0 )
                    {
                        sv_tab[sj + k] = 1;
                        df->sv_index[k1] = sj + k;
                        df->alpha[k1++] = alpha[ci + k];
                    }
                }
            }
        }

        // allocate support vectors and initialize sv_tab
        for( i = 0, k = 0; i < sample_count; i++ )
        {
            if( sv_tab[i] )
                sv_tab[i] = ++k;
        }

        sv_total = k;
        CV_CALL( sv = (float**)cvMemStorageAlloc( storage, sv_total*sizeof(sv[0])));

        for( i = 0, k = 0; i < sample_count; i++ )
        {
            if( sv_tab[i] )
            {
                CV_CALL( sv[k] = (float*)cvMemStorageAlloc( storage, sample_size ));
                memcpy( sv[k], samples[i], sample_size );
                k++;
            }
        }

        df = (CvSVMDecisionFunc*)decision_func;

        // set sv pointers
        for( i = 0; i < class_count; i++ )
        {
            for( j = i+1; j < class_count; j++, df++ )
            {
                for( k = 0; k < df->sv_count; k++ )
                {
                    df->sv_index[k] = sv_tab[df->sv_index[k]]-1;
                    assert( (unsigned)df->sv_index[k] < (unsigned)sv_total );
                }
            }
        }
    }

    optimize_linear_svm();
    ok = true;

    __END__;

    return ok;
}


void CvSVM::optimize_linear_svm()
{
    // we optimize only linear SVM: compress all the support vectors into one.
    if( params.kernelType != LINEAR )
        return;

    int class_count = class_labels ? class_labels->cols :
            params.svmType == CvSVM::ONE_CLASS ? 1 : 0;

    int i, df_count = class_count > 1 ? class_count*(class_count-1)/2 : 1;
    CvSVMDecisionFunc* df = decision_func;

    for( i = 0; i < df_count; i++ )
    {
        int sv_count = df[i].sv_count;
        if( sv_count != 1 )
            break;
    }

    // if every decision functions uses a single support vector;
    // it's already compressed. skip it then.
    if( i == df_count )
        return;

    int var_count = get_var_count();
    cv::AutoBuffer<double> vbuf(var_count);
    double* v = vbuf;
    float** new_sv = (float**)cvMemStorageAlloc(storage, df_count*sizeof(new_sv[0]));

    for( i = 0; i < df_count; i++ )
    {
        new_sv[i] = (float*)cvMemStorageAlloc(storage, var_count*sizeof(new_sv[i][0]));
        float* dst = new_sv[i];
        memset(v, 0, var_count*sizeof(v[0]));
        int j, k, sv_count = df[i].sv_count;
        for( j = 0; j < sv_count; j++ )
        {
            const float* src = class_count > 1 && df[i].sv_index ? sv[df[i].sv_index[j]] : sv[j];
            double a = df[i].alpha[j];
            for( k = 0; k < var_count; k++ )
                v[k] += src[k]*a;
        }
        for( k = 0; k < var_count; k++ )
            dst[k] = (float)v[k];
        df[i].sv_count = 1;
        df[i].alpha[0] = 1.;
        if( class_count > 1 && df[i].sv_index )
            df[i].sv_index[0] = i;
    }

    sv = new_sv;
    sv_total = df_count;
}


bool CvSVM::train( const CvMat* _train_data, const CvMat* _responses,
    const CvMat* _var_idx, const CvMat* _sample_idx, CvSVMParams _params )
{
    bool ok = false;
    CvMat* responses = 0;
    CvMemStorage* temp_storage = 0;
    const float** samples = 0;

    CV_FUNCNAME( "CvSVM::train" );

    __BEGIN__;

    int svmType, sample_count, var_count, sample_size;
    int block_size = 1 << 16;
    double* alpha;

    clear();
    CV_CALL( set_params( _params ));

    svmType = _params.svmType;

    /* Prepare training data and related parameters */
    CV_CALL( cvPrepareTrainData( "CvSVM::train", _train_data, CV_ROW_SAMPLE,
                                 svmType != CvSVM::ONE_CLASS ? _responses : 0,
                                 svmType == CvSVM::C_SVC ||
                                 svmType == CvSVM::NU_SVC ? CV_VAR_CATEGORICAL :
                                 CV_VAR_ORDERED, _var_idx, _sample_idx,
                                 false, &samples, &sample_count, &var_count, &var_all,
                                 &responses, &class_labels, &var_idx ));


    sample_size = var_count*sizeof(samples[0][0]);

    // make the storage block size large enough to fit all
    // the temporary vectors and output support vectors.
    block_size = MAX( block_size, sample_count*(int)sizeof(CvSVMKernelRow));
    block_size = MAX( block_size, sample_count*2*(int)sizeof(double) + 1024 );
    block_size = MAX( block_size, sample_size*2 + 1024 );

    CV_CALL( storage = cvCreateMemStorage(block_size + sizeof(CvMemBlock) + sizeof(CvSeqBlock)));
    CV_CALL( temp_storage = cvCreateChildMemStorage(storage));
    CV_CALL( alpha = (double*)cvMemStorageAlloc(temp_storage, sample_count*sizeof(double)));

    create_kernel();
    create_solver();

    if( !do_train( svmType, sample_count, var_count, samples, responses, temp_storage, alpha ))
        EXIT;

    ok = true; // model has been trained succesfully

    __END__;

    delete solver;
    solver = 0;
    cvReleaseMemStorage( &temp_storage );
    cvReleaseMat( &responses );
    cvFree( &samples );

    if( cvGetErrStatus() < 0 || !ok )
        clear();

    return ok;
}

struct indexedratio
{
    double val;
    int ind;
    int count_smallest, count_biggest;
    void eval() { val = (double) count_smallest/(count_smallest+count_biggest); }
};

static int CV_CDECL
icvCmpIndexedratio( const void* a, const void* b )
{
    return ((const indexedratio*)a)->val < ((const indexedratio*)b)->val ? -1
    : ((const indexedratio*)a)->val > ((const indexedratio*)b)->val ? 1
    : 0;
}

bool CvSVM::train_auto( const CvMat* _train_data, const CvMat* _responses,
    const CvMat* _var_idx, const CvMat* _sample_idx, CvSVMParams _params, int k_fold,
    CvParamGrid C_grid, CvParamGrid gamma_grid, CvParamGrid p_grid,
    CvParamGrid nu_grid, CvParamGrid coef_grid, CvParamGrid degree_grid,
    bool balanced)
{
    bool ok = false;
    CvMat* responses = 0;
    CvMat* responses_local = 0;
    CvMemStorage* temp_storage = 0;
    const float** samples = 0;
    const float** samples_local = 0;

    CV_FUNCNAME( "CvSVM::train_auto" );
    __BEGIN__;

    int svmType, sample_count, var_count, sample_size;
    int block_size = 1 << 16;
    double* alpha;
    RNG* rng = &theRNG();

    // all steps are logarithmic and must be > 1
    double degree_step = 10, g_step = 10, coef_step = 10, C_step = 10, nu_step = 10, p_step = 10;
    double gamma = 0, curr_c = 0, degree = 0, coef = 0, p = 0, nu = 0;
    double best_degree = 0, best_gamma = 0, best_coef = 0, best_C = 0, best_nu = 0, best_p = 0;
    float min_error = FLT_MAX, error;

    if( _params.svmType == CvSVM::ONE_CLASS )
    {
        if(!train( _train_data, _responses, _var_idx, _sample_idx, _params ))
            EXIT;
        return true;
    }

    clear();

    if( k_fold < 2 )
        CV_ERROR( CV_StsBadArg, "Parameter <k_fold> must be > 1" );

    CV_CALL(set_params( _params ));
    svmType = _params.svmType;

    // All the parameters except, possibly, <coef0> are positive.
    // <coef0> is nonnegative
    if( C_grid.logStep <= 1 )
    {
        C_grid.minVal = C_grid.maxVal = params.C;
        C_grid.logStep = 10;
    }
    else
        CV_CALL(C_grid.check());

    if( gamma_grid.logStep <= 1 )
    {
        gamma_grid.minVal = gamma_grid.maxVal = params.gamma;
        gamma_grid.logStep = 10;
    }
    else
        CV_CALL(gamma_grid.check());

    if( p_grid.logStep <= 1 )
    {
        p_grid.minVal = p_grid.maxVal = params.p;
        p_grid.logStep = 10;
    }
    else
        CV_CALL(p_grid.check());

    if( nu_grid.logStep <= 1 )
    {
        nu_grid.minVal = nu_grid.maxVal = params.nu;
        nu_grid.logStep = 10;
    }
    else
        CV_CALL(nu_grid.check());

    if( coef_grid.logStep <= 1 )
    {
        coef_grid.minVal = coef_grid.maxVal = params.coef0;
        coef_grid.logStep = 10;
    }
    else
        CV_CALL(coef_grid.check());

    if( degree_grid.logStep <= 1 )
    {
        degree_grid.minVal = degree_grid.maxVal = params.degree;
        degree_grid.logStep = 10;
    }
    else
        CV_CALL(degree_grid.check());

    // these parameters are not used:
    if( params.kernelType != CvSVM::POLY )
        degree_grid.minVal = degree_grid.maxVal = params.degree;
    if( params.kernelType == CvSVM::LINEAR )
        gamma_grid.minVal = gamma_grid.maxVal = params.gamma;
    if( params.kernelType != CvSVM::POLY && params.kernelType != CvSVM::SIGMOID )
        coef_grid.minVal = coef_grid.maxVal = params.coef0;
    if( svmType == CvSVM::NU_SVC || svmType == CvSVM::ONE_CLASS )
        C_grid.minVal = C_grid.maxVal = params.C;
    if( svmType == CvSVM::C_SVC || svmType == CvSVM::EPS_SVR )
        nu_grid.minVal = nu_grid.maxVal = params.nu;
    if( svmType != CvSVM::EPS_SVR )
        p_grid.minVal = p_grid.maxVal = params.p;

    CV_ASSERT( g_step > 1 && degree_step > 1 && coef_step > 1);
    CV_ASSERT( p_step > 1 && C_step > 1 && nu_step > 1 );

    /* Prepare training data and related parameters */
    CV_CALL(cvPrepareTrainData( "CvSVM::train_auto", _train_data, CV_ROW_SAMPLE,
                                 svmType != CvSVM::ONE_CLASS ? _responses : 0,
                                 svmType == CvSVM::C_SVC ||
                                 svmType == CvSVM::NU_SVC ? CV_VAR_CATEGORICAL :
                                 CV_VAR_ORDERED, _var_idx, _sample_idx,
                                 false, &samples, &sample_count, &var_count, &var_all,
                                 &responses, &class_labels, &var_idx ));

    sample_size = var_count*sizeof(samples[0][0]);

    // make the storage block size large enough to fit all
    // the temporary vectors and output support vectors.
    block_size = MAX( block_size, sample_count*(int)sizeof(CvSVMKernelRow));
    block_size = MAX( block_size, sample_count*2*(int)sizeof(double) + 1024 );
    block_size = MAX( block_size, sample_size*2 + 1024 );

    CV_CALL( storage = cvCreateMemStorage(block_size + sizeof(CvMemBlock) + sizeof(CvSeqBlock)));
    CV_CALL(temp_storage = cvCreateChildMemStorage(storage));
    CV_CALL(alpha = (double*)cvMemStorageAlloc(temp_storage, sample_count*sizeof(double)));

    create_kernel();
    create_solver();

    {
    const int testset_size = sample_count/k_fold;
    const int trainset_size = sample_count - testset_size;
    const int last_testset_size = sample_count - testset_size*(k_fold-1);
    const int last_trainset_size = sample_count - last_testset_size;
    const bool is_regression = (svmType == EPS_SVR) || (svmType == NU_SVR);

    size_t resp_elem_size = CV_ELEM_SIZE(responses->type);
    size_t size = 2*last_trainset_size*sizeof(samples[0]);

    samples_local = (const float**) cvAlloc( size );
    memset( samples_local, 0, size );

    responses_local = cvCreateMat( 1, trainset_size, CV_MAT_TYPE(responses->type) );
    cvZero( responses_local );

    // randomly permute samples and responses
    for(int i = 0; i < sample_count; i++ )
    {
        int i1 = (*rng)(sample_count);
        int i2 = (*rng)(sample_count);
        const float* temp;
        float t;
        int y;

        CV_SWAP( samples[i1], samples[i2], temp );
        if( is_regression )
            CV_SWAP( responses->data.fl[i1], responses->data.fl[i2], t );
        else
            CV_SWAP( responses->data.i[i1], responses->data.i[i2], y );
    }

    if (!is_regression && class_labels->cols==2 && balanced)
    {
        // count class samples
        int num_0=0,num_1=0;
        for (int i=0; i<sample_count; ++i)
        {
            if (responses->data.i[i]==class_labels->data.i[0])
                ++num_0;
            else
                ++num_1;
        }

        int label_smallest_class;
        int label_biggest_class;
        if (num_0 < num_1)
        {
            label_biggest_class = class_labels->data.i[1];
            label_smallest_class = class_labels->data.i[0];
        }
        else
        {
            label_biggest_class = class_labels->data.i[0];
            label_smallest_class = class_labels->data.i[1];
            int y;
            CV_SWAP(num_0,num_1,y);
        }
        const double class_ratio = (double) num_0/sample_count;
        // calculate class ratio of each fold
        indexedratio *ratios=0;
        ratios = (indexedratio*) cvAlloc(k_fold*sizeof(*ratios));
        for (int k=0, i_begin=0; k<k_fold; ++k, i_begin+=testset_size)
        {
            int count0=0;
            int count1=0;
            int i_end = i_begin + (k<k_fold-1 ? testset_size : last_testset_size);
            for (int i=i_begin; i<i_end; ++i)
            {
                if (responses->data.i[i]==label_smallest_class)
                    ++count0;
                else
                    ++count1;
            }
            ratios[k].ind = k;
            ratios[k].count_smallest = count0;
            ratios[k].count_biggest = count1;
            ratios[k].eval();
        }
        // initial distance
        qsort(ratios, k_fold, sizeof(ratios[0]), icvCmpIndexedratio);
        double old_dist = 0.0;
        for (int k=0; k<k_fold; ++k)
            old_dist += cv::abs(ratios[k].val-class_ratio);
        double new_dist = 1.0;
        // iterate to make the folds more balanced
        while (new_dist > 0.0)
        {
            if (ratios[0].count_biggest==0 || ratios[k_fold-1].count_smallest==0)
                break; // we are not able to swap samples anymore
            // what if we swap the samples, calculate the new distance
            ratios[0].count_smallest++;
            ratios[0].count_biggest--;
            ratios[0].eval();
            ratios[k_fold-1].count_smallest--;
            ratios[k_fold-1].count_biggest++;
            ratios[k_fold-1].eval();
            qsort(ratios, k_fold, sizeof(ratios[0]), icvCmpIndexedratio);
            new_dist = 0.0;
            for (int k=0; k<k_fold; ++k)
                new_dist += cv::abs(ratios[k].val-class_ratio);
            if (new_dist < old_dist)
            {
                // swapping really improves, so swap the samples
                // index of the biggest_class sample from the minimum ratio fold
                int i1 = ratios[0].ind * testset_size;
                for ( ; i1<sample_count; ++i1)
                {
                    if (responses->data.i[i1]==label_biggest_class)
                        break;
                }
                // index of the smallest_class sample from the maximum ratio fold
                int i2 = ratios[k_fold-1].ind * testset_size;
                for ( ; i2<sample_count; ++i2)
                {
                    if (responses->data.i[i2]==label_smallest_class)
                        break;
                }
                // swap
                const float* temp;
                int y;
                CV_SWAP( samples[i1], samples[i2], temp );
                CV_SWAP( responses->data.i[i1], responses->data.i[i2], y );
                old_dist = new_dist;
            }
            else
                break; // does not improve, so break the loop
        }
        cvFree(&ratios);
    }

    int* cls_lbls = class_labels ? class_labels->data.i : 0;
    curr_c = C_grid.minVal;
    do
    {
      params.C = curr_c;
      gamma = gamma_grid.minVal;
      do
      {
        params.gamma = gamma;
        p = p_grid.minVal;
        do
        {
          params.p = p;
          nu = nu_grid.minVal;
          do
          {
            params.nu = nu;
            coef = coef_grid.minVal;
            do
            {
              params.coef0 = coef;
              degree = degree_grid.minVal;
              do
              {
                params.degree = degree;

                float** test_samples_ptr = (float**)samples;
                uchar* true_resp = responses->data.ptr;
                int test_size = testset_size;
                int train_size = trainset_size;

                error = 0;
                for(int k = 0; k < k_fold; k++ )
                {
                    memcpy( samples_local, samples, sizeof(samples[0])*test_size*k );
                    memcpy( samples_local + test_size*k, test_samples_ptr + test_size,
                        sizeof(samples[0])*(sample_count - testset_size*(k+1)) );

                    memcpy( responses_local->data.ptr, responses->data.ptr, resp_elem_size*test_size*k );
                    memcpy( responses_local->data.ptr + resp_elem_size*test_size*k,
                        true_resp + resp_elem_size*test_size,
                        resp_elem_size*(sample_count - testset_size*(k+1)) );

                    if( k == k_fold - 1 )
                    {
                        test_size = last_testset_size;
                        train_size = last_trainset_size;
                        responses_local->cols = last_trainset_size;
                    }

                    // Train SVM on <train_size> samples
                    if( !do_train( svmType, train_size, var_count,
                        (const float**)samples_local, responses_local, temp_storage, alpha ) )
                        EXIT;

                    // Compute test set error on <test_size> samples
                    for(int i = 0; i < test_size; i++, true_resp += resp_elem_size, test_samples_ptr++ )
                    {
                        float resp = predict( *test_samples_ptr, var_count );
                        error += is_regression ? powf( resp - *(float*)true_resp, 2 )
                            : ((int)resp != cls_lbls[*(int*)true_resp]);
                    }
                }
                if( min_error > error )
                {
                    min_error   = error;
                    best_degree = degree;
                    best_gamma  = gamma;
                    best_coef   = coef;
                    best_C      = curr_c;
                    best_nu     = nu;
                    best_p      = p;
                }
                degree *= degree_grid.logStep;
              }
              while( degree < degree_grid.maxVal );
              coef *= coef_grid.logStep;
            }
            while( coef < coef_grid.maxVal );
            nu *= nu_grid.logStep;
          }
          while( nu < nu_grid.maxVal );
          p *= p_grid.logStep;
        }
        while( p < p_grid.maxVal );
        gamma *= gamma_grid.logStep;
      }
      while( gamma < gamma_grid.maxVal );
      curr_c *= C_grid.logStep;
    }
    while( curr_c < C_grid.maxVal );
    }

    min_error /= (float) sample_count;

    params.C      = best_C;
    params.nu     = best_nu;
    params.p      = best_p;
    params.gamma  = best_gamma;
    params.degree = best_degree;
    params.coef0  = best_coef;

    CV_CALL(ok = do_train( svmType, sample_count, var_count, samples, responses, temp_storage, alpha ));

    __END__;

    delete solver;
    solver = 0;
    cvReleaseMemStorage( &temp_storage );
    cvReleaseMat( &responses );
    cvReleaseMat( &responses_local );
    cvFree( &samples );
    cvFree( &samples_local );

    if( cvGetErrStatus() < 0 || !ok )
        clear();

    return ok;
}

float CvSVM::predict( const float* row_sample, int row_len, bool returnDFVal ) const
{
    assert( kernel );
    assert( row_sample );

    int var_count = get_var_count();
    assert( row_len == var_count );
    (void)row_len;

    int class_count = class_labels ? class_labels->cols :
                  params.svmType == ONE_CLASS ? 1 : 0;

    float result = 0;
    cv::AutoBuffer<float> _buffer(sv_total + (class_count+1)*2);
    float* buffer = _buffer;

    if( params.svmType == EPS_SVR ||
        params.svmType == NU_SVR ||
        params.svmType == ONE_CLASS )
    {
        CvSVMDecisionFunc* df = (CvSVMDecisionFunc*)decision_func;
        int i, sv_count = df->sv_count;
        double sum = -df->rho;

        kernel->calc( sv_count, var_count, (const float**)sv, row_sample, buffer );
        for( i = 0; i < sv_count; i++ )
            sum += buffer[i]*df->alpha[i];

        result = params.svmType == ONE_CLASS ? (float)(sum > 0) : (float)sum;
    }
    else if( params.svmType == C_SVC ||
             params.svmType == NU_SVC )
    {
        CvSVMDecisionFunc* df = (CvSVMDecisionFunc*)decision_func;
        int* vote = (int*)(buffer + sv_total);
        int i, j, k;

        memset( vote, 0, class_count*sizeof(vote[0]));
        kernel->calc( sv_total, var_count, (const float**)sv, row_sample, buffer );
        double sum = 0.;

        for( i = 0; i < class_count; i++ )
        {
            for( j = i+1; j < class_count; j++, df++ )
            {
                sum = -df->rho;
                int sv_count = df->sv_count;
                for( k = 0; k < sv_count; k++ )
                    sum += df->alpha[k]*buffer[df->sv_index[k]];

                vote[sum > 0 ? i : j]++;
            }
        }

        for( i = 1, k = 0; i < class_count; i++ )
        {
            if( vote[i] > vote[k] )
                k = i;
        }
        result = returnDFVal && class_count == 2 ? (float)sum : (float)(class_labels->data.i[k]);
    }
    else
        CV_Error( CV_StsBadArg, "INTERNAL ERROR: Unknown SVM type, "
                                "the SVM structure is probably corrupted" );

    return result;
}

float CvSVM::predict( const CvMat* sample, bool returnDFVal ) const
{
    float result = 0;
    float* row_sample = 0;

    CV_FUNCNAME( "CvSVM::predict" );

    __BEGIN__;

    int class_count;

    if( !kernel )
        CV_ERROR( CV_StsBadArg, "The SVM should be trained first" );

    class_count = class_labels ? class_labels->cols :
                  params.svmType == ONE_CLASS ? 1 : 0;

    CV_CALL( cvPreparePredictData( sample, var_all, var_idx,
                                   class_count, 0, &row_sample ));
    result = predict( row_sample, get_var_count(), returnDFVal );

    __END__;

    if( sample && (!CV_IS_MAT(sample) || sample->data.fl != row_sample) )
        cvFree( &row_sample );

    return result;
}

struct predict_body_svm : ParallelLoopBody {
    predict_body_svm(const CvSVM* _pointer, float* _result, const CvMat* _samples, CvMat* _results, bool _returnDFVal)
    {
        pointer = _pointer;
        result = _result;
        samples = _samples;
        results = _results;
        returnDFVal = _returnDFVal;
    }

    const CvSVM* pointer;
    float* result;
    const CvMat* samples;
    CvMat* results;
    bool returnDFVal;

    void operator()( const cv::Range& range ) const
    {
        for(int i = range.start; i < range.end; i++ )
        {
            CvMat sample;
            cvGetRow( samples, &sample, i );
            int r = (int)pointer->predict(&sample, returnDFVal);
            if (results)
                results->data.fl[i] = (float)r;
            if (i == 0)
                *result = (float)r;
    }
    }
};

float CvSVM::predict(const CvMat* samples, CV_OUT CvMat* results, bool returnDFVal) const
{
    float result = 0;
    cv::parallel_for_(cv::Range(0, samples->rows),
             predict_body_svm(this, &result, samples, results, returnDFVal)
    );
    return result;
}

void CvSVM::predict( cv::InputArray _samples, cv::OutputArray _results ) const
{
    _results.create(_samples.size().height, 1, CV_32F);
    CvMat samples = _samples.getMat(), results = _results.getMat();
    predict(&samples, &results);
}

CvSVM::CvSVM( const Mat& _train_data, const Mat& _responses,
              const Mat& _var_idx, const Mat& _sample_idx, CvSVMParams _params )
{
    decision_func = 0;
    class_labels = 0;
    class_weights = 0;
    storage = 0;
    var_idx = 0;
    kernel = 0;
    solver = 0;
    default_model_name = "my_svm";

    train( _train_data, _responses, _var_idx, _sample_idx, _params );
}

bool CvSVM::train( const Mat& _train_data, const Mat& _responses,
                  const Mat& _var_idx, const Mat& _sample_idx, CvSVMParams _params )
{
    CvMat tdata = _train_data, responses = _responses, vidx = _var_idx, sidx = _sample_idx;
    return train(&tdata, &responses, vidx.data.ptr ? &vidx : 0, sidx.data.ptr ? &sidx : 0, _params);
}


bool CvSVM::train_auto( const Mat& _train_data, const Mat& _responses,
                       const Mat& _var_idx, const Mat& _sample_idx, CvSVMParams _params, int k_fold,
                       CvParamGrid C_grid, CvParamGrid gamma_grid, CvParamGrid p_grid,
                       CvParamGrid nu_grid, CvParamGrid coef_grid, CvParamGrid degree_grid, bool balanced )
{
    CvMat tdata = _train_data, responses = _responses, vidx = _var_idx, sidx = _sample_idx;
    return train_auto(&tdata, &responses, vidx.data.ptr ? &vidx : 0,
                      sidx.data.ptr ? &sidx : 0, _params, k_fold, C_grid, gamma_grid, p_grid,
                      nu_grid, coef_grid, degree_grid, balanced);
}

float CvSVM::predict( const Mat& _sample, bool returnDFVal ) const
{
    CvMat sample = _sample;
    return predict(&sample, returnDFVal);
}


void CvSVM::write_params( CvFileStorage* fs ) const
{
    int svmType = params.svmType;
    int kernelType = params.kernelType;

    const char* svm_type_str =
        svmType == CvSVM::C_SVC ? "C_SVC" :
        svmType == CvSVM::NU_SVC ? "NU_SVC" :
        svmType == CvSVM::ONE_CLASS ? "ONE_CLASS" :
        svmType == CvSVM::EPS_SVR ? "EPS_SVR" :
        svmType == CvSVM::NU_SVR ? "NU_SVR" : 0;
    const char* kernel_type_str =
        kernelType == CvSVM::LINEAR ? "LINEAR" :
        kernelType == CvSVM::POLY ? "POLY" :
        kernelType == CvSVM::RBF ? "RBF" :
        kernelType == CvSVM::SIGMOID ? "SIGMOID" : 0;

    if( svm_type_str )
        cvWriteString( fs, "svmType", svm_type_str );
    else
        cvWriteInt( fs, "svmType", svmType );

    // save kernel
    cvStartWriteStruct( fs, "kernel", CV_NODE_MAP + CV_NODE_FLOW );

    if( kernel_type_str )
        cvWriteString( fs, "type", kernel_type_str );
    else
        cvWriteInt( fs, "type", kernelType );

    if( kernelType == CvSVM::POLY || !kernel_type_str )
        cvWriteReal( fs, "degree", params.degree );

    if( kernelType != CvSVM::LINEAR || !kernel_type_str )
        cvWriteReal( fs, "gamma", params.gamma );

    if( kernelType == CvSVM::POLY || kernelType == CvSVM::SIGMOID || !kernel_type_str )
        cvWriteReal( fs, "coef0", params.coef0 );

    cvEndWriteStruct(fs);

    if( svmType == CvSVM::C_SVC || svmType == CvSVM::EPS_SVR ||
        svmType == CvSVM::NU_SVR || !svm_type_str )
        cvWriteReal( fs, "C", params.C );

    if( svmType == CvSVM::NU_SVC || svmType == CvSVM::ONE_CLASS ||
        svmType == CvSVM::NU_SVR || !svm_type_str )
        cvWriteReal( fs, "nu", params.nu );

    if( svmType == CvSVM::EPS_SVR || !svm_type_str )
        cvWriteReal( fs, "p", params.p );

    cvStartWriteStruct( fs, "term_criteria", CV_NODE_MAP + CV_NODE_FLOW );
    if( params.termCrit.type & CV_TERMCRIT_EPS )
        cvWriteReal( fs, "epsilon", params.termCrit.epsilon );
    if( params.termCrit.type & CV_TERMCRIT_ITER )
        cvWriteInt( fs, "iterations", params.termCrit.max_iter );
    cvEndWriteStruct( fs );
}


static bool isSvmModelApplicable(int sv_total, int var_all, int var_count, int class_count)
{
    return (sv_total > 0 && var_count > 0 && var_count <= var_all && class_count >= 0);
}


void CvSVM::write( CvFileStorage* fs, const char* name ) const
{
    int i, var_count = get_var_count(), df_count;
    int class_count = class_labels ? class_labels->cols :
                      params.svmType == CvSVM::ONE_CLASS ? 1 : 0;
    const CvSVMDecisionFunc* df = decision_func;
    if( !isSvmModelApplicable(sv_total, var_all, var_count, class_count) )
        CV_ERROR( CV_StsParseError, "SVM model data is invalid, check sv_count, var_* and class_count tags" );

    cvStartWriteStruct( fs, name, CV_NODE_MAP, CV_TYPE_NAME_ML_SVM );

    write_params( fs );

    cvWriteInt( fs, "var_all", var_all );
    cvWriteInt( fs, "var_count", var_count );

    if( class_count )
    {
        cvWriteInt( fs, "class_count", class_count );

        if( class_labels )
            cvWrite( fs, "class_labels", class_labels );

        if( class_weights )
            cvWrite( fs, "class_weights", class_weights );
    }

    if( var_idx )
        cvWrite( fs, "var_idx", var_idx );

    // write the joint collection of support vectors
    cvWriteInt( fs, "sv_total", sv_total );
    cvStartWriteStruct( fs, "support_vectors", CV_NODE_SEQ );
    for( i = 0; i < sv_total; i++ )
    {
        cvStartWriteStruct( fs, 0, CV_NODE_SEQ + CV_NODE_FLOW );
        cvWriteRawData( fs, sv[i], var_count, "f" );
        cvEndWriteStruct( fs );
    }

    cvEndWriteStruct( fs );

    // write decision functions
    df_count = class_count > 1 ? class_count*(class_count-1)/2 : 1;
    df = decision_func;

    cvStartWriteStruct( fs, "decision_functions", CV_NODE_SEQ );
    for( i = 0; i < df_count; i++ )
    {
        int sv_count = df[i].sv_count;
        cvStartWriteStruct( fs, 0, CV_NODE_MAP );
        cvWriteInt( fs, "sv_count", sv_count );
        cvWriteReal( fs, "rho", df[i].rho );
        cvStartWriteStruct( fs, "alpha", CV_NODE_SEQ+CV_NODE_FLOW );
        cvWriteRawData( fs, df[i].alpha, df[i].sv_count, "d" );
        cvEndWriteStruct( fs );
        if( class_count > 1 )
        {
            cvStartWriteStruct( fs, "index", CV_NODE_SEQ+CV_NODE_FLOW );
            cvWriteRawData( fs, df[i].sv_index, df[i].sv_count, "i" );
            cvEndWriteStruct( fs );
        }
        else
            CV_ASSERT( sv_count == sv_total );
        cvEndWriteStruct( fs );
    }
    cvEndWriteStruct( fs );
    cvEndWriteStruct( fs );
}


void CvSVM::read_params( CvFileStorage* fs, CvFileNode* svm_node )
{
    int svmType, kernelType;
    CvSVMParams _params;

    CvFileNode* tmp_node = cvGetFileNodeByName( fs, svm_node, "svmType" );
    CvFileNode* kernel_node;
    if( !tmp_node )
        CV_ERROR( CV_StsBadArg, "svmType tag is not found" );

    if( CV_NODE_TYPE(tmp_node->tag) == CV_NODE_INT )
        svmType = cvReadInt( tmp_node, -1 );
    else
    {
        const char* svm_type_str = cvReadString( tmp_node, "" );
        svmType =
            strcmp( svm_type_str, "C_SVC" ) == 0 ? CvSVM::C_SVC :
            strcmp( svm_type_str, "NU_SVC" ) == 0 ? CvSVM::NU_SVC :
            strcmp( svm_type_str, "ONE_CLASS" ) == 0 ? CvSVM::ONE_CLASS :
            strcmp( svm_type_str, "EPS_SVR" ) == 0 ? CvSVM::EPS_SVR :
            strcmp( svm_type_str, "NU_SVR" ) == 0 ? CvSVM::NU_SVR : -1;

        if( svmType < 0 )
            CV_ERROR( CV_StsParseError, "Missing of invalid SVM type" );
    }

    kernel_node = cvGetFileNodeByName( fs, svm_node, "kernel" );
    if( !kernel_node )
        CV_ERROR( CV_StsParseError, "SVM kernel tag is not found" );

    tmp_node = cvGetFileNodeByName( fs, kernel_node, "type" );
    if( !tmp_node )
        CV_ERROR( CV_StsParseError, "SVM kernel type tag is not found" );

    if( CV_NODE_TYPE(tmp_node->tag) == CV_NODE_INT )
        kernelType = cvReadInt( tmp_node, -1 );
    else
    {
        const char* kernel_type_str = cvReadString( tmp_node, "" );
        kernelType =
            strcmp( kernel_type_str, "LINEAR" ) == 0 ? CvSVM::LINEAR :
            strcmp( kernel_type_str, "POLY" ) == 0 ? CvSVM::POLY :
            strcmp( kernel_type_str, "RBF" ) == 0 ? CvSVM::RBF :
            strcmp( kernel_type_str, "SIGMOID" ) == 0 ? CvSVM::SIGMOID : -1;

        if( kernelType < 0 )
            CV_ERROR( CV_StsParseError, "Missing of invalid SVM kernel type" );
    }

    _params.svmType = svmType;
    _params.kernelType = kernelType;
    _params.degree = cvReadRealByName( fs, kernel_node, "degree", 0 );
    _params.gamma = cvReadRealByName( fs, kernel_node, "gamma", 0 );
    _params.coef0 = cvReadRealByName( fs, kernel_node, "coef0", 0 );

    _params.C = cvReadRealByName( fs, svm_node, "C", 0 );
    _params.nu = cvReadRealByName( fs, svm_node, "nu", 0 );
    _params.p = cvReadRealByName( fs, svm_node, "p", 0 );
    _params.class_weights = 0;

    tmp_node = cvGetFileNodeByName( fs, svm_node, "term_criteria" );
    if( tmp_node )
    {
        _params.termCrit.epsilon = cvReadRealByName( fs, tmp_node, "epsilon", -1. );
        _params.termCrit.max_iter = cvReadIntByName( fs, tmp_node, "iterations", -1 );
        _params.termCrit.type = (_params.termCrit.epsilon >= 0 ? CV_TERMCRIT_EPS : 0) +
                               (_params.termCrit.max_iter >= 0 ? CV_TERMCRIT_ITER : 0);
    }
    else
        _params.termCrit = cvTermCriteria( CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 1000, FLT_EPSILON );

    set_params( _params );
}

void CvSVM::read( CvFileStorage* fs, CvFileNode* svm_node )
{
    const double not_found_dbl = DBL_MAX;

    CV_FUNCNAME( "CvSVM::read" );

    __BEGIN__;

    int i, var_count, df_count, class_count;
    int block_size = 1 << 16, sv_size;
    CvFileNode *sv_node, *df_node;
    CvSVMDecisionFunc* df;
    CvSeqReader reader;

    if( !svm_node )
        CV_ERROR( CV_StsParseError, "The requested element is not found" );

    clear();

    // read SVM parameters
    read_params( fs, svm_node );

    // and top-level data
    sv_total = cvReadIntByName( fs, svm_node, "sv_total", -1 );
    var_all = cvReadIntByName( fs, svm_node, "var_all", -1 );
    var_count = cvReadIntByName( fs, svm_node, "var_count", var_all );
    class_count = cvReadIntByName( fs, svm_node, "class_count", 0 );

    if( !isSvmModelApplicable(sv_total, var_all, var_count, class_count) )
        CV_ERROR( CV_StsParseError, "SVM model data is invalid, check sv_count, var_* and class_count tags" );

    CV_CALL( class_labels = (CvMat*)cvReadByName( fs, svm_node, "class_labels" ));
    CV_CALL( class_weights = (CvMat*)cvReadByName( fs, svm_node, "class_weights" ));
    CV_CALL( var_idx = (CvMat*)cvReadByName( fs, svm_node, "var_idx" ));

    if( class_count > 1 && (!class_labels ||
        !CV_IS_MAT(class_labels) || class_labels->cols != class_count))
        CV_ERROR( CV_StsParseError, "Array of class labels is missing or invalid" );

    if( var_count < var_all && (!var_idx || !CV_IS_MAT(var_idx) || var_idx->cols != var_count) )
        CV_ERROR( CV_StsParseError, "var_idx array is missing or invalid" );

    // read support vectors
    sv_node = cvGetFileNodeByName( fs, svm_node, "support_vectors" );
    if( !sv_node || !CV_NODE_IS_SEQ(sv_node->tag))
        CV_ERROR( CV_StsParseError, "Missing or invalid sequence of support vectors" );

    block_size = MAX( block_size, sv_total*(int)sizeof(CvSVMKernelRow));
    block_size = MAX( block_size, sv_total*2*(int)sizeof(double));
    block_size = MAX( block_size, var_all*(int)sizeof(double));

    CV_CALL( storage = cvCreateMemStorage(block_size + sizeof(CvMemBlock) + sizeof(CvSeqBlock)));
    CV_CALL( sv = (float**)cvMemStorageAlloc( storage,
                                sv_total*sizeof(sv[0]) ));

    CV_CALL( cvStartReadSeq( sv_node->data.seq, &reader, 0 ));
    sv_size = var_count*sizeof(sv[0][0]);

    for( i = 0; i < sv_total; i++ )
    {
        CvFileNode* sv_elem = (CvFileNode*)reader.ptr;
        CV_ASSERT( var_count == 1 || (CV_NODE_IS_SEQ(sv_elem->tag) &&
                   sv_elem->data.seq->total == var_count) );

        CV_CALL( sv[i] = (float*)cvMemStorageAlloc( storage, sv_size ));
        CV_CALL( cvReadRawData( fs, sv_elem, sv[i], "f" ));
        CV_NEXT_SEQ_ELEM( sv_node->data.seq->elem_size, reader );
    }

    // read decision functions
    df_count = class_count > 1 ? class_count*(class_count-1)/2 : 1;
    df_node = cvGetFileNodeByName( fs, svm_node, "decision_functions" );
    if( !df_node || !CV_NODE_IS_SEQ(df_node->tag) ||
        df_node->data.seq->total != df_count )
        CV_ERROR( CV_StsParseError, "decision_functions is missing or is not a collection "
                  "or has a wrong number of elements" );

    CV_CALL( df = decision_func = (CvSVMDecisionFunc*)cvAlloc( df_count*sizeof(df[0]) ));
    cvStartReadSeq( df_node->data.seq, &reader, 0 );

    for( i = 0; i < df_count; i++ )
    {
        CvFileNode* df_elem = (CvFileNode*)reader.ptr;
        CvFileNode* alpha_node = cvGetFileNodeByName( fs, df_elem, "alpha" );

        int sv_count = cvReadIntByName( fs, df_elem, "sv_count", -1 );
        if( sv_count <= 0 )
            CV_ERROR( CV_StsParseError, "sv_count is missing or non-positive" );
        df[i].sv_count = sv_count;

        df[i].rho = cvReadRealByName( fs, df_elem, "rho", not_found_dbl );
        if( fabs(df[i].rho - not_found_dbl) < DBL_EPSILON )
            CV_ERROR( CV_StsParseError, "rho is missing" );

        if( !alpha_node )
            CV_ERROR( CV_StsParseError, "alpha is missing in the decision function" );

        CV_CALL( df[i].alpha = (double*)cvMemStorageAlloc( storage,
                                        sv_count*sizeof(df[i].alpha[0])));
        CV_ASSERT( sv_count == 1 || (CV_NODE_IS_SEQ(alpha_node->tag) &&
                   alpha_node->data.seq->total == sv_count) );
        CV_CALL( cvReadRawData( fs, alpha_node, df[i].alpha, "d" ));

        if( class_count > 1 )
        {
            CvFileNode* index_node = cvGetFileNodeByName( fs, df_elem, "index" );
            if( !index_node )
                CV_ERROR( CV_StsParseError, "index is missing in the decision function" );
            CV_CALL( df[i].sv_index = (int*)cvMemStorageAlloc( storage,
                                            sv_count*sizeof(df[i].sv_index[0])));
            CV_ASSERT( sv_count == 1 || (CV_NODE_IS_SEQ(index_node->tag) &&
                   index_node->data.seq->total == sv_count) );
            CV_CALL( cvReadRawData( fs, index_node, df[i].sv_index, "i" ));
        }
        else
            df[i].sv_index = 0;

        CV_NEXT_SEQ_ELEM( df_node->data.seq->elem_size, reader );
    }

    if( cvReadIntByName(fs, svm_node, "optimize_linear", 1) != 0 )
        optimize_linear_svm();
    create_kernel();

    __END__;
}


typedef struct CvSampleResponsePair
{
    const float* sample;
    const uchar* mask;
    int response;
    int index;
}
CvSampleResponsePair;


static int
CV_CDECL icvCmpSampleResponsePairs( const void* a, const void* b )
{
    int ra = ((const CvSampleResponsePair*)a)->response;
    int rb = ((const CvSampleResponsePair*)b)->response;
    int ia = ((const CvSampleResponsePair*)a)->index;
    int ib = ((const CvSampleResponsePair*)b)->index;

    return ra < rb ? -1 : ra > rb ? 1 : ia - ib;
    //return (ra > rb ? -1 : 0)|(ra < rb);
}

void
cvSortSamplesByClasses( const float** samples, const CvMat* classes,
                       int* class_ranges, const uchar** mask )
{
    CvSampleResponsePair* pairs = 0;
    CV_FUNCNAME( "cvSortSamplesByClasses" );

    __BEGIN__;

    int i, k = 0, sample_count;

    if( !samples || !classes || !class_ranges )
        CV_ERROR( CV_StsNullPtr, "INTERNAL ERROR: some of the args are NULL pointers" );

    if( classes->rows != 1 || CV_MAT_TYPE(classes->type) != CV_32SC1 )
        CV_ERROR( CV_StsBadArg, "classes array must be a single row of integers" );

    sample_count = classes->cols;
    CV_CALL( pairs = (CvSampleResponsePair*)cvAlloc( (sample_count+1)*sizeof(pairs[0])));

    for( i = 0; i < sample_count; i++ )
    {
        pairs[i].sample = samples[i];
        pairs[i].mask = (mask) ? (mask[i]) : 0;
        pairs[i].response = classes->data.i[i];
        pairs[i].index = i;
        assert( classes->data.i[i] >= 0 );
    }

    qsort( pairs, sample_count, sizeof(pairs[0]), icvCmpSampleResponsePairs );
    pairs[sample_count].response = -1;
    class_ranges[0] = 0;

    for( i = 0; i < sample_count; i++ )
    {
        samples[i] = pairs[i].sample;
        if (mask)
            mask[i] = pairs[i].mask;
        classes->data.i[i] = pairs[i].response;

        if( pairs[i].response != pairs[i+1].response )
            class_ranges[++k] = i+1;
    }

    __END__;

    cvFree( &pairs );
}


void
cvPreparePredictData( const CvArr* _sample, int dims_all,
                     const CvMat* comp_idx, int class_count,
                     const CvMat* prob, float** _row_sample,
                     int as_sparse )
{
    float* row_sample = 0;
    int* inverse_comp_idx = 0;

    CV_FUNCNAME( "cvPreparePredictData" );

    __BEGIN__;

    const CvMat* sample = (const CvMat*)_sample;
    float* sample_data;
    int sample_step;
    int is_sparse = CV_IS_SPARSE_MAT(sample);
    int d, sizes[CV_MAX_DIM];
    int i, dims_selected;
    int vec_size;

    if( !is_sparse && !CV_IS_MAT(sample) )
        CV_ERROR( !sample ? CV_StsNullPtr : CV_StsBadArg, "The sample is not a valid vector" );

    if( cvGetElemType( sample ) != CV_32FC1 )
        CV_ERROR( CV_StsUnsupportedFormat, "Input sample must have 32fC1 type" );

    CV_CALL( d = cvGetDims( sample, sizes ));

    if( !((is_sparse && d == 1) || (!is_sparse && d == 2 && (sample->rows == 1 || sample->cols == 1))) )
        CV_ERROR( CV_StsBadSize, "Input sample must be 1-dimensional vector" );

    if( d == 1 )
        sizes[1] = 1;

    if( sizes[0] + sizes[1] - 1 != dims_all )
        CV_ERROR( CV_StsUnmatchedSizes,
                 "The sample size is different from what has been used for training" );

    if( !_row_sample )
        CV_ERROR( CV_StsNullPtr, "INTERNAL ERROR: The row_sample pointer is NULL" );

    if( comp_idx && (!CV_IS_MAT(comp_idx) || comp_idx->rows != 1 ||
                     CV_MAT_TYPE(comp_idx->type) != CV_32SC1) )
        CV_ERROR( CV_StsBadArg, "INTERNAL ERROR: invalid comp_idx" );

    dims_selected = comp_idx ? comp_idx->cols : dims_all;

    if( prob )
    {
        if( !CV_IS_MAT(prob) )
            CV_ERROR( CV_StsBadArg, "The output matrix of probabilities is invalid" );

        if( (prob->rows != 1 && prob->cols != 1) ||
           (CV_MAT_TYPE(prob->type) != CV_32FC1 &&
            CV_MAT_TYPE(prob->type) != CV_64FC1) )
            CV_ERROR( CV_StsBadSize,
                     "The matrix of probabilities must be 1-dimensional vector of 32fC1 type" );

        if( prob->rows + prob->cols - 1 != class_count )
            CV_ERROR( CV_StsUnmatchedSizes,
                     "The vector of probabilities must contain as many elements as "
                     "the number of classes in the training set" );
    }

    vec_size = !as_sparse ? dims_selected*sizeof(row_sample[0]) :
    (dims_selected + 1)*sizeof(CvSparseVecElem32f);

    if( CV_IS_MAT(sample) )
    {
        sample_data = sample->data.fl;
        sample_step = CV_IS_MAT_CONT(sample->type) ? 1 : sample->step/sizeof(row_sample[0]);

        if( !comp_idx && CV_IS_MAT_CONT(sample->type) && !as_sparse )
            *_row_sample = sample_data;
        else
        {
            CV_CALL( row_sample = (float*)cvAlloc( vec_size ));

            if( !comp_idx )
                for( i = 0; i < dims_selected; i++ )
                    row_sample[i] = sample_data[sample_step*i];
            else
            {
                int* comp = comp_idx->data.i;
                for( i = 0; i < dims_selected; i++ )
                    row_sample[i] = sample_data[sample_step*comp[i]];
            }

            *_row_sample = row_sample;
        }

        if( as_sparse )
        {
            const float* src = (const float*)row_sample;
            CvSparseVecElem32f* dst = (CvSparseVecElem32f*)row_sample;

            dst[dims_selected].idx = -1;
            for( i = dims_selected - 1; i >= 0; i-- )
            {
                dst[i].idx = i;
                dst[i].val = src[i];
            }
        }
    }
    else
    {
        CvSparseNode* node;
        CvSparseMatIterator mat_iterator;
        const CvSparseMat* sparse = (const CvSparseMat*)sample;
        assert( is_sparse );

        node = cvInitSparseMatIterator( sparse, &mat_iterator );
        CV_CALL( row_sample = (float*)cvAlloc( vec_size ));

        if( comp_idx )
        {
            CV_CALL( inverse_comp_idx = (int*)cvAlloc( dims_all*sizeof(int) ));
            memset( inverse_comp_idx, -1, dims_all*sizeof(int) );
            for( i = 0; i < dims_selected; i++ )
                inverse_comp_idx[comp_idx->data.i[i]] = i;
        }

        if( !as_sparse )
        {
            memset( row_sample, 0, vec_size );

            for( ; node != 0; node = cvGetNextSparseNode(&mat_iterator) )
            {
                int idx = *CV_NODE_IDX( sparse, node );
                if( inverse_comp_idx )
                {
                    idx = inverse_comp_idx[idx];
                    if( idx < 0 )
                        continue;
                }
                row_sample[idx] = *(float*)CV_NODE_VAL( sparse, node );
            }
        }
        else
        {
            CvSparseVecElem32f* ptr = (CvSparseVecElem32f*)row_sample;
            
            for( ; node != 0; node = cvGetNextSparseNode(&mat_iterator) )
            {
                int idx = *CV_NODE_IDX( sparse, node );
                if( inverse_comp_idx )
                {
                    idx = inverse_comp_idx[idx];
                    if( idx < 0 )
                        continue;
                }
                ptr->idx = idx;
                ptr->val = *(float*)CV_NODE_VAL( sparse, node );
                ptr++;
            }
            
            qsort( row_sample, ptr - (CvSparseVecElem32f*)row_sample,
                  sizeof(ptr[0]), icvCmpSparseVecElems );
            ptr->idx = -1;
        }
        
        *_row_sample = row_sample;
    }
    
    __END__;
    
    if( inverse_comp_idx )
        cvFree( &inverse_comp_idx );
    
    if( cvGetErrStatus() < 0 && _row_sample )
    {
        cvFree( &row_sample );
        *_row_sample = 0;
    }
}


}
}

#endif

/* End of file. */

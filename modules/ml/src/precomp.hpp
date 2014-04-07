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

#ifndef __OPENCV_ML_PRECOMP_HPP__
#define __OPENCV_ML_PRECOMP_HPP__

#include "opencv2/core.hpp"
#include "opencv2/ml.hpp"
#include "opencv2/core/core_c.h"
#include "opencv2/core/utility.hpp"

#include "opencv2/core/private.hpp"

#include <assert.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <vector>

/****************************************************************************************\
 *                               Main struct definitions                                  *
 \****************************************************************************************/

/* log(2*PI) */
#define CV_LOG2PI (1.8378770664093454835606594728112)

/* columns of <trainData> matrix are training samples */
#define CV_COL_SAMPLE 0

/* rows of <trainData> matrix are training samples */
#define CV_ROW_SAMPLE 1

#define CV_TRAIN_ERROR  0
#define CV_TEST_ERROR   1

#define CV_IS_ROW_SAMPLE(flags) ((flags) & CV_ROW_SAMPLE)

#define CV_TYPE_NAME_ML_SVM         "opencv-ml-svm"
#define CV_TYPE_NAME_ML_KNN         "opencv-ml-knn"
#define CV_TYPE_NAME_ML_NBAYES      "opencv-ml-bayesian"
#define CV_TYPE_NAME_ML_EM          "opencv-ml-em"
#define CV_TYPE_NAME_ML_BOOSTING    "opencv-ml-boost-tree"
#define CV_TYPE_NAME_ML_TREE        "opencv-ml-tree"
#define CV_TYPE_NAME_ML_ANN_MLP     "opencv-ml-ann-mlp"
#define CV_TYPE_NAME_ML_CNN         "opencv-ml-cnn"
#define CV_TYPE_NAME_ML_RTREES      "opencv-ml-random-trees"
#define CV_TYPE_NAME_ML_ERTREES     "opencv-ml-extremely-randomized-trees"
#define CV_TYPE_NAME_ML_GBT         "opencv-ml-gradient-boosting-trees"

struct CvVectors
{
    int type;
    int dims, count;
    CvVectors* next;
    union
    {
        uchar** ptr;
        float** fl;
        double** db;
    } data;
};

struct CV_EXPORTS_W_MAP CvParamGrid
{
    // SVM params type
    enum { SVM_C=0, SVM_GAMMA=1, SVM_P=2, SVM_NU=3, SVM_COEF=4, SVM_DEGREE=5 };

    CvParamGrid()
    {
        min_val = max_val = step = 0;
    }

    CvParamGrid( double min_val, double max_val, double log_step );
    //CvParamGrid( int param_id );
    bool check() const;

    CV_PROP_RW double min_val;
    CV_PROP_RW double max_val;
    CV_PROP_RW double step;
};

inline CvParamGrid::CvParamGrid( double _min_val, double _max_val, double _log_step )
{
    min_val = _min_val;
    max_val = _max_val;
    step = _log_step;
}

struct CvPair16u32s
{
    unsigned short* u;
    int* i;
};

#define CV_MAT_ELEM_FLAG( mat, type, comp, vect, tflag )    \
    (( tflag == CV_ROW_SAMPLE )                             \
    ? (CV_MAT_ELEM( mat, type, comp, vect ))                \
    : (CV_MAT_ELEM( mat, type, vect, comp )))

/* Convert matrix to vector */
#define ICV_MAT2VEC( mat, vdata, vstep, num )      \
    if( MIN( (mat).rows, (mat).cols ) != 1 )       \
        CV_ERROR( CV_StsBadArg, "" );              \
    (vdata) = ((mat).data.ptr);                    \
    if( (mat).rows == 1 )                          \
    {                                              \
        (vstep) = CV_ELEM_SIZE( (mat).type );      \
        (num) = (mat).cols;                        \
    }                                              \
    else                                           \
    {                                              \
        (vstep) = (mat).step;                      \
        (num) = (mat).rows;                        \
    }

/* get raw data */
#define ICV_RAWDATA( mat, flags, rdata, sstep, cstep, m, n )         \
    (rdata) = (mat).data.ptr;                                        \
    if( CV_IS_ROW_SAMPLE( flags ) )                                  \
    {                                                                \
        (sstep) = (mat).step;                                        \
        (cstep) = CV_ELEM_SIZE( (mat).type );                        \
        (m) = (mat).rows;                                            \
        (n) = (mat).cols;                                            \
    }                                                                \
    else                                                             \
    {                                                                \
        (cstep) = (mat).step;                                        \
        (sstep) = CV_ELEM_SIZE( (mat).type );                        \
        (n) = (mat).rows;                                            \
        (m) = (mat).cols;                                            \
    }

#define ICV_IS_MAT_OF_TYPE( mat, mat_type) \
    (CV_IS_MAT( mat ) && CV_MAT_TYPE( mat->type ) == (mat_type) &&   \
    (mat)->cols > 0 && (mat)->rows > 0)

/*
    uchar* data; int sstep, cstep;      - trainData->data
    uchar* classes; int clstep; int ncl;- trainClasses
    uchar* tmask; int tmstep; int ntm;  - typeMask
    uchar* missed;int msstep, mcstep;   -missedMeasurements...
    int mm, mn;                         == m,n == size,dim
    uchar* sidx;int sistep;             - sampleIdx
    uchar* cidx;int cistep;             - compIdx
    int k, l;                           == n,m == dim,size (length of cidx, sidx)
    int m, n;                           == size,dim
*/
#define ICV_DECLARE_TRAIN_ARGS()                                                    \
    uchar* data;                                                                    \
    int sstep, cstep;                                                               \
    uchar* classes;                                                                 \
    int clstep;                                                                     \
    int ncl;                                                                        \
    uchar* tmask;                                                                   \
    int tmstep;                                                                     \
    int ntm;                                                                        \
    uchar* missed;                                                                  \
    int msstep, mcstep;                                                             \
    int mm, mn;                                                                     \
    uchar* sidx;                                                                    \
    int sistep;                                                                     \
    uchar* cidx;                                                                    \
    int cistep;                                                                     \
    int k, l;                                                                       \
    int m, n;                                                                       \
                                                                                    \
    data = classes = tmask = missed = sidx = cidx = NULL;                           \
    sstep = cstep = clstep = ncl = tmstep = ntm = msstep = mcstep = mm = mn = 0;    \
    sistep = cistep = k = l = m = n = 0;

#define ICV_TRAIN_DATA_REQUIRED( param, flags )                                     \
    if( !ICV_IS_MAT_OF_TYPE( (param), CV_32FC1 ) )                                  \
    {                                                                               \
        CV_ERROR( CV_StsBadArg, "Invalid " #param " parameter" );                   \
    }                                                                               \
    else                                                                            \
    {                                                                               \
        ICV_RAWDATA( *(param), (flags), data, sstep, cstep, m, n );                 \
        k = n;                                                                      \
        l = m;                                                                      \
    }

#define ICV_TRAIN_CLASSES_REQUIRED( param )                                         \
    if( !ICV_IS_MAT_OF_TYPE( (param), CV_32FC1 ) )                                  \
    {                                                                               \
        CV_ERROR( CV_StsBadArg, "Invalid " #param " parameter" );                   \
    }                                                                               \
    else                                                                            \
    {                                                                               \
        ICV_MAT2VEC( *(param), classes, clstep, ncl );                              \
        if( m != ncl )                                                              \
        {                                                                           \
            CV_ERROR( CV_StsBadArg, "Unmatched sizes" );                            \
        }                                                                           \
    }

#define ICV_ARG_NULL( param )                                                       \
    if( (param) != NULL )                                                           \
    {                                                                               \
        CV_ERROR( CV_StsBadArg, #param " parameter must be NULL" );                 \
    }

#define ICV_MISSED_MEASUREMENTS_OPTIONAL( param, flags )                            \
    if( param )                                                                     \
    {                                                                               \
        if( !ICV_IS_MAT_OF_TYPE( param, CV_8UC1 ) )                                 \
        {                                                                           \
            CV_ERROR( CV_StsBadArg, "Invalid " #param " parameter" );               \
        }                                                                           \
        else                                                                        \
        {                                                                           \
            ICV_RAWDATA( *(param), (flags), missed, msstep, mcstep, mm, mn );       \
            if( mm != m || mn != n )                                                \
            {                                                                       \
                CV_ERROR( CV_StsBadArg, "Unmatched sizes" );                        \
            }                                                                       \
        }                                                                           \
    }

#define ICV_COMP_IDX_OPTIONAL( param )                                              \
    if( param )                                                                     \
    {                                                                               \
        if( !ICV_IS_MAT_OF_TYPE( param, CV_32SC1 ) )                                \
        {                                                                           \
            CV_ERROR( CV_StsBadArg, "Invalid " #param " parameter" );               \
        }                                                                           \
        else                                                                        \
        {                                                                           \
            ICV_MAT2VEC( *(param), cidx, cistep, k );                               \
            if( k > n )                                                             \
                CV_ERROR( CV_StsBadArg, "Invalid " #param " parameter" );           \
        }                                                                           \
    }

#define ICV_SAMPLE_IDX_OPTIONAL( param )                                            \
    if( param )                                                                     \
    {                                                                               \
        if( !ICV_IS_MAT_OF_TYPE( param, CV_32SC1 ) )                                \
        {                                                                           \
            CV_ERROR( CV_StsBadArg, "Invalid " #param " parameter" );               \
        }                                                                           \
        else                                                                        \
        {                                                                           \
            ICV_MAT2VEC( *sampleIdx, sidx, sistep, l );                             \
            if( l > m )                                                             \
                CV_ERROR( CV_StsBadArg, "Invalid " #param " parameter" );           \
        }                                                                           \
    }

/****************************************************************************************/
#define ICV_CONVERT_FLOAT_ARRAY_TO_MATRICE( array, matrice )        \
{                                                                   \
    CvMat a, b;                                                     \
    int dims = (matrice)->cols;                                     \
    int nsamples = (matrice)->rows;                                 \
    int type = CV_MAT_TYPE((matrice)->type);                        \
    int i, offset = dims;                                           \
                                                                    \
    CV_ASSERT( type == CV_32FC1 || type == CV_64FC1 );              \
    offset *= ((type == CV_32FC1) ? sizeof(float) : sizeof(double));\
                                                                    \
    b = cvMat( 1, dims, CV_32FC1 );                                 \
    cvGetRow( matrice, &a, 0 );                                     \
    for( i = 0; i < nsamples; i++, a.data.ptr += offset )           \
    {                                                               \
        b.data.fl = (float*)array[i];                               \
        CV_CALL( cvConvert( &b, &a ) );                             \
    }                                                               \
}

/****************************************************************************************\
*                       Auxiliary functions declarations                                 *
\****************************************************************************************/

/* Generates a set of classes centers in quantity <num_of_clusters> that are generated as
   uniform random vectors in parallelepiped, where <data> is concentrated. Vectors in
   <data> should have horizontal orientation. If <centers> != NULL, the function doesn't
   allocate any memory and stores generated centers in <centers>, returns <centers>.
   If <centers> == NULL, the function allocates memory and creates the matrice. Centers
   are supposed to be oriented horizontally. */
CvMat* icvGenerateRandomClusterCenters( int seed,
                                        const CvMat* data,
                                        int num_of_clusters,
                                        CvMat* centers CV_DEFAULT(0));

/* Fills the <labels> using <probs> by choosing the maximal probability. Outliers are
   fixed by <oulier_tresh> and have cluster label (-1). Function also controls that there
   weren't "empty" clusters by filling empty clusters with the maximal probability vector.
   If probs_sums != NULL, filles it with the sums of probabilities for each sample (it is
   useful for normalizing probabilities' matrice of FCM) */
void icvFindClusterLabels( const CvMat* probs, float outlier_thresh, float r,
                           const CvMat* labels );

typedef struct CvSparseVecElem32f
{
    int idx;
    float val;
}
CvSparseVecElem32f;

/* Prepare training data and related parameters */
#define CV_TRAIN_STATMODEL_DEFRAGMENT_TRAIN_DATA    1
#define CV_TRAIN_STATMODEL_SAMPLES_AS_ROWS          2
#define CV_TRAIN_STATMODEL_SAMPLES_AS_COLUMNS       4
#define CV_TRAIN_STATMODEL_CATEGORICAL_RESPONSE     8
#define CV_TRAIN_STATMODEL_ORDERED_RESPONSE         16
#define CV_TRAIN_STATMODEL_RESPONSES_ON_OUTPUT      32
#define CV_TRAIN_STATMODEL_ALWAYS_COPY_TRAIN_DATA   64
#define CV_TRAIN_STATMODEL_SPARSE_AS_SPARSE         128

int
cvPrepareTrainData( const char* /*funcname*/,
                    const CvMat* train_data, int tflag,
                    const CvMat* responses, int response_type,
                    const CvMat* var_idx,
                    const CvMat* sample_idx,
                    bool always_copy_data,
                    const float*** out_train_samples,
                    int* _sample_count,
                    int* _var_count,
                    int* _var_all,
                    CvMat** out_responses,
                    CvMat** out_response_map,
                    CvMat** out_var_idx,
                    CvMat** out_sample_idx=0 );

void
cvSortSamplesByClasses( const float** samples, const CvMat* classes,
                        int* class_ranges, const uchar** mask CV_DEFAULT(0) );

void
cvCombineResponseMaps (CvMat*  _responses,
                 const CvMat*  old_response_map,
                       CvMat*  new_response_map,
                       CvMat** out_response_map);

void
cvPreparePredictData( const CvArr* sample, int dims_all, const CvMat* comp_idx,
                      int class_count, const CvMat* prob, float** row_sample,
                      int as_sparse CV_DEFAULT(0) );

/* copies clustering [or batch "predict"] results
   (labels and/or centers and/or probs) back to the output arrays */
void
cvWritebackLabels( const CvMat* labels, CvMat* dst_labels,
                   const CvMat* centers, CvMat* dst_centers,
                   const CvMat* probs, CvMat* dst_probs,
                   const CvMat* sample_idx, int samples_all,
                   const CvMat* comp_idx, int dims_all );
#define cvWritebackResponses cvWritebackLabels

#define XML_FIELD_NAME "_name"
CvFileNode* icvFileNodeGetChild(CvFileNode* father, const char* name);
CvFileNode* icvFileNodeGetChildArrayElem(CvFileNode* father, const char* name,int index);
CvFileNode* icvFileNodeGetNext(CvFileNode* n, const char* name);


void cvCheckTrainData( const CvMat* train_data, int tflag,
                       const CvMat* missing_mask,
                       int* var_all, int* sample_all );

CvMat* cvPreprocessIndexArray( const CvMat* idx_arr, int data_arr_size, bool check_for_duplicates=false );

CvMat* cvPreprocessVarType( const CvMat* type_mask, const CvMat* var_idx,
                            int var_all, int* response_type );

CvMat* cvPreprocessOrderedResponses( const CvMat* responses,
                const CvMat* sample_idx, int sample_all );

CvMat* cvPreprocessCategoricalResponses( const CvMat* responses,
                const CvMat* sample_idx, int sample_all,
                CvMat** out_response_map, CvMat** class_counts=0 );

const float** cvGetTrainSamples( const CvMat* train_data, int tflag,
                   const CvMat* var_idx, const CvMat* sample_idx,
                   int* _var_count, int* _sample_count,
                   bool always_copy_data=false );

namespace cv
{
namespace ml
{
    #define CV_DTREE_CAT_DIR(idx,subset) \
        (2*((subset[(idx)>>5]&(1 << ((idx) & 31)))==0)-1)

    struct CvDTreeSplit
    {
        int var_idx;
        int condensed_idx;
        int inversed;
        float quality;
        CvDTreeSplit* next;
        union
        {
            int subset[2];
            struct
            {
                float c;
                int split_point;
            }
            ord;
        };
    };

    struct CvDTreeNode
    {
        int class_idx;
        int Tn;
        double value;

        CvDTreeNode* parent;
        CvDTreeNode* left;
        CvDTreeNode* right;

        CvDTreeSplit* split;

        int sample_count;
        int depth;
        int* num_valid;
        int offset;
        int buf_idx;
        double maxlr;

        // global pruning data
        int complexity;
        double alpha;
        double node_risk, tree_risk, tree_error;

        // cross-validation pruning data
        int* cv_Tn;
        double* cv_node_risk;
        double* cv_node_error;
        
        int get_num_valid(int vi) { return num_valid ? num_valid[vi] : sample_count; }
        void set_num_valid(int vi, int n) { if( num_valid ) num_valid[vi] = n; }
    };

#if 0
    struct DTreeBestSplitFinder
    {
        DTreeBestSplitFinder(){ node = 0; }
        DTreeBestSplitFinder( const Ptr<DTree>& _tree, int _node);
        DTreeBestSplitFinder( const DTreeBestSplitFinder& finder, Split );
        virtual ~DTreeBestSplitFinder() {}
        virtual void operator()(const Range& range);
        void join( DTreeBestSplitFinder& rhs );
        DTree::Split bestSplit;
        DTree::Split split;
        int splitSize;
        Ptr<DTree> tree;
        int node;
    };

    struct ForestTreeBestSplitFinder : DTreeBestSplitFinder
    {
        ForestTreeBestSplitFinder() : DTreeBestSplitFinder() {}
        ForestTreeBestSplitFinder( CvForestTree* _tree, CvDTreeNode* _node );
        ForestTreeBestSplitFinder( const ForestTreeBestSplitFinder& finder, Split );
        virtual void operator()(const BlockedRange& range);
    };
#endif

    static inline void setRangeVector(std::vector<int>& vec, int n)
    {
        vec.resize(n);
        for( int i = 0; i < n; i++ )
            vec[i] = i;
    }
}}

#endif /* __OPENCV_ML_PRECOMP_HPP__ */

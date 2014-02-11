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
#include <ctype.h>

namespace cv { namespace ml {

static const float MISSED_VAL = FLT_MAX;
static const int VAR_MISSED = VAR_ORDERED;

TrainData::Params::Params()
{
    headerLines = 0;
    responseIdx = -1;
    delimiter = ',';
    missch = '?';
}

TrainData::Params::Params(int _headerLines, int _responseIdx, char _delimiter, char _missch, const String& _varTypeSpec)
{
    headerLines = _headerLines;
    responseIdx = _responseIdx;
    delimiter = _delimiter;
    missch = _missch;
    varTypeSpec = _varTypeSpec;
}

TrainData::~TrainData() {}

class TrainDataImpl : public TrainData
{
public:
    TrainDataImpl()
    {
        clear();
    }

    TrainDataImpl(const Params& p)
    {
        params = p;
        clear();
    }

    virtual ~TrainDataImpl() { closeFile(); }

    bool getTFlag() const { return tflag; }
    Mat getSamples() const { return samples; }
    Mat getResponses() { return responses; }
    Mat getMissing() const { return missing; }
    Mat getVarIdx() const { return varIdx; }
    Mat getVarType() const { return varType; }
    Mat getTrainSampleIdx() const { return !trainSampleIdx.empty() ? trainSampleIdx : sampleIdx; }
    Mat getTestSampleIdx() const { return testSampleIdx; }
    Mat getNormCatResponses() const { return normCatResponses; }
    Mat getClassLabels() const { return classLabels; }
    Mat getClassCounters() const { return classCounters; }

    Params getParams() const { return params; }

    void closeFile() { if(file) fclose(file); file=0; }
    void clear()
    {
        closeFile();
        samples.release();
        missing.release();
        varType.release();
        responses.release();
        sampleIdx.release();
        trainSampleIdx.release();
        testSampleIdx.release();
        normCatResponses.release();
        classLabels.release();
        classCounters.release();
        rng = RNG(-1);
        totalClassCount = 0;
        tflag = false;
    }

    void setData(InputArray _samples, bool _tflag, InputArray _responses,
                 InputArray _varIdx, InputArray _sampleIdx,
                 InputArray _varType, InputArray _missing)
    {
        clear();
        samples = _samples.getMat();
        tflag = _tflag;
        responses = _responses.getMat();
        varIdx = _varIdx.getMat();
        sampleIdx = _sampleIdx.getMat();
        varType = _varType.getMat();
        missing = _missing.getMat();

        int nsamples = tflag ? samples.cols : samples.rows;
        int nvars = tflag ? samples.rows : samples.cols;

        CV_Assert( samples.type() == CV_32F || samples.type() == CV_32S );

        if( !sampleIdx.empty() )
        {
            CV_Assert( (sampleIdx.checkVector(1, CV_32S, true) > 0 &&
                       checkRange(sampleIdx, true, 0, 0, nsamples-1)) ||
                       sampleIdx.checkVector(1, CV_8U, true) == nsamples );
            if( sampleIdx.type() == CV_8U )
                sampleIdx = convertMaskToIdx(sampleIdx);
        }

        if( !varIdx.empty() )
        {
            CV_Assert( (varIdx.checkVector(1, CV_32S, true) > 0 &&
                       checkRange(varIdx, true, 0, 0, nvars-1)) ||
                       varIdx.checkVector(1, CV_8U, true) == nvars );
            if( varIdx.type() == CV_8U )
                varIdx = convertMaskToIdx(varIdx);
        }

        if( !varType.empty() )
        {
            CV_Assert( varType.checkVector(1, CV_8U, true) == nvars+1 &&
                       checkRange(varType, true, 0, VAR_ORDERED, VAR_CATEGORICAL) );
        }
        else
        {
            varType.create(1, nvars+1, CV_8U);
            varType = Scalar::all(VAR_ORDERED);
            varType.at<uchar>(nvars) = responses.type() < CV_32F ? VAR_CATEGORICAL : VAR_ORDERED;
        }

        if( !responses.empty() )
        {
            CV_Assert( responses.checkVector(1, CV_32S, true) == nsamples ||
                       responses.checkVector(1, CV_32F, true) == nsamples );
        }

        if( varType.at<uchar>(nvars) == VAR_CATEGORICAL )
            preprocessCategorical(responses, normCatResponses, classLabels, classCounters);

        if( !missing.empty() )
        {
            CV_Assert( missing.size() == samples.size() && missing.type() == CV_8U );
        }
    }

    Mat convertMaskToIdx(const Mat& mask)
    {
        int i, j, nz = countNonZero(mask), n = mask.cols + mask.rows - 1;
        Mat idx(1, nz, CV_32S);
        for( i = j = 0; i < n; i++ )
            if( mask.at<uchar>(i) )
                idx.at<int>(j++) = i;
        return idx;
    }

    struct CmpByIdx
    {
        CmpByIdx(const int* _data, int _step) : data(_data), step(_step) {}
        bool operator ()(int i, int j) const { return data[i*step] < data[j*step]; }
        const int* data;
        int step;
    };

    void preprocessCategorical(const Mat& data, Mat& normdata, Mat& labels, Mat& counters)
    {
        CV_Assert((data.cols == 1 || data.rows == 1) && (data.type() == CV_32S || data.type() == CV_32F));
        normdata.create(data.size(), CV_32S);

        int i, n = data.cols + data.rows - 1;
        cv::AutoBuffer<int> ibuf(n*2);
        int* idx = ibuf;
        int* idata = (int*)data.ptr<int>();
        int istep = (int)(data.step/sizeof(int));
        int* odata = normdata.ptr<int>();
        int ostep = (int)(normdata.step/sizeof(int));

        if( data.type() == CV_32F )
        {
            idata = idx + n;
            const float* fdata = data.ptr<float>();
            for( i = 0; i < n; i++ )
            {
                idata[i] = cvRound(fdata[i*istep]);
                CV_Assert( (float)idata[i] == fdata[i*istep] );
            }
            istep = 1;
        }

        for( i = 0; i < n; i++ )
            idx[i] = i;

        std::sort(idx, idx + n, CmpByIdx(idata, istep));

        int clscount = 0;
        for( i = 1; i < n; i++ )
            clscount += idata[idx[i]*istep] != idata[idx[i-1]*istep];

        int clslabel = -1;
        int prev = ~idata[idx[0]*istep];
        int previdx = 0;

        labels.create(1, clscount, CV_32S);
        counters.create(1, clscount, CV_32S);

        for( i = 0; i < n; i++ )
        {
            int l = idata[idx[i]*istep];
            if( l != prev )
            {
                clslabel++;
                labels.at<int>(clslabel) = l;
                int k = i - previdx;
                if( clslabel > 0 )
                    counters.at<int>(clslabel-1) = k;
                prev = l;
                previdx = i;
            }
            odata[i*ostep] = clslabel;
        }
        counters.at<int>(clslabel) = i - previdx;
    }

    bool loadCSV(const String& filename)
    {
        const int M = 1000000;
        const char delimiters[3] = { ' ', params.delimiter, '\0' };
        int nvars = 0;
        bool varTypesSet = false;

        clear();

        file = fopen( filename.c_str(), "rt" );

        if( !file )
            return false;

        std::vector<char> _buf(M);
        std::vector<float> allresponses;
        std::vector<float> rowvals;
        std::vector<uchar> vtypes, rowtypes;
        bool haveMissed = false;
        char* buf = &_buf[0];

        int i, ridx = params.responseIdx, ninputvars = 0;

        samples.release();

        // skip header lines
        int lineno = 0;
        for(;;lineno++)
        {
            if( !fgets(buf, M, file) )
                break;
            if(lineno < params.headerLines )
                continue;
            // trim trailing spaces
            int idx = (int)strlen(buf)-1;
            while( idx >= 0 && isspace(buf[idx]) )
                buf[idx--] = '\0';
            // skip spaces in the beginning
            char* ptr = buf;
            while( *ptr != '\0' && isspace(*ptr) )
                ptr++;
            // skip commented off lines
            if(*ptr == '#')
                continue;
            rowvals.clear();
            rowtypes.clear();

            char* token = strtok(buf, delimiters);
            if (!token)
                break;

            for(;;)
            {
                float val=0.f; int tp = 0;
                decodeElem( token, val, tp );
                if( tp == VAR_MISSED )
                    haveMissed = true;
                rowvals.push_back(val);
                rowtypes.push_back(tp);
                token = strtok(NULL, delimiters);
                if (!token)
                    break;
            }

            if( nvars == 0 )
            {
                if( rowvals.empty() )
                    CV_Error(CV_StsBadArg, "invalid CSV format; no data found");
                nvars = (int)rowvals.size();
                if( !params.varTypeSpec.empty() && params.varTypeSpec.size() > 0 )
                {
                    setVarTypes(params.varTypeSpec, nvars, vtypes);
                    varTypesSet = true;
                }
                else
                    vtypes = rowtypes;

                ridx = ridx >= 0 ? ridx : ridx == -1 ? nvars - 1 : -1;
                ninputvars = nvars - (ridx >= 0);
            }
            else
                CV_Assert( nvars == (int)rowvals.size() );

            // check var types
            for( i = 0; i < nvars; i++ )
            {
                CV_Assert( (!varTypesSet && vtypes[i] == rowtypes[i]) ||
                           (varTypesSet && (vtypes[i] == rowtypes[i] || rowtypes[i] == VAR_ORDERED)) );
            }

            if( ridx >= 0 )
            {
                for( i = ridx; i < nvars-1; i++ )
                    std::swap(rowvals[i], rowvals[i+1]);
                float rval = rowvals[nvars-1];
                allresponses.push_back(rval);
                rowvals.pop_back();
            }
            Mat rmat(1, ninputvars, CV_32F, &rowvals[0]);
            samples.push_back(rmat);
        }

        closeFile();

        int nsamples = samples.rows;
        if( nsamples == 0 )
            return false;

        if( haveMissed )
            compare(samples, MISSED_VAL, missing, CMP_EQ);

        if( ridx >= 0 )
        {
            for( i = ridx; i < nvars-1; i++ )
                std::swap(vtypes[i], vtypes[i+1]);
        }

        if( !varTypesSet && vtypes[ninputvars] == VAR_ORDERED )
        {
            for( i = 0; i < nsamples; i++ )
                if( allresponses[i] != cvRound(allresponses[i]) )
                    break;
            if( i == nsamples )
                vtypes[ninputvars] = VAR_CATEGORICAL;
        }

        Mat(allresponses).copyTo(responses);
        Mat(vtypes).copyTo(varType);

        if( vtypes[ninputvars] == VAR_CATEGORICAL )
            preprocessCategorical(responses, normCatResponses, classLabels, classCounters);

        return true;
    }

    void decodeElem( const char* token, float& elem, int& type)
    {
        char* stopstring = NULL;
        elem = (float)strtod( token, &stopstring );
        if( *stopstring == params.missch && strlen(stopstring) == 1 ) // missed value
        {
            elem = MISSED_VAL;
            type = VAR_MISSED;
        }
        else if( *stopstring != '\0' )
        {
            MapType::iterator it = classMap.find(token);
            if( it == classMap.end() )
            {
                classMap[token] = ++totalClassCount;
                elem = (float)totalClassCount;
            }
            else
                elem = (float)it->second;
            type = VAR_CATEGORICAL;
        }
        else
            type = VAR_ORDERED;
    }

    void setVarTypes( const String& s, int nvars, std::vector<uchar>& vtypes )
    {
        const char* errmsg = "type spec is not correct; it should have format \"cat\", \"ord\" or "
          "\"ord[n1,n2-n3,n4-n5,...]cat[m1-m2,m3,m4-m5,...]\", where n's and m's are 0-based variable indices";
        const char* str = s.c_str();
        int specCounter = 0;

        vtypes.resize(nvars);

        for( int k = 0; k < 2; k++ )
        {
            const char* ptr = strstr(str, k == 0 ? "ord" : "cat");
            int tp = k == 0 ? VAR_ORDERED : VAR_CATEGORICAL;
            if( ptr ) // parse ord/cat str
            {
                char* stopstring = NULL;

                if( ptr[3] == '\0' )
                {
                    for( int i = 0; i < nvars; i++ )
                        vtypes[i] = (uchar)tp;
                    specCounter = nvars;
                    break;
                }

                if ( ptr[3] != '[')
                    CV_Error( CV_StsBadArg, errmsg );

                ptr += 4; // pass "ord["
                do
                {
                    int b1 = (int)strtod( ptr, &stopstring );
                    if( *stopstring == 0 || (*stopstring != ',' && *stopstring != ']' && *stopstring != '-') )
                        CV_Error( CV_StsBadArg, errmsg );
                    ptr = stopstring + 1;
                    if( (stopstring[0] == ',') || (stopstring[0] == ']'))
                    {
                        CV_Assert( 0 <= b1 && b1 < nvars );
                        vtypes[b1] = (uchar)tp;
                        specCounter++;
                    }
                    else
                    {
                        if( stopstring[0] == '-')
                        {
                            int b2 = (int)strtod( ptr, &stopstring);
                            if ( (*stopstring == 0) || (*stopstring != ',' && *stopstring != ']') )
                                CV_Error( CV_StsBadArg, errmsg );
                            ptr = stopstring + 1;
                            CV_Assert( 0 <= b1 && b1 <= b2 && b2 < nvars );
                            for (int i = b1; i <= b2; i++)
                                varType.at<uchar>(i) = (uchar)tp;
                            specCounter += b2 - b1 + 1;
                        }
                        else
                            CV_Error( CV_StsBadArg, errmsg );

                    }
                }
                while(*stopstring != ']');

                if( stopstring[1] != '\0' && stopstring[1] != ',')
                    CV_Error( CV_StsBadArg, errmsg );
            }
        }

        if( specCounter != nvars )
            CV_Error( CV_StsBadArg, "type of some variables is not specified" );
    }

    int getNSamples() const
    {
        return !sampleIdx.empty() ? (sampleIdx.cols + sampleIdx.rows - 1) : getTotalSamples();
    }

    int getTotalSamples() const
    {
        return tflag ? samples.rows : samples.cols;
    }

    void setTrainTestSplitRatio(float ratio, bool shuffle)
    {
        CV_Assert( 0 <= ratio && ratio <= 1 );
        setTrainTestSplit(cvRound(getNSamples()*ratio), shuffle);
    }

    void setTrainTestSplit(int count, bool shuffle)
    {
        int i, nsamples = getNSamples();
        CV_Assert( 0 <= count < nsamples );

        trainSampleIdx.release();
        testSampleIdx.release();

        if( count == 0 )
            trainSampleIdx = sampleIdx;
        else if( count == nsamples )
            testSampleIdx = sampleIdx;
        else
        {
            Mat mask(1, nsamples, CV_8U);
            uchar* mptr = mask.data;
            for( i = 0; i < nsamples; i++ )
                mptr[i] = (uchar)(i < count);
            if( shuffle )
            {
                for( i = 0; i < nsamples; i++)
                {
                    int a = rng.uniform(0, nsamples);
                    int b = rng.uniform(0, nsamples);
                    std::swap(mptr[a], mptr[b]);
                }
            }
            trainSampleIdx.create(1, count, CV_32S);
            testSampleIdx.create(1, nsamples - count, CV_32S);
            int j0 = 0, j1 = 0;
            const int* sptr = !sampleIdx.empty() ? sampleIdx.ptr<int>() : 0;
            int* trainptr = trainSampleIdx.ptr<int>();
            int* testptr = testSampleIdx.ptr<int>();
            for( i = 0; i < nsamples; i++ )
            {
                int idx = sptr ? sptr[i] : i;
                if( mptr[i] )
                    trainptr[j0++] = idx;
                else
                    testptr[j1++] = idx;
            }
        }
    }

    FILE* file;
    bool tflag;
    Params params;
    Mat samples, missing, varType, varIdx, responses;
    Mat sampleIdx, trainSampleIdx, testSampleIdx;
    Mat normCatResponses, classLabels, classCounters;
    RNG rng;
    typedef std::map<String, int> MapType;
    MapType classMap;
    int totalClassCount;
};


Ptr<TrainData> loadDataFromCSV(const String& filename, const TrainData::Params& params)
{
    Ptr<TrainDataImpl> td = makePtr<TrainDataImpl>(params);
    if(!td->loadCSV(filename))
        td.release();
    return td;
}

Ptr<TrainData> createTrainData(InputArray _samples, bool _tflag, InputArray _responses,
                               InputArray _varIdx, InputArray _sampleIdx,
                               InputArray _varType, InputArray _missing)
{
    Ptr<TrainDataImpl> td = makePtr<TrainDataImpl>();
    td->setData(_samples, _tflag, _responses, _varIdx, _sampleIdx, _varType, _missing);
    return td;
}

}}

/* End of file. */

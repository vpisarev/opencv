// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "layers_common.hpp"
#include "../net_impl.hpp"
#include "conv2_common.hpp"
#include "opencv2/core/hal/intrin.hpp"

namespace cv
{
namespace dnn
{

static void global_average_pool_32f(const void* inp_, const MatShape& shape, void* out_)
{
    CV_Assert(shape.layout == DATA_LAYOUT_BLOCK);
    int ndims = shape.dims;
    int N = shape[0], C1 = shape[1], C0_ = shape[ndims-1];
    int nlanes_ = (int)VTraits<v_float32>::vlanes();
    CV_Assert(C0_ == nlanes_ || C0_ == nlanes_*2 || C0_ % (nlanes_*4) == 0);
    int planesize = 1;
    N *= C1;
    for (int i = 2; i < ndims-1; i++)
        planesize *= shape[i];

    parallel_for_(Range(0, (int)N), [&](const Range& r) {
        v_float32 scale = vx_setall_f32(planesize > 0 ? (float)(1./planesize) : 0);
        int n0 = r.start, n1 = r.end;
        int nlanes = nlanes_, C0 = (int)C0_;
        for (int n = n0; n < n1; n++) {
            const float* inp = (const float*)inp_ + planesize*C0*n;
            float* out = (float*)out_ + C0*n;
            int planesize_C0 = planesize*C0;
            // by computing sum in blocks we probably increase accuracy
            int BLOCK_SIZE = 256*C0, blocksize = 0;
            if (nlanes == C0) {
                v_float32 s0 = vx_setzero_f32();
                for (int i = 0; i < planesize_C0; i += blocksize, inp += blocksize) {
                    blocksize = std::min(planesize_C0 - i, BLOCK_SIZE);
                    v_float32 bs0 = vx_setzero_f32();

                    for (int j = 0; j < blocksize; j += C0)
                        bs0 = v_add(bs0, vx_load(inp + j));
                    s0 = v_add(s0, bs0);
                }
                s0 = v_mul(s0, scale);
                v_store(out, s0);
            } else if (nlanes*2 == C0) {
                v_float32 s0 = vx_setzero_f32(), s1 = s0;
                for (int i = 0; i < planesize_C0; i += blocksize, inp += blocksize) {
                    blocksize = std::min(planesize_C0 - i, BLOCK_SIZE);
                    v_float32 bs0 = vx_setzero_f32(), bs1 = bs0;

                    for (int j = 0; j < blocksize; j += C0) {
                        bs0 = v_add(bs0, vx_load(inp + j));
                        bs1 = v_add(bs1, vx_load(inp + j + nlanes));
                    }

                    s0 = v_add(s0, bs0);
                    s1 = v_add(s1, bs1);
                }
                s0 = v_mul(s0, scale);
                s1 = v_mul(s1, scale);
                v_store(out, s0);
                v_store(out + nlanes, s1);
            } else {
                memset(out, 0, C0*sizeof(out[0]));
                for (int i = 0; i < planesize_C0; i += blocksize, inp += blocksize) {
                    blocksize = std::min(planesize_C0 - i, BLOCK_SIZE);
                    for (int c = 0; c < C0; c += nlanes*4) {
                        v_float32 s0 = vx_load(out + c);
                        v_float32 s1 = vx_load(out + c + nlanes);
                        v_float32 s2 = vx_load(out + c + nlanes*2);
                        v_float32 s3 = vx_load(out + c + nlanes*3);

                        v_float32 bs0 = vx_setzero_f32(), bs1 = bs0, bs2 = bs0, bs3 = bs0;

                        for (int j = 0; j < blocksize; j += C0) {
                            bs0 = v_add(bs0, vx_load(inp + c + j));
                            bs1 = v_add(bs1, vx_load(inp + c + nlanes + j));
                            bs2 = v_add(bs2, vx_load(inp + c + nlanes*2 + j));
                            bs3 = v_add(bs3, vx_load(inp + c + nlanes*3 + j));
                        }

                        s0 = v_add(s0, bs0);
                        s1 = v_add(s1, bs1);
                        s2 = v_add(s2, bs2);
                        s3 = v_add(s3, bs3);
                        vx_store(out + c, s0);
                        vx_store(out + c + nlanes, s1);
                        vx_store(out + c + nlanes*2, s2);
                        vx_store(out + c + nlanes*3, s3);
                    }
                }
                for (int c = 0; c < C0; c += nlanes*2) {
                    v_float32 s0 = vx_load(out + c);
                    v_float32 s1 = vx_load(out + c + nlanes);
                    s0 = v_mul(s0, scale);
                    s1 = v_mul(s1, scale);
                    vx_store(out + c, s0);
                    vx_store(out + c + nlanes, s1);
                }
            }
        }
    });
}

template<typename _Tp>
void global_average_pool_16(const _Tp* inp_, const MatShape& shape, _Tp* out_)
{
    CV_Assert(shape.layout == DATA_LAYOUT_BLOCK);
    int ndims = shape.dims;
    int N = shape[0], C1 = shape[1], C0_ = shape[ndims-1];
    int nlanes_ = (int)VTraits<v_float32>::vlanes();
    CV_Assert(C0_ == nlanes_ || C0_ == nlanes_*2 || C0_ % (nlanes_*4) == 0);
    int planesize = 1;
    N *= C1;
    for (int i = 2; i < ndims-1; i++)
        planesize *= shape[i];

    parallel_for_(Range(0, N), [&](const Range& r) {
        v_float32 scale = vx_setall_f32(planesize > 0 ? (float)(1./planesize) : 0);
        int n0 = r.start, n1 = r.end;
        int nlanes = nlanes_, C0 = (int)C0_;
        AutoBuffer<float> sbuf_(C0);
        float* sbuf = sbuf_.data();

        for (int n = n0; n < n1; n++) {
            const _Tp* inp = inp_ + planesize*C0*n;
            _Tp* out = out_ + C0*n;
            int planesize_C0 = planesize*C0;
            // by computing sum in blocks we probably increase accuracy
            int BLOCK_SIZE = 256*C0, blocksize = 0;
            if (nlanes == C0) {
                v_float32 s0 = vx_setzero_f32();
                for (int i = 0; i < planesize_C0; i += blocksize, inp += blocksize) {
                    blocksize = std::min(planesize_C0 - i, BLOCK_SIZE);
                    v_float32 bs0 = vx_setzero_f32();

                    for (int j = 0; j < blocksize; j += C0)
                        bs0 = v_add(bs0, vx_load_expand(inp + j));
                    s0 = v_add(s0, bs0);
                }
                s0 = v_mul(s0, scale);
                v_pack_store(out, s0);
            } else if (nlanes*2 == C0) {
                v_float32 s0 = vx_setzero_f32(), s1 = s0;
                for (int i = 0; i < planesize_C0; i += blocksize, inp += blocksize) {
                    blocksize = std::min(planesize_C0 - i, BLOCK_SIZE);
                    v_float32 bs0 = vx_setzero_f32(), bs1 = bs0;

                    for (int j = 0; j < blocksize; j += C0) {
                        bs0 = v_add(bs0, vx_load_expand(inp + j));
                        bs1 = v_add(bs1, vx_load_expand(inp + j + nlanes));
                    }

                    s0 = v_add(s0, bs0);
                    s1 = v_add(s1, bs1);
                }
                s0 = v_mul(s0, scale);
                s1 = v_mul(s1, scale);
                v_pack_store(out, s0);
                v_pack_store(out + nlanes, s1);
            } else {
                memset(sbuf, 0, C0*sizeof(sbuf[0]));
                for (int i = 0; i < planesize_C0; i += blocksize, inp += blocksize) {
                    blocksize = std::min(planesize_C0 - i, BLOCK_SIZE);
                    for (int c = 0; c < C0; c += nlanes*4) {
                        v_float32 s0 = vx_load(sbuf + c);
                        v_float32 s1 = vx_load(sbuf + c + nlanes);
                        v_float32 s2 = vx_load(sbuf + c + nlanes*2);
                        v_float32 s3 = vx_load(sbuf + c + nlanes*3);

                        v_float32 bs0 = vx_setzero_f32(), bs1 = bs0, bs2 = bs0, bs3 = bs0;

                        for (int j = 0; j < blocksize; j += C0) {
                            bs0 = v_add(bs0, vx_load_expand(inp + c + j));
                            bs1 = v_add(bs1, vx_load_expand(inp + c + nlanes + j));
                            bs2 = v_add(bs2, vx_load_expand(inp + c + nlanes*2 + j));
                            bs3 = v_add(bs3, vx_load_expand(inp + c + nlanes*3 + j));
                        }

                        s0 = v_add(s0, bs0);
                        s1 = v_add(s1, bs1);
                        s2 = v_add(s2, bs2);
                        s3 = v_add(s3, bs3);
                        vx_store(sbuf + c, s0);
                        vx_store(sbuf + c + nlanes, s1);
                        vx_store(sbuf + c + nlanes*2, s2);
                        vx_store(sbuf + c + nlanes*3, s3);
                    }
                }
                for (int c = 0; c < C0; c += nlanes*2) {
                    v_float32 s0 = vx_load(sbuf + c);
                    v_float32 s1 = vx_load(sbuf + c + nlanes);
                    s0 = v_mul(s0, scale);
                    s1 = v_mul(s1, scale);
                    v_pack_store(out + c, s0);
                    v_pack_store(out + c + nlanes, s1);
                }
            }
        }
    });
}

static void global_average_pool_16f(const void* inp_, const MatShape& size, void* out_)
{
    global_average_pool_16((const hfloat*)inp_, size, (hfloat*)out_);
}

static void global_average_pool_16bf(const void* inp_, const MatShape& size, void* out_)
{
    global_average_pool_16((const bfloat*)inp_, size, (bfloat*)out_);
}

typedef void (*global_avgpool_func_t)(const void* inp, const MatShape& size, void* out);

class GlobalAveragePoolLayerImpl : public GlobalAveragePoolLayer
{
public:
    GlobalAveragePoolLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
    }
    
    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV;
    }

    virtual int supportBlockLayout(int) const CV_OVERRIDE
    {
        return 1;
    }

    virtual int64_t getFLOPS(const std::vector<MatShape> &inputs,
                             const std::vector<MatShape> &outputs) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() == 1);
        CV_Assert(outputs.size() == 1);
        return (int64_t)inputs[0].total();
    }

    int inferType(int inptype0) const
    {
        return inptype0;
    }

    virtual void getTypes(const std::vector<MatType>& inptypes,
                          const int, const int,
                          std::vector<MatType>& outtypes,
                          std::vector<MatType>& temptypes) const CV_OVERRIDE
    {
        int ninputs = (int)inptypes.size();
        CV_Assert(ninputs == 1);

        outtypes.assign(1, inferType(inptypes[0]));
        temptypes.clear();
    }
    
    MatShape inferShape(const MatShape& inpshape) const
    {
        int ndims = inpshape.dims;
        DataLayout inplayout = inpshape.layout;
        MatShape outshape = inpshape;
        CV_Assert(inplayout == DATA_LAYOUT_BLOCK || inplayout == DATA_LAYOUT_NCHW);

        outshape.dims = ndims - (inplayout == DATA_LAYOUT_BLOCK);
        outshape.layout = DATA_LAYOUT_NCHW;
        outshape.C = 0;
        
        for (int i = 2; i < outshape.dims; i++)
            outshape[i] = 1;

        return outshape;
    }

    virtual bool getMemoryShapes(const std::vector<MatShape>& inpshapes,
                                 const int,
                                 std::vector<MatShape> &outshapes,
                                 std::vector<MatShape> &tempshapes) const CV_OVERRIDE
    {
        size_t ninputs = inpshapes.size();
        CV_Assert(ninputs == 1);

        outshapes.assign(1, inferShape(inpshapes[0]));
        tempshapes.clear();
        return true;
    }
    
    void finalize(InputArrayOfArrays, OutputArrayOfArrays outputs_arr) CV_OVERRIDE
    {
    }

    void forward(InputArrayOfArrays inputs_arr,
                 OutputArrayOfArrays outputs_arr,
                 OutputArrayOfArrays) CV_OVERRIDE
    {
        size_t ninputs = inputs_arr.total();
        CV_Assert(ninputs == 1);
        
        int inptype = inputs_arr.type(0), outtype = inferType(inptype);
        MatShape inpshape = inputs_arr.shape(0);
        MatShape outshape = inferShape(inpshape);
        
        int outKind = outputs_arr.kind();
        CV_Assert(outKind == _InputArray::STD_VECTOR_MAT ||
                  outKind == _InputArray::STD_VECTOR_UMAT);
        
        if (outKind == _InputArray::STD_VECTOR_MAT) {
            Mat inp = inputs_arr.getMat(0);
            std::vector<Mat>& outs = outputs_arr.getMatVecRef();
            outs.resize(1);
            outs[0].fit(outshape, outtype);
            runOp(inp, outs[0]);
        } else {
            // [TODO] more efficient OpenCL implementation
            Mat inp = inputs_arr.getMat(0);
            std::vector<UMat>& outs = outputs_arr.getUMatVecRef();
            outs.resize(1);
            outs[0].fit(outshape, inptype);
            Mat temp(outshape, outtype);
            runOp(inp, temp);
            temp.copyTo(outs[0]);
        }
    }
    
    void runOp(const Mat& inp, Mat& out)
    {
        int inptype = inp.type();
        global_avgpool_func_t func =
            inptype == CV_32F ? global_average_pool_32f :
            inptype == CV_16F ? global_average_pool_16f :
            inptype == CV_16BF ? global_average_pool_16bf : nullptr;

        CV_Assert(func != nullptr);
        func(inp.data, inp.shape(), out.data);
    }
};

Ptr<GlobalAveragePoolLayer> GlobalAveragePoolLayer::create(const LayerParams& params)
{
    return Ptr<GlobalAveragePoolLayer>(new GlobalAveragePoolLayerImpl(params));
}

}}

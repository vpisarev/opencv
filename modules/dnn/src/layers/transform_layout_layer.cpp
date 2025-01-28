// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "layers_common.hpp"
#include "../net_impl.hpp"

#if 0
namespace cv
{
namespace dnn
{

template <typename _Tp>
void transformLayout_(const _Tp* inp_, int istep, int istep0, int istep1,
                      _Tp* out_, int ostep, int ostep0, int ostep1,
                      int npix, int C0, int C1, int C)
{
    CV_Assert(C0 % 8 == 0 || C0 == 4 || C1 == 1);
    CV_Assert(istep0 == 1 || ostep0 == 1);
    const int dC0 = std::min(C0, (int)8);
    for (int c1 = 0; c1 < C1; c1++) {
        for (int c0 = 0; c0 < C0; c0 += dC0) {
            const _Tp* inp = inp_ + istep0*c0 + istep1*c1;
            _Tp* out = out_ + ostep0*c0 + ostep1*c1;
            int dc = std::min(C - (c1*C0 + c0), dC0);
            if (dc == 8) {
                if (istep0 == 1) {
                    for (int i = 0; i < npix; i++, inp += istep, out += ostep) {
                        _Tp x0 = inp[0], x1 = inp[1], x2 = inp[2], x3 = inp[3];
                        _Tp x4 = inp[4], x5 = inp[5], x6 = inp[6], x7 = inp[7];
                        out[0] = x0; out[ostep0] = x1; out[ostep0*2] = x2; out[ostep0*3] = x3;
                        out[ostep0*4] = x4; out[ostep0*5] = x5; out[ostep0*6] = x6; out[ostep0*7] = x7;
                    }
                } else {
                    for (int i = 0; i < npix; i++, inp += istep, out += ostep) {
                        _Tp x0 = inp[0], x1 = inp[istep0], x2 = inp[istep0*2], x3 = inp[istep0*3];
                        _Tp x4 = inp[istep0*4], x5 = inp[istep0*5], x6 = inp[istep0*6], x7 = inp[istep0*7];
                        out[0] = x0; out[1] = x1; out[2] = x2; out[3] = x3;
                        out[4] = x4; out[5] = x5; out[6] = x6; out[7] = x7;
                    }
                }
            } else if (dc == 4) {
                if (istep0 == 1) {
                    for (int i = 0; i < npix; i++, inp += istep, out += ostep) {
                        _Tp x0 = inp[0], x1 = inp[1], x2 = inp[2], x3 = inp[3];
                        out[0] = x0; out[ostep0] = x1; out[ostep0*2] = x2; out[ostep0*3] = x3;
                    }
                } else {
                    for (int i = 0; i < npix; i++, inp += istep, out += ostep) {
                        _Tp x0 = inp[0], x1 = inp[istep0], x2 = inp[istep0*2], x3 = inp[istep0*3];
                        out[0] = x0; out[1] = x1; out[2] = x2; out[3] = x3;
                    }
                }
            } else if (dc == 3 && ostep0 == 1 && ostep == C0) {
                memset(out, 0, npix*C0*sizeof(out[0]));
                for (int i = 0; i < npix; i++, inp += istep, out += ostep) {
                    _Tp x0 = inp[0], x1 = inp[istep0], x2 = inp[istep0*2];
                    out[0] = x0; out[1] = x1; out[2] = x2;
                }
            } else {
                for (int i = 0; i < npix; i++, inp += istep, out += ostep) {
                    int c = 0;
                    for (; c < dc; c++)
                        out[ostep0*c] = inp[istep0*c];
                    if (ostep == C0) {
                        for (; c < dC0; c++)
                            out[ostep0*c] = 0;
                    }
                }
            }
        }
    }
}

#undef CV_TRANSFORM_LAYOUT_IMPL
#define CV_TRANSFORM_LAYOUT_IMPL(typ, suffix) \
static void transformLayout_##suffix(const void* inp_, int istep, int istep0, int istep1, \
                                      void* out_, int ostep, int ostep0, int ostep1, \
                                      int npix, int C0, int C1, int C) \
{ \
    transformLayout_((const typ*)inp_, istep, istep0, istep1, \
                     (typ*)out_, ostep, ostep0, ostep1, npix, C0, C1, C); \
}

CV_TRANSFORM_LAYOUT_IMPL(uint8_t, 8u)
CV_TRANSFORM_LAYOUT_IMPL(uint16_t, 16u)
CV_TRANSFORM_LAYOUT_IMPL(uint32_t, 32u)
CV_TRANSFORM_LAYOUT_IMPL(uint, 64u)

typedef void (*transform_layout_func_t)(const void* inp, int istep, int istep0, int istep1,
                                        void* out, int ostep, int ostep0, int ostep1,
                                        int npix, int C0, int C1, int C);

static void transformLayout(const Mat& inp, Mat& out)
{
    MatShape inpshape = inp.shape();
    MatShape outshape = out.shape();
    DataLayout inplayout = inpshape.layout;
    DataLayout outlayout = outshape.layout;

    if (inp.empty())
        return;

    if (inplayout == outlayout) {
        inp.copyTo(out);
        return;
    }

    int inp_ndims = inpshape.dims;
    int out_ndims = outshape.dims;
    int N = inpshape[0];
    int C = inplayout == DATA_LAYOUT_BLOCK ? inpshape.C :
        inpshape[inplayout == DATA_LAYOUT_NCHW ? 1 : inp_ndims-1];
    int inptotal = (int)inp.total();
    int outtotal = (int)out.total();
    int inplanesize_C = inptotal / N;
    int outplanesize_C = outtotal / N;
    int planesize = (inplayout != DATA_LAYOUT_BLOCK ? inplanesize_C : outplanesize_C)/C;
    int allplanes = planesize*N;

    constexpr int BLOCK_SIZE = 1 << 17;
    int nblocks = (outtotal + BLOCK_SIZE - 1)/BLOCK_SIZE;
    nblocks = std::min(nblocks, allplanes);

    size_t esz = CV_ELEM_SIZE(inptype);
    int istep0, istep1, istep;
    int ostep0, ostep1, ostep;
    int C0_ = C, C1_ = 1;

    if (inplayout == DATA_LAYOUT_BLOCK || outlayout == DATA_LAYOUT_BLOCK) {
        C0_ = inplayout == DATA_LAYOUT_BLOCK ? inpshape[inp_ndims-1] : outshape[out_ndims-1];
        C1_ = (C + C0_ - 1)/C0_;
    }

    if (inplayout == DATA_LAYOUT_NCHW) {
        istep = 1;
        istep0 = planesize;
        istep1 = planesize*C0_;
    } else if (inplayout == DATA_LAYOUT_NHWC) {
        istep = C;
        istep0 = 1;
        istep1 = C0_;
    } else {
        istep = C0_;
        istep0 = 1;
        istep1 = planesize*C0_;
    }

    if (outlayout == DATA_LAYOUT_NCHW) {
        ostep = 1;
        ostep0 = planesize;
        ostep1 = planesize*C0_;
    } else if (outlayout == DATA_LAYOUT_NHWC) {
        ostep = C;
        ostep0 = 1;
        ostep1 = C0_;
    } else {
        ostep = C0_;
        ostep0 = 1;
        ostep1 = planesize*C0_;
    }

    const char* inptr0 = (const char*)inp.data;
    char* outptr0 = (char*)out.data;

    transform_layout_func_t transform_layout_func =
        esz == 1 ? transformLayout_8u :
        esz == 2 ? transformLayout_16u :
        esz == 4 ? transformLayout_32u :
        esz == 8 ? transformLayout_64u : nullptr;

    CV_Assert(transform_layout_func != nullptr);

    parallel_for_(Range(0, nblocks), [&](const Range& r) {
        int start = r.start*allplanes/nblocks;
        int end = r.end*allplanes/nblocks;
        int npix = 0;

        for (int ofs = start; ofs < end; ofs += npix) {
            int sample_idx = ofs/planesize;
            int rawofs = ofs - sample_idx*planesize;
            npix = std::min(planesize - rawofs, end - ofs);
            const char* inptr = inptr0 + (inplanesize_C*sample_idx + istep*rawofs)*esz;
            char* outptr = outptr0 + (outplanesize_C*sample_idx + ostep*rawofs)*esz;
            transform_layout_func(inptr, istep, istep0, istep1,
                                  outptr, ostep, ostep0, ostep1,
                                  npix, C0_, C1_, C);
        }
    });
}

class TransformLayoutLayerImpl : public TransformLayoutLayer
{
public:
    TransformLayoutLayerImpl(const LayerParams& params)
    {
        layout = (DataLayout)params.get<int>("layout");
        C0 = params.get<int>("C0", 1);
    }

    virtual std::ostream& dumpAttrs(std::ostream& strm, int indent) const CV_OVERRIDE
    {
        prindent(strm, indent);
        strm << "layout: " << layoutToString(layout) << ",\n";

        if (layout == DATA_LAYOUT_BLOCK) {
            prindent(strm, indent);
            strm << "C0: " << C0 << ",\n";
        }
        return strm;
    }

    int inferType(int inptype0) const
    {
        return inptype0;
    }

    virtual bool alwaysSupportInplace() const CV_OVERRIDE
    {
        return false;
    }

    virtual int64_t getFLOPS(const std::vector<MatShape> &inputs,
                             const std::vector<MatShape> &outputs) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() == 1);
        CV_Assert(outputs.size() == 1);
        // probably, there should be a coefficient in the case of complex reduction functions
        return (int64_t)std::max(inputs[0].size.total(), outputs[0].size.total());
    }

    virtual void inferTypes(const Net2& net, const Graph& graph,
                            const std::vector<Arg>& inpargs,
                            const std::vector<int>& inptypes,
                            const std::vector<Arg>& outargs,
                            std::vector<int>& outtypes) const CV_OVERRIDE
    {
        int ninputs = (int)inpargs.size(), noutputs = (int)outargs.size();
        CV_Assert(minNumInputs() <= ninputs && ninputs <= maxNumInputs());
        CV_Assert((int)inptypes.size() == ninputs);
        CV_Assert(noutputs == 1);

        outtypes.resize(1);
        outtypes[0] = inferType(inptypes[0]);
    }

    MatShape inferShapes_(const MatShape& inpshape) const
    {
        int ndims = inpshape.ndims;
        TensorLayout inplayout = inpshape.layout;
        CV_Assert(layout == LAYOUT_NCHWc || layout == LAYOUT_NCHW || layout == LAYOUT_NHWC);
        CV_Assert(inplayout == LAYOUT_NCHWc || inplayout == LAYOUT_NCHW || inplayout == LAYOUT_NHWC);

        if (layout == inplayout) {
            // identity
            CV_Assert(layout != LAYOUT_NCHWc || C0 == inpshape[ndims-1]);
            return inpshape;
        }

        // non-block => block
        if (layout == LAYOUT_NCHWc)
            return inpshape.toBlock(C0);

        // block => non-block
        if (inplayout == LAYOUT_NCHWc)
            return inpshape.fromBlock(layout);

        MatShape outshape = inpshape;
        outshape.layout = layout;

        // NHWC => NCHW
        if (layout == LAYOUT_NCHW) {
            CV_Assert(inplayout == LAYOUT_NHWC);
            int C = inpshape[ndims-1];
            for (int i = 2; i < ndims; i++)
                outshape[i] = inpshape[i-1];
            outshape[1] = C;
        } else {
            // NCHW => NHWC
            CV_Assert(layout == LAYOUT_NHWC && inplayout == LAYOUT_NCHW);
            int C = inpshape[1];
            for (int i = 2; i < ndims; i++)
                outshape[i-1] = inpshape[i];
            outshape[ndims-1] = C;
        }
        return outshape;
    }

    virtual void inferShapes(Net2& net, const Graph& graph,
                             const std::vector<Arg>& inpargs,
                             const std::vector<MatShape>& inpshapes,
                             const std::vector<Arg>& outargs,
                             std::vector<MatShape>& outshapes,
                             bool symbolic) const CV_OVERRIDE
    {
        int ninputs = (int)inpargs.size(), noutputs = (int)outargs.size();
        CV_Assert(minNumInputs() <= ninputs && ninputs <= maxNumInputs());
        CV_Assert(noutputs == 1);
        outshapes.resize(1);

        const MatShape& inpshape = inpshapes[0];
        outshapes[0] = inferShapes_(inpshape);
    }

    virtual void forward(Net2& net, Graph& graph,
                        const std::vector<Tensor>& inputs,
                        std::vector<Tensor>& outputs,
                        std::vector<Buffer>& tempbufs) CV_OVERRIDE
    {
        size_t ninputs = inputs.size();
        CV_Assert(minNumInputs() <= ninputs && ninputs <= maxNumInputs());
        const Tensor& inp = inputs[0];
        CV_Assert(inp.isContinuous());

        int inptype = inp.type(), outtype = inferType(inptype);
        MatShape inpshape = inp.size();
        MatShape outshape = inferShapes_(inpshape);
        outputs.resize(1);
        Tensor& out = outputs[0];
        out.fitSameDevice(inp, outshape, outtype);
        CV_Assert(out.isContinuous());
    }
};

Ptr<TransformLayoutLayer> TransformLayoutLayer::create(const LayerParams& params)
{
    return std::make_shared<TransformLayoutLayerImpl>(params);
}

}}
#endif

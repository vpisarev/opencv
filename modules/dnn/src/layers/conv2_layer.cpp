// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../net_impl.hpp"
#include "layers_common.hpp"
#include "conv2_common.hpp"
#include "opencv2/core/hal/intrin.hpp"

namespace cv
{
namespace dnn
{

/*
    Convolution layer, as defined in ONNX specification:
    https://onnx.ai/onnx/operators/onnx__Conv.html

    Opset's 1 to 22 are covered.
*/

static void initConv2DTables(const ConvState& cs,
                             std::vector<int32_t>& ofsbuf_,
                             std::vector<int32_t>& ofs0_,
                             std::vector<int32_t*>& ofsptrs_,
                             std::vector<uint8_t>& mask_)
{
    int Hk_ = cs.kshape[0], Wk_ = cs.kshape[1];
    int DY_ = cs.dilations[0], DX_ = cs.dilations[1];
    int pad_y0 = cs.pads[0], pad_x0 = cs.pads[1];
    int Hi_ = cs.inpshape[2], Wi_ = cs.inpshape[3], H = cs.outshape[2], W = cs.outshape[3];
    int C0_ = cs.inpshape.back(), C1_ = cs.inpshape[1], K1 = cs.outshape[1];
    int ngroups = cs.ngroups, C1g = C1_/ngroups;
    int inner_y0 = cs.inner[0], inner_y1 = cs.inner[1];
    int inner_x0 = cs.inner[cs.nspatialdims], inner_x1 = cs.inner[cs.nspatialdims + 1];

    mask_.resize(H*W);
    ofs0_.resize(H*W);
    ofsptrs_.resize(H*W);

    int ofs_blocksize = C1g*Hk_*Wk_;
    bool have_inner = inner_y0 < inner_y1 && inner_x0 < inner_x1;

    int nblocks = have_inner ? 1 + (inner_y0 + (H - inner_y1))*W +
        (inner_y1 - inner_y0)*(inner_x0 + W - inner_x1) : W*H;

    ofsbuf_.resize(ofs_blocksize*nblocks);
    int32_t* ofsbuf = ofsbuf_.data();

    if (have_inner) {
        for (int c = 0, k = 0; c < C1g; c++) {
            for (int dy = 0; dy < Hk_; dy++) {
                int yi = dy*DY_;
                for (int dx = 0; dx < Wk_; dx++, k++) {
                    int xi = dx*DX_;
                    ofsbuf[k] = (int32_t)(((c*Hi_ + yi)*Wi_ + xi)*C0_);
                }
            }
        }
    }

    parallel_for_(Range(0, H), [&](const Range& r) {
        int C0 = C0_;
        int Hk = Hk_, Wk = Wk_;
        int Hi = Hi_, Wi = Wi_;
        int SY = cs.strides[0], SX = cs.strides[1];
        int DY = cs.dilations[0], DX = cs.dilations[1];
        uint8_t* mask = mask_.data();
        int32_t* ofs0 = ofs0_.data();
        int32_t** ofsptrs = ofsptrs_.data();
        int64_t curr_block = 1;
        if (have_inner) {
            curr_block += std::min(r.start, inner_y0)*W;
            curr_block += std::min(std::max(r.start - inner_y0, 0),
                                   inner_y1 - inner_y0)*(inner_x0 + W - inner_x1);
            curr_block += std::max(r.start - inner_y1, 0)*W;
        } else {
            curr_block = r.start*W;
        }
        for (int y0 = r.start; y0 < r.end; y0++) {
            int yi_ = y0*SY - pad_y0;
            bool y_inside = inner_y0 <= y0 && y0 < inner_y1;

            for (int x0 = 0; x0 < W; x0++) {
                int xi_ = x0*SX - pad_x0;
                bool x_inside = inner_x0 <= x0 && x0 < inner_x1;
                uint8_t m = (uint8_t)(y_inside & x_inside);

                mask[y0*W + x0] = m;

                if (m) {
                    ofs0[y0*W + x0] = (int32_t)((yi_*Wi + xi_)*C0);
                    ofsptrs[y0*W + x0] = ofsbuf;
                } else {
                    ofs0[y0*W + x0] = 0;
                    int32_t* ofsptr = ofsbuf + curr_block*ofs_blocksize;
                    ofsptrs[y0*W + x0] = ofsptr;
                    curr_block++;

                    for (int c = 0, k = 0; c < C1g; c++) {
                        for (int dy = 0; dy < Hk; dy++) {
                            int yi = yi_ + dy*DY;
                            bool yi_inside = 0 <= yi && yi < Hi;

                            for (int dx = 0; dx < Wk; dx++, k++) {
                                int xi = xi_ + dx*DX;
                                bool xi_inside = 0 <= xi && xi < Wi;
                                ofsptr[k] = (yi_inside & xi_inside) ?
                                    (int32_t)(((c*Hi + yi)*Wi + xi)*C0) : INT_MIN/2;
                            }
                        }
                    }
                }
            }
        }
    });
}

template<typename _InpT, typename _OutT> void
repackConv2DWeights_(const _InpT* inpw_, _OutT* outw_,
                     size_t inp_step_c, size_t inp_step_k, int ksize,
                     int C0, int K0, int curr_C0, int curr_K0)
{
    const _InpT* inpw = inpw_;
    _OutT* outw = outw_;
    for (int xy = 0; xy < ksize; xy++, inpw++, outw += C0*K0) {
        for (int c0 = 0; c0 < curr_C0; c0++) {
            for (int k0 = 0; k0 < curr_K0; k0++) {
                outw[c0*K0 + k0] = _OutT(inpw[inp_step_k*k0 + inp_step_c*c0]);
            }
        }
    }
}


// K x (C/ngroups) x Hk x Wk => K1 x C1/ngroups x Hk x Wk x C0 x K0,
// where K0 == C0
static void repackConv2DWeights(const void* inpw__, int inptype_, void* outw__, int outtype_,
                                  const MatShape& wshape, int C0_)
{
    CV_Assert(inptype_ == CV_32F || inptype_ == CV_16F);
    CV_Assert(outtype_ == CV_32F || outtype_ == CV_16F);

    int K1 = (wshape[0] + C0_ - 1)/C0_;
    parallel_for_(Range(0, K1), [&](const Range& r) {
        int inptype = inptype_, outtype = outtype_;
        size_t inp_esz = CV_ELEM_SIZE(inptype);
        size_t out_esz = CV_ELEM_SIZE(outtype);
        int C0 = C0_, K0 = C0_;
        int K = wshape[0], Cg = wshape[1];
        int C1g = (Cg + C0 - 1)/C0;
        int Hk = wshape[2], Wk = wshape[3];
        int ksize = Hk*Wk;
        size_t inp_step_c = ksize, inp_step_k = Cg*ksize;
        size_t out_microplane_size = ksize*C0*K0*out_esz;

        for (int k1 = r.start; k1 < r.end; k1++) {
            int curr_K0 = std::min(K - k1*K0, K0);
            for (int c1g = 0; c1g < C1g; c1g++) {
                uint8_t* inpw_ = (uint8_t*)inpw__ + (k1*K0*inp_step_k + c1g*C0*inp_step_c)*inp_esz;
                uint8_t* outw_ = (uint8_t*)outw__ + (k1*C1g + c1g)*out_microplane_size;
                int curr_C0 = std::min(Cg - c1g*C0, C0);
                if (curr_K0 != K0 || curr_C0 != C0)
                    memset(outw_, 0, out_microplane_size);

                if (inptype == CV_32F && outtype == CV_32F)
                    repackConv2DWeights_((const float*)inpw_, (float*)outw_, inp_step_c,
                                         inp_step_k, ksize, C0, K0, curr_C0, curr_K0);
                else if (inptype == CV_32F && outtype == CV_16F)
                    repackConv2DWeights_((const float*)inpw_, (hfloat*)outw_, inp_step_c,
                                         inp_step_k, ksize, C0, K0, curr_C0, curr_K0);
                else if (inptype == CV_16F && outtype == CV_32F)
                    repackConv2DWeights_((const hfloat*)inpw_, (float*)outw_, inp_step_c,
                                         inp_step_k, ksize, C0, K0, curr_C0, curr_K0);
                else if (inptype == CV_16F && outtype == CV_16F)
                    repackConv2DWeights_((const hfloat*)inpw_, (hfloat*)outw_, inp_step_c,
                                         inp_step_k, ksize, C0, K0, curr_C0, curr_K0);
                else break;
            }
        }
    });
}

static void conv2d_32f(const void* inp__, const void* residual__, void* out__,
                       const ConvState& cs, const void* weights__,
                       const float* scale__, const float* bias__,
                       const int32_t* ofs0__, const int32_t** ofsptrs__,
                       const uint8_t* mask__)
{
    int nlanes_ = VTraits<v_float32>::vlanes();
    int C0_ = cs.inpshape.back();

    CV_Assert(C0_ == nlanes_ || C0_ == nlanes_*2 || C0_ % (nlanes_*4) == 0);
    CV_Assert(cs.activation == nullptr || cs.fastActivation == FAST_ACTIV_NONE);

    int NK1 = cs.outshape[0]*cs.outshape[1];

    parallel_for_(Range(0, NK1), [&](const Range& r) {
        const float* scale_ = scale__;
        const float* bias_ = bias__;
        const int32_t* ofs0_ = ofs0__;
        const int32_t** ofsptrs_ = ofsptrs__;
        constexpr int BLOCK_SIZE = 10;
        int nk0 = r.start, nk1 = r.end;
        int C0 = C0_, K0 = C0;
        int Hi = cs.inpshape[2], Wi = cs.inpshape[3];
        int H = cs.outshape[2], W = cs.outshape[3];
        int iplanesize = Hi*Wi;
        int planesize = H*W;
        int Hk = cs.kshape[0], Wk = cs.kshape[1];
        int C1 = cs.inpshape[1], K1 = cs.outshape[1];
        int ngroups = cs.ngroups, K1g = K1/ngroups, C1g = C1/ngroups;
        int nC = C1g*Hk*Wk*C0*K0;
        AutoBuffer<float> sumbuf(BLOCK_SIZE*K0*3);
        float* sum = sumbuf.data();
        float* scale = sum + BLOCK_SIZE*K0;
        float* bias = sum + BLOCK_SIZE*K0*2;
        const float* inptrs[BLOCK_SIZE];
        const int32_t* ofsptrs[BLOCK_SIZE];
        FastActivation fastActivation = cs.fastActivation;
        const float* activParams = cs.activParams;
        activation_func_t activation = cs.activation;
        float maxval = fastActivation == FAST_ACTIV_CLIP ? activParams[1] : FLT_MAX;
        float alpha = fastActivation == FAST_ACTIV_LEAKY_RELU ? activParams[0] :
                    fastActivation == FAST_ACTIV_NONE ? 1.f : 0.f;

        for (int j = 0; j < BLOCK_SIZE*K0; j++) {
            scale[j] = 1.f;
            bias[j] = 0.f;
        }

        for (int nk = nk0; nk < nk1; nk++) {
            int n = nk/K1, k1 = nk - n*K1;
            int g = k1/K1g;
            float* out = (float*)out__ + nk*planesize*K0;
            const float* inp0 = (const float*)inp__ + (n*C1 + g*C1g)*iplanesize*C0;
            const float* resptr = residual__ ? (const float*)residual__ + nk*planesize*K0 : nullptr;
            const float* wptr = (const float*)weights__ + k1*nC;

            if (scale_) {
                for (int b = 0; b < BLOCK_SIZE; b++)
                    for (int k = 0; k < K0; k++)
                        scale[b*K0 + k] = scale_[k1*K0 + k];
            }

            if (bias_) {
                for (int b = 0; b < BLOCK_SIZE; b++)
                    for (int k = 0; k < K0; k++)
                        bias[b*K0 + k] = bias_[k1*K0 + k];
            }

            for (int xy0 = 0; xy0 < planesize; xy0 += BLOCK_SIZE, out += K0*BLOCK_SIZE,
                                            resptr += (resptr ? K0*BLOCK_SIZE : 0)) {
                int j = 0, blocksize = std::min(planesize - xy0, BLOCK_SIZE);

                for (; j < blocksize; j++) {
                    inptrs[j] = inp0 + ofs0_[xy0 + j];
                    ofsptrs[j] = ofsptrs_[xy0 + j];
                }

                if (j < BLOCK_SIZE) {
                    const float* last_inptr = inptrs[blocksize-1];
                    const int32_t* last_ofsptr = ofsptrs[blocksize-1];
                    for (; j < BLOCK_SIZE; j++) {
                        inptrs[j] = last_inptr;
                        ofsptrs[j] = last_ofsptr;
                    }
                }

                for (int i = 0; i < BLOCK_SIZE*K0; i++)
                    sum[i] = 0.f;

                for (int c1 = 0, i = 0; c1 < nC; c1 += K0*C0, i++) {
                    for (j = 0; j < BLOCK_SIZE; j++) {
                        int32_t ofs_ij = ofsptrs[j][i];
                        const float* x = &inptrs[j][std::max(ofs_ij, 0)];
                        float mij = (float)(ofs_ij >= 0);
                        for (int c0 = 0; c0 < C0; c0++) {
                            float xc = x[c0]*mij;
                            for (int k = 0; k < K0; k++) {
                                float w = wptr[c1 + c0*K0 + k];
                                sum[K0*j + k] += xc*w;
                            }
                        }
                    }
                }

                if (activation) {
                    if (resptr) {
                        for (j = 0; j < blocksize*K0; j++) {
                            float v = sum[j]*scale[j] + bias[j] + resptr[j];
                            sum[j] = v;
                        }
                    } else {
                        for (j = 0; j < blocksize*K0; j++) {
                            float v = sum[j]*scale[j] + bias[j];
                            sum[j] = v;
                        }
                    }
                    activation(sum, out, blocksize*K0, activParams);
                } else {
                    if (resptr) {
                        for (j = 0; j < blocksize*K0; j++) {
                            float v = sum[j]*scale[j] + bias[j] + resptr[j];
                            v = std::min(v*(v >= 0 ? 1.f : alpha), maxval);
                            out[j] = v;
                        }
                    } else {
                        for (j = 0; j < blocksize*K0; j++) {
                            float v = sum[j]*scale[j] + bias[j];
                            v = std::min(v*(v >= 0 ? 1.f : alpha), maxval);
                            out[j] = v;
                        }
                    }
                }
            }
        }
    });
}

static void conv2d_1x1_32f(const void* inp__, const void* residual__, void* out__,
                           const ConvState& cs, const void* weights__,
                           const float* scale__, const float* bias__,
                           const int32_t*, const int32_t**, const uint8_t*)
{
    int nlanes_ = VTraits<v_float32>::vlanes();
    int C0_ = cs.inpshape.back();

    CV_Assert(C0_ == nlanes_ || C0_ == nlanes_*2 || C0_ % (nlanes_*4) == 0);
    CV_Assert(cs.activation == nullptr || cs.fastActivation == FAST_ACTIV_NONE);
    CV_Assert(cs.kshape[0] == 1 && cs.kshape[1] == 1);
    CV_Assert(cs.outshape.back() == cs.inpshape.back());
    CV_Assert(cs.pads[0] == 0 && cs.pads[1] == 0 &&
              cs.pads[cs.nspatialdims] == 0 && cs.pads[cs.nspatialdims+1] == 0);

    int NK1 = cs.outshape[0]*cs.outshape[1];

    parallel_for_(Range(0, NK1), [&](const Range& r) {
        const float* scale_ = scale__;
        const float* bias_ = bias__;
        constexpr int BLOCK_SIZE = 10;
        int nk0 = r.start, nk1 = r.end;
        //int nlanes = nlanes_;
        int C0 = C0_, K0 = C0;
        int Hi = cs.inpshape[2], Wi = cs.inpshape[3];
        int H0 = cs.outshape[2], W0 = cs.outshape[3];
        int iplanesize = Hi*Wi;
        int planesize = H0*W0;
        int SY = cs.strides[0], SX = cs.strides[1];
        int C1 = cs.inpshape[1], K1 = cs.outshape[1];
        int ngroups = cs.ngroups, K1g = K1/ngroups, C1g = C1/ngroups;
        int nC = C1g*C0*K0;
        AutoBuffer<float> sumbuf(BLOCK_SIZE*K0*3);
        float* sum = sumbuf.data();
        float* scale = sum + BLOCK_SIZE*K0;
        float* bias = sum + BLOCK_SIZE*K0*2;
        const float* inptrs[BLOCK_SIZE];
        FastActivation fastActivation = cs.fastActivation;
        const float* activParams = cs.activParams;
        activation_func_t activation = cs.activation;
        float maxval = fastActivation == FAST_ACTIV_CLIP ? activParams[1] : FLT_MAX;
        float alpha = fastActivation == FAST_ACTIV_LEAKY_RELU ? activParams[0] :
                    fastActivation == FAST_ACTIV_NONE ? 1.f : 0.f;
        bool S1 = SY == 1 && SX == 1;

        for (int j = 0; j < BLOCK_SIZE*K0; j++) {
            scale[j] = 1.f;
            bias[j] = 0.f;
        }

        for (int nk = nk0; nk < nk1; nk++) {
            int n = nk/K1, k1 = nk - n*K1;
            int g = k1/K1g;
            float* out = (float*)out__ + nk*planesize*K0;
            const float* inp0 = (const float*)inp__ + (n*C1 + g*C1g)*iplanesize*C0;
            const float* resptr = residual__ ? (const float*)residual__ + nk*planesize*K0 : nullptr;
            const float* wptr = (const float*)weights__ + k1*nC;

            if (scale_) {
                for (int b = 0; b < BLOCK_SIZE; b++)
                    for (int k = 0; k < K0; k++)
                        scale[b*K0 + k] = scale_[k1*K0 + k];
            }

            if (bias_) {
                for (int b = 0; b < BLOCK_SIZE; b++)
                    for (int k = 0; k < K0; k++)
                        bias[b*K0 + k] = bias_[k1*K0 + k];
            }

            int yiWi = 0, xi = 0;
            for (int xy0 = 0; xy0 < W0*H0; xy0 += BLOCK_SIZE, out += K0*BLOCK_SIZE,
                                               resptr += (resptr ? K0*BLOCK_SIZE : 0))
            {
                int j = 0, blocksize = std::min(W0*H0 - xy0, BLOCK_SIZE);

                if (S1) {
                    for (; j < blocksize; j++) {
                        inptrs[j] = inp0 + (xy0 + j)*C0;
                    }
                } else {
                    for (; j < blocksize; j++) {
                        inptrs[j] = inp0 + (yiWi + xi)*C0;
                        if ((xi += SX) >= Wi) {
                            yiWi += Wi*SY;
                            xi = 0;
                        }
                    }
                }

                if (j < BLOCK_SIZE) {
                    const float* last_inptr = inptrs[blocksize-1];
                    for (; j < BLOCK_SIZE; j++)
                        inptrs[j] = last_inptr;
                }

                for (int i = 0; i < BLOCK_SIZE*K0; i++)
                    sum[i] = 0.f;

                for (int c1 = 0, i = 0; c1 < nC; c1 += K0*C0, i++) {
                    int ofs_ij = i*iplanesize*C0;
                    for (j = 0; j < BLOCK_SIZE; j++) {
                        const float* x = &inptrs[j][ofs_ij];
                        for (int c0 = 0; c0 < C0; c0++) {
                            float xc = x[c0];
                            for (int k = 0; k < K0; k++) {
                                float w = wptr[c1 + c0*K0 + k];
                                sum[K0*j + k] += xc*w;
                            }
                        }
                    }
                }

                if (activation) {
                    if (resptr) {
                        for (j = 0; j < blocksize*K0; j++) {
                            float v = sum[j]*scale[j] + bias[j] + resptr[j];
                            sum[j] = v;
                        }
                    } else {
                        for (j = 0; j < blocksize*K0; j++) {
                            float v = sum[j]*scale[j] + bias[j];
                            sum[j] = v;
                        }
                    }
                    activation(sum, out, blocksize*K0, activParams);
                } else {
                    if (resptr) {
                        for (j = 0; j < blocksize*K0; j++) {
                            float v = sum[j]*scale[j] + bias[j] + resptr[j];
                            v = std::min(v*(v >= 0 ? 1.f : alpha), maxval);
                            out[j] = v;
                        }
                    } else {
                        for (j = 0; j < blocksize*K0; j++) {
                            float v = sum[j]*scale[j] + bias[j];
                            v = std::min(v*(v >= 0 ? 1.f : alpha), maxval);
                            out[j] = v;
                        }
                    }
                }
            }
        }
    });
}


typedef void (*conv_func_t)(const void* inp, const void* residual, void* out,
                            const ConvState& cs, const void* weights,
                            const float* scale, const float* bias,
                            const int32_t* ofs0, const int32_t** ofsptrs,
                            const uint8_t* mask);

class ConvLayerImpl : public ConvLayer
{
public:
    ConvLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        auto_pad = getAutoPadding(params);
        ceil_mode = params.get<bool>("ceil_mode", false);
        strides = params.getVector<int>("strides");
        dilations = params.getVector<int>("dilations");
        pads = params.getVector<int>("pads");
        ngroups = params.get<int>("group", 1);
        fused_batch_norm = false;
        add_residual = false;
    }

    virtual std::ostream& dumpAttrs(std::ostream& strm, int indent) const CV_OVERRIDE
    {
        prindent(strm, indent);
        strm << "ngroups: " << ngroups << ",\n";

        /*prindent(strm, indent);
        strm << "ksizes: [";
        for (int k = 0; k < wshape0.ndims; k++)
            strm << (k > 0 ? ", " : "") << wshape0.size[k];
        strm << "],\n";*/

        prindent(strm, indent);
        strm << "dilations: [";
        for (size_t k = 0; k < dilations.size(); k++)
            strm << (k > 0 ? ", " : "") << params.dilations[k];
        strm << "],\n";

        prindent(strm, indent);
        strm << "pads: [";
        for (size_t k = 0; k < pads.size(); k++)
            strm << (k > 0 ? ", " : "") << params.pads[k];
        strm << "],\n";

        prindent(strm, indent);
        strm << "strides: [";
        for (size_t k = 0; k < strides.size(); k++)
            strm << (k > 0 ? ", " : "") << params.strides[k];
        strm << "],\n";

        if (batchNorm) {
            prindent(strm, indent);
            strm << "batch_norm: true,\n";
        }

        if (add_residual) {
            prindent(strm, indent);
            strm << "add_residual: true,\n";
        }

        if (activ) {
            prindent(strm, indent);
            strm << "activation: " << activ->name() << ",\n";
        }

        return strm;
    }

    int inferType(int inptype0) const
    {
        return inptype0;
    }

    virtual bool supportType(int, int depth) const CV_OVERRIDE
    {
        return depth == CV_32F;
    }

    virtual int supportBlockLayout(int input) const CV_OVERRIDE
    {
        int ninputs = (int)inputs.size();
        return input == 0 || (add_residual && input == ninputs-1) ? 1 : -1;
    }

    virtual void setWeights(const Mat& weights_, const Mat& bias_,
                            int C0, int accuracy) CV_OVERRIDE
    {
        CV_Assert(!weights_.empty());
        int wtype0 = weights_.type();
        CV_Assert(wtype0 == CV_32F || wtype0 == CV_16F || wtype0 == CV_16BF);
        CV_Assert(accuracy == -1 || accuracy == CV_32F);
        int wtype = accuracy < 0 ? CV_32F : accuracy;

        wshape0 = weights_.size();
        MatShape wshape1 = wshape0;
        bool depthwise = ngroups == wshape0[0] && wshape0[1] == 1;

        if (depthwise) {
            wshape1.layout = DATA_LAYOUT_BLOCK;
            wshape1.C = wshape1.size[0];
            wshape1.size[0] = (wshape1.size[0] + C0 - 1)/C0;
            for (int i = 2; i < wshape1.ndims; i++)
                wshape1.size[i-1] = wshape1.size[i];
            wshape1.size[wshape1.ndims-1] = C0;
            weights.fit(wshape1, wtype);

            repackDepthwiseConvWeights(weights_.data(), wtype0, weights.data(), wtype, wshape0, C0);
        } else {
            wshape1.ndims += 2;
            wshape1.size[wshape1.ndims-1] = wshape1.size[wshape1.ndims-2] = C0;
            wshape1.size[0] = (wshape1.size[0] + C0 - 1)/C0;
            wshape1.size[1] = (wshape1.size[1] + C0 - 1)/C0;
            weights.fit(wshape1, wtype);

            repackConvWeights(weights_.data(), wtype0, weights.data(), wtype, wshape0, C0);
        }

        if (!bias_.empty()) {
            CV_Assert(bias_.isContinuous() && bias_.total() == wshape0.size[0]);
            bias_.convertTo(bias, CV_32F);
        }
    }

    void fuseBatchNormWeights(const Ptr<Layer>& bnlayer)
    {
        BatchNormLayer* bn = dynamic_cast<BatchNormOp*>(bnlayer.get());
        CV_Assert(bn != nullptr);
        const Tensor &bn_scale = bn->scale, &bn_bias = bn->bias;

        CV_Assert(bn_scale.isContinuous() && bn_bias.isContinuous());
        CV_Assert(bn_scale.type() == CV_32F && bn_bias.type() == CV_32F);
        CV_Assert(bn_scale.total() == bn_bias.total());
        size_t K = bn_scale.total();
        CV_Assert(bias.empty() || (bias.type() == CV_32F && bias.total() == K));
        const float* bias_data = bias.ptr<float>();

        fused_scale.fit(MatShape(1, &K), CV_32F);
        fused_bias.fit(MatShape(1, &K), CV_32F);

        const float* bn_scale_data = bn_scale.ptr<float>();
        const float* bn_bias_data = bn_bias.ptr<float>();
        float* fused_scale_data = fused_scale.ptr<float>();
        float* fused_bias_data = fused_bias.ptr<float>();

        // (sum(x*w) + bias)*bn_scale + bn_bias => sum(x*w)*fused_scale + fused_bias,
        // where fused_scale = bn_scale and fused_bias = bias*bn_scale + bn_bias.
        for (size_t i = 0; i < K; i++) {
            fused_scale_data[i] = bn_scale_data[i];
            fused_bias_data[i] = (bias_data ? bn_scale_data[i]*bias_data[i] : 0.f) + bn_bias_data[i];
        }
    }

    virtual bool fuseBatchNorm(const Op& op) override
    {
        BatchNormOp* bn = dynamic_cast<BatchNormOp*>(op.get());
        if (batchNorm || !bn)
            return false;
        batchNorm = op;
        return true;
    }

    virtual bool fuseActivation(const Op& op) override
    {
        ElemwiseOp* activ_ptr = dynamic_cast<ElemwiseOp*>(op.get());
        if (activ || activ_ptr->maxNumInputs() != 1 || !activ_ptr || !activ_ptr->getActivation(CV_32F))
            return false;
        activ = op;
        return true;
    }

    virtual int64_t getFLOPS(const std::vector<SizeType> &inputs,
                             const std::vector<SizeType> &outputs) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() >= 2);
        CV_Assert(outputs.size() == 1);
        // probably, there should be a coefficient in the case of complex reduction functions
        MatShape inpsize = inputs[0].size, wsize = inputs[1].size;
        int C = inpsize.size[1]*inpsize.size[inpsize.ndims-1];
        size_t ksize = wsize.total();
        return (int64_t)((inputs[0].size.total()/C)*ksize/params.ngroups);
    }

    virtual void getTypes(const Net2& net, const Graph& graph,
                            const std::vector<Arg>& inpargs,
                            const std::vector<int>& inptypes,
                            const std::vector<Arg>& outargs,
                            std::vector<int>& outtypes) const CV_OVERRIDE
    {
        int ninputs = (int)inpargs.size(), noutputs = (int)outargs.size();
        CV_Assert(minNumInputs() <= ninputs && ninputs <= maxNumInputs());
        CV_Assert((int)inptypes.size() == ninputs);
        CV_Assert(noutputs == 1);

        outtypes.assign(1, inferType(inptypes[0]));
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
        if (add_residual)
            ninputs--;

        const MatShape& inpsize = inpshapes[0];
        MatShape wsize = ninputs > 1 ? inpshapes[1] : wshape0;

        outshapes[0] = convInferShape(net, inpsize, params, wsize, symbolic);
    }

    virtual void forward(Net2& net, Graph& graph,
                        const std::vector<Tensor>& inputs,
                        std::vector<Tensor>& outputs,
                        std::vector<Buffer>& tempbufs) CV_OVERRIDE
    {
        size_t ninputs = inputs.size();
        CV_Assert(minNumInputs() <= ninputs && ninputs <= maxNumInputs());
        const Tensor& inp = inputs[0];
        const Tensor* residual = nullptr;
        const void* resptr = nullptr;
        int inptype = inp.type();
        MatShape inpsize = inp.size();
        CV_Assert(inpsize.layout == DATA_LAYOUT_BLOCK);
        CV_Assert(inp.isContinuous());

        if (add_residual) {
            residual = &inputs[ninputs-1];
            resptr = residual->data();
            ninputs--;
        }

        bool dynamic_weights = ninputs > 1;
        if (dynamic_weights) {
            setWeights(inputs[1], ninputs > 2 ? inputs[2] : Tensor(),
                       inpsize.size[inpsize.ndims-1], net.getAccuracy());
        }

        MatShape outsize = convInferShape(net, inpsize, params, wshape0);
        int outtype = inferType(inptype);
        outputs.resize(1);
        Tensor& out = outputs[0];
        out.fitSameDevice(inp, outsize, outtype);
        CV_Assert(out.isContinuous());

        if (add_residual) {
            CV_Assert(outsize == residual->size());
            CV_Assert(outtype == residual->type());
        }

        CV_Assert(inpsize.layout == DATA_LAYOUT_BLOCK);
        int nspatialdims = inpsize.ndims - 3;
        CV_Assert(wshape0.ndims == nspatialdims+2);

        if (inp.empty())
            return;

        const void* inptr = inp.data();
        void* outptr = out.data();
        const void* wptr = weights.data();

        int64_t ksize = 1;
        for (int i = 0; i < nspatialdims; i++)
            ksize *= wshape0.size[wshape0.ndims - nspatialdims + i];
        AutoBuffer<int64_t> buf(ksize*2);
        int64_t* ofstab = buf.data();
        int* yxtab = (int*)(ofstab + ksize);

        ConvState cs = initConvState(net, inpsize, wshape0, params, activ, yxtab, ofstab);
        bool conv1x1 = cs.Hk == 1 && cs.Wk == 1;
        bool depthwise = cs.ngroups == cs.C;
        const float* bias_data = bias.ptr<float>();

        if (batchNorm) {
            fuseBatchNormWeights();
            bias_data = fused_bias.ptr<float>();
        }

        if (depthwise) {
            depthwise_conv2d_t func = getDepthwiseConv2DFunc(inptype);
            CV_Assert(func != nullptr);

            func(inptr, resptr, outptr, cs, wptr,
                 fused_scale.ptr<float>(), bias_data);
        } else {
            if (!conv1x1 && (ofs0.empty() || !cs.sameShape(prev_cs))) {
                conv2d_init_tables(cs, ofsbuf, ofs0, ofsptrs, mask);
                prev_cs = cs;
            }

            conv_func_t func = conv1x1 ?
                (inptype == CV_32F ? conv2d_1x1_32f : nullptr) :
                (inptype == CV_32F ? conv2d_32f : nullptr);
            CV_Assert(func != nullptr);

            func(inptr, resptr, outptr, cs, wptr,
                 fused_scale.ptr<float>(), bias_data, ofs0.data(),
                 (const int32_t**)ofsptrs.data(), mask.data());
        }

        if (dynamic_weights) {
            // to keep memory footprint low in the case of
            // very rare situation of dynamic convolution weights,
            // we release temporarily allocated and reordered copy of the weights
            weights.release();
        }
    }

    Ptr<Layer> activ;
    Mat weights, bias, fused_scale, fused_bias;
    MatShape wshape0;
    ConvState prev_cs;
    std::vector<int32_t> ofsbuf;
    std::vector<int32_t> ofs0;
    std::vector<int32_t*> ofsptrs;
    std::vector<uint8_t> mask;
    bool fused_batch_norm;
};

Ptr<ConvLayer> ConvOp::create(const LayerParams& params)
{
    return std::make_shared<ConvOpImpl>(params);
}

}}

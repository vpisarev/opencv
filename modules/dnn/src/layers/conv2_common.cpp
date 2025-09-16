// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../net_impl.hpp"
#include "layers_common.hpp"
#include "conv2_common.hpp"
#include <math.h>

namespace cv { namespace dnn {

CV__DNN_INLINE_NS_BEGIN

AutoPadding getAutoPadding(const LayerParams& params)
{
    std::string auto_pad = params.get<std::string>("auto_pad", "NOTSET");
    if (auto_pad == "NOTSET")
        return AUTO_PAD_NONE;
    if (auto_pad == "SAME_UPPER")
        return AUTO_PAD_SAME_UPPER;
    if (auto_pad == "SAME_LOWER")
        return AUTO_PAD_SAME_LOWER;
    if (auto_pad != "VALID") {
        CV_Error_(Error::StsBadArg, ("invalid auto_pad value '%s'", auto_pad.c_str()));
    }
    return AUTO_PAD_VALID;
}

static inline void getPadding(const std::vector<int>& pads,
                              const std::vector<int>& kernelShape,
                              int dim, AutoPadding autoPad,
                              int& pad0, int& pad1)
{
    int nspatialdims = int(kernelShape.size());
    CV_Assert(0 <= dim && dim < nspatialdims);

    if (autoPad == AUTO_PAD_NONE || autoPad == AUTO_PAD_VALID) {
        if (!pads.empty()) {
            pad0 = pads[dim];
            pad1 = pads[dim + nspatialdims];
        } else {
            pad0 = pad1 = 0;
        }
    } else {
        CV_Assert(autoPad == AUTO_PAD_SAME_LOWER || autoPad == AUTO_PAD_SAME_UPPER);
        pad0 = pad1 = kernelShape[dim]/2;
        if (pad0*2 == kernelShape[dim]) {
            pad0 -= autoPad == AUTO_PAD_SAME_UPPER;
            pad1 -= autoPad == AUTO_PAD_SAME_LOWER;
        }
    }
}

// computes shape of the output tensor of convolution
// (including depth-wise convolution), max pooling or average pooling operations
MatShape convInferShape(const MatShape& inpShape, const MatShape& wshape,
                        const std::vector<int>& kernelShape, int ngroups,
                        const std::vector<int>& strides,
                        const std::vector<int>& dilations,
                        const std::vector<int>& pads,
                        AutoPadding autoPad, bool ceilMode)
{
    int blockLayout = inpShape.layout == DATA_LAYOUT_BLOCK;
    int ndims = inpShape.dims;
    size_t nspatialdims = (size_t)(ndims - 2 - blockLayout);
    MatShape outshape = inpShape;
    int kshape[MatShape::MAX_DIMS];

    if (!kernelShape.empty()) {
        size_t kshape_size = kernelShape.size();
        CV_Assert(kshape_size == nspatialdims || kshape_size == nspatialdims+2);
        for (size_t i = 0; i < nspatialdims; i++)
            kshape[i] = kernelShape[kshape_size - nspatialdims + i];
    } else {
        CV_Assert(!wshape.empty() && wshape.dims == nspatialdims + 2);
        for (size_t i = 0; i < nspatialdims; i++)
            kshape[i] = wshape[wshape.dims - nspatialdims + i];
    }

    if (ngroups == 0 || wshape.empty()) {
        outshape[1] = inpShape[1];
    } else if (blockLayout) {
        int C0 = inpShape[ndims-1];
        outshape[1] = (wshape[0] + C0 - 1)/C0;
    } else {
        outshape[1] = wshape[0];
    }

    CV_Assert(strides.empty() || strides.size() == nspatialdims);
    CV_Assert(dilations.empty() || dilations.size() == nspatialdims);
    CV_Assert(autoPad == AUTO_PAD_NONE || pads.empty());
    CV_Assert(pads.empty() || pads.size() == nspatialdims*2);

    for (size_t i = 0; i < nspatialdims; i++) {
        int inpsz = inpShape[i+2], k_i = kshape[i];
        int stride = strides.empty() ? 1 : strides[i];
        int dilation = dilations.empty() ? 1 : dilations[i];
        int outsz;
        if (autoPad == AUTO_PAD_NONE || autoPad == AUTO_PAD_VALID) {
            int pad = 0;
            if (!pads.empty()) {
                pad = pads[i] + pads[i + nspatialdims];
            }
            outsz = (inpsz + pad - 1 - dilation * (k_i - 1) + (ceilMode ? stride - 1 : 0)) / stride + 1;
        } else {
            if (ceilMode)
                outsz = (inpsz + stride - 1)/stride;
            else
                outsz = (inpsz - 1)/stride + 1;
        }
        outshape[i + 2] = outsz;
    }

    if (blockLayout) {
        outshape.C = ngroups == 0 || wshape.empty() ? inpShape.C : wshape[0];
    } else {
        outshape.C = 0;
    }

    return outshape;
}


void initPoolingState(const MatShape& inpShape,
                      const MatShape& outShape,
                      const std::vector<int>& kernelShape,
                      const std::vector<int>& strides,
                      const std::vector<int>& dilations,
                      const std::vector<int>& pads,
                      AutoPadding autoPad, bool ceilMode,
                      int mindims, ConvState& cs)
{
    int kdims = int(kernelShape.size());
    CV_Assert(kdims <= ConvState::MAX_CONV_DIMS);
    CV_Assert(strides.empty() || (strides.size() == size_t(kdims)));
    CV_Assert(dilations.empty() || (dilations.size() == size_t(kdims)));
    CV_Assert(pads.empty() || (pads.size() == size_t(kdims*2)));
    CV_Assert(inpShape.layout == DATA_LAYOUT_BLOCK);
    CV_Assert(inpShape.dims == 3 || (5 <= inpShape.dims &&
            inpShape.dims <= 3 + ConvState::MAX_CONV_DIMS));

    int C = inpShape.C;
    cs.inpshape = inpShape;
    cs.outshape = outShape;
    cs.ngroups = C;
    cs.nspatialdims = inpShape.dims == 3 ? 1 : inpShape.dims - 3;

    cs.fastActivation = FAST_ACTIV_NONE;
    cs.activation = nullptr;
    for (int i = 0; i < ConvState::ACTIV_MAX_PARAMS; i++) {
        cs.activParams[i] = 0.f;
    }

    CV_Assert(cs.nspatialdims == int(kdims));

    for (int i = 0; i < kdims; i++) {
        cs.kshape[i] = kernelShape[i];
        CV_Assert(kernelShape[i] > 0);

        cs.strides[i] = strides.empty() ? 1 : strides[i];
        cs.dilations[i] = dilations.empty() ? 1 : dilations[i];

        CV_Assert(cs.strides[i] > 0);
        CV_Assert(cs.dilations[i] > 0);

        int pad0, pad1;
        getPadding(pads, kernelShape, int(i), autoPad, pad0, pad1);
        CV_Assert_N(pad0 >= 0, pad1 >= 0);
        cs.pads[i] = pad0;
        cs.pads[i + cs.nspatialdims] = pad1;

        int inner0 = (pad0 + cs.strides[i] - 1)/cs.strides[i];
        int inner1 = (inpShape[i+2] - (cs.kshape[i] - 1)*cs.dilations[i] + pad0)/cs.strides[i];
        inner1 += inner1*cs.strides[i] - pad0 + (cs.kshape[i] - 1)*cs.dilations[i] < inpShape[i+2];
        inner1 = std::min(inner1, outShape[i+2]);
        if (inner0 >= inner1) {
            inner0 = inner1 = outShape[i+2];
        }
        cs.inner[i] = inner0;
        cs.inner[i + nspatialdims] = inner1;
    }


    //std::vector<int> coordtab;
    //std::vector<int> ofstab;
}

CV__DNN_INLINE_NS_END

}
}

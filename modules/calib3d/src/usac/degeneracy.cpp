// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../usac.hpp"

namespace cv { namespace usac {
class HomographyDegeneracyImpl : public HomographyDegeneracy {
private:
    const double * const points;
    const int sample_size;
public:
    explicit HomographyDegeneracyImpl (const Mat &points_, int sample_size_) :
            points ((double *)points_.data), sample_size (sample_size_) {}

    inline bool isSampleGood (const std::vector<int>& sample) const override {
        const int smpl1 = 4*sample[0], smpl2 = 4*sample[1], smpl3 = 4*sample[2], smpl4 = 4*sample[3];
        const float x1 = points[smpl1], y1 = points[smpl1+1], X1 = points[smpl1+2], Y1 = points[smpl1+3];
        const float x2 = points[smpl2], y2 = points[smpl2+1], X2 = points[smpl2+2], Y2 = points[smpl2+3];
        const float x3 = points[smpl3], y3 = points[smpl3+1], X3 = points[smpl3+2], Y3 = points[smpl3+3];
        const float x4 = points[smpl4], y4 = points[smpl4+1], X4 = points[smpl4+2], Y4 = points[smpl4+3];

        const float ab_cross_x = y1 - y2, ab_cross_y = x2 - x1, ab_cross_z = x1 * y2 - y1 * x2;
        const float AB_cross_x = Y1 - Y2, AB_cross_y = X2 - X1, AB_cross_z = X1 * Y2 - Y1 * X2;

        // check ab cross with point c and d
        if ((ab_cross_x * x3 + ab_cross_y * y3 + ab_cross_z) *
            (AB_cross_x * X3 + AB_cross_y * Y3 + AB_cross_z) < 0)
            return false;
        if ((ab_cross_x * x4 + ab_cross_y * y4 + ab_cross_z) *
            (AB_cross_x * X4 + AB_cross_y * Y4 + AB_cross_z) < 0)
            return false;

        const float cd_cross_x = y3 - y4, cd_cross_y = x4 - x3, cd_cross_z = x3 * y4 - y3 * x4;
        const float CD_cross_x = Y3 - Y4, CD_cross_y = X4 - X3, CD_cross_z = X3 * Y4 - Y3 * X4;

        // check ab cross with point a and b
        if ((cd_cross_x * x1 + cd_cross_y * y1 + cd_cross_z) *
            (CD_cross_x * X1 + CD_cross_y * Y1 + CD_cross_z) < 0)
            return false;
        if ((cd_cross_x * x2 + cd_cross_y * y2 + cd_cross_z) *
            (CD_cross_x * X2 + CD_cross_y * Y2 + CD_cross_z) < 0)
            return false;
        return true;
    }

    Ptr<Degeneracy> clone() const override {
        return makePtr<HomographyDegeneracyImpl>(*this);
    }
};

Ptr<HomographyDegeneracy> HomographyDegeneracy::create (const Mat &points_, int sample_size_) {
    return makePtr<HomographyDegeneracyImpl>(points_, sample_size_);
}
}}

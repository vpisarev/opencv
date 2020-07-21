// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../usac.hpp"

namespace cv { namespace usac {
class HomographyEstimatorImpl : public HomographyEstimator {
private:
    const Ptr<MinimalSolver> min_solver;
    const Ptr<NonMinimalSolver> non_min_solver;
    const Ptr<Degeneracy> degeneracy;
public:
    HomographyEstimatorImpl (const Ptr<MinimalSolver> &min_solver_,
            const Ptr<NonMinimalSolver> &non_min_solver_, const Ptr<Degeneracy> &degeneracy_) :
            min_solver (min_solver_), non_min_solver (non_min_solver_), degeneracy (degeneracy_) {}

    inline int estimateModels (const std::vector<int> &sample, std::vector<Mat> &models) const override {
        if (! degeneracy->isSampleGood(sample)) return 0;
        return min_solver->estimate (sample, models);
    }
    int estimateModelNonMinimalSample(const std::vector<int> &sample, int sample_size,
            std::vector<Mat> &models, const std::vector<double> &weights) const override {
        return non_min_solver->estimate (sample, sample_size, models, weights);
    };
    int getMaxNumSolutions () const override {
        return min_solver->getMaxNumberOfSolutions();
    }
    int getMaxNumSolutionsNonMinimal () const override {
        return non_min_solver->getMaxNumberOfSolutions();
    }
    int getMinimalSampleSize () const override {
        return min_solver->getSampleSize();
    }
    int getNonMinimalSampleSize () const override {
        return non_min_solver->getMinimumRequiredSampleSize();
    }
    Ptr<Estimator> clone() const override {
        return makePtr<HomographyEstimatorImpl>(min_solver->clone(), non_min_solver->clone(),
                degeneracy->clone(0 /*we don't need state here*/));
    }
};
Ptr<HomographyEstimator> HomographyEstimator::create (const Ptr<MinimalSolver> &min_solver_,
        const Ptr<NonMinimalSolver> &non_min_solver_, const Ptr<Degeneracy> &degeneracy_) {
    return makePtr<HomographyEstimatorImpl>(min_solver_, non_min_solver_, degeneracy_);
}

/////////////////////////////////////////////////////////////////////////
class FundamentalEstimatorImpl : public FundamentalEstimator {
private:
    const Ptr<MinimalSolver> min_solver;
    const Ptr<NonMinimalSolver> non_min_solver;
    const Ptr<Degeneracy> degeneracy;
public:
    FundamentalEstimatorImpl (const Ptr<MinimalSolver> &min_solver_,
            const Ptr<NonMinimalSolver> &non_min_solver_, const Ptr<Degeneracy> &degeneracy_) :
         min_solver (min_solver_), non_min_solver (non_min_solver_), degeneracy (degeneracy_) {}

    inline int
    estimateModels(const std::vector<int> &sample, std::vector<Mat> &models) const override {
        std::vector<Mat> F;
        const int models_count = min_solver->estimate(sample, F);
        int valid_models_count = 0;

        for (int i = 0; i < models_count; i++)
            if (degeneracy->isModelValid(F[i], sample))
                models[valid_models_count++] = F[i];

        return valid_models_count;
    }
    int estimateModelNonMinimalSample(const std::vector<int> &sample, int sample_size,
            std::vector<Mat> &models, const std::vector<double> &weights) const override {
        return non_min_solver->estimate(sample, sample_size, models, weights);
    }
    int getMaxNumSolutions () const override {
        return min_solver->getMaxNumberOfSolutions();
    }
    int getMinimalSampleSize () const override {
        return min_solver->getSampleSize();
    }
    int getNonMinimalSampleSize () const override {
        return non_min_solver->getMinimumRequiredSampleSize();
    }
    int getMaxNumSolutionsNonMinimal () const override {
        return non_min_solver->getMaxNumberOfSolutions();
    }
    Ptr<Estimator> clone() const override {
        return makePtr<FundamentalEstimatorImpl>(min_solver->clone(), non_min_solver->clone(),
                degeneracy->clone(0));
    }
};
Ptr<FundamentalEstimator> FundamentalEstimator::create (const Ptr<MinimalSolver> &min_solver_,
        const Ptr<NonMinimalSolver> &non_min_solver_, const Ptr<Degeneracy> &degeneracy_) {
    return makePtr<FundamentalEstimatorImpl>(min_solver_, non_min_solver_, degeneracy_);
}

///////////////////////////////////////////// ERROR /////////////////////////////////////////
// Symmetric Reprojection Error
class ReprojectedErrorSymmetricImpl : public ReprojectionErrorSymmetric {
private:
    const Mat * points_mat;
    const float * const points;
    float m11, m12, m13, m21, m22, m23, m31, m32, m33;
    float minv11, minv12, minv13, minv21, minv22, minv23, minv31, minv32, minv33;
    std::vector<float> errors;
public:
    explicit ReprojectedErrorSymmetricImpl (const Mat &points_) :
            points_mat(&points_), points ((float *) points_.data), errors(points_.rows) {}

    inline void setModelParameters (const Mat &model) override {
        const auto * const m = (double *) model.data;
        m11=static_cast<float>(m[0]); m12=static_cast<float>(m[1]); m13=static_cast<float>(m[2]);
        m21=static_cast<float>(m[3]); m22=static_cast<float>(m[4]); m23=static_cast<float>(m[5]);
        m31=static_cast<float>(m[6]); m32=static_cast<float>(m[7]); m33=static_cast<float>(m[8]);

        const Mat model_inv = model.inv();
        const auto * const minv = (double *) model_inv.data;
        minv11=(float)minv[0]; minv12=(float)minv[1]; minv13=(float)minv[2];
        minv21=(float)minv[3]; minv22=(float)minv[4]; minv23=(float)minv[5];
        minv31=(float)minv[6]; minv32=(float)minv[7]; minv33=(float)minv[8];
    }
    inline float getError (int point_idx) const override {
        const int smpl = 4*point_idx;
        const float x1=points[smpl], y1=points[smpl+1], x2=points[smpl+2], y2=points[smpl+3];
        const float est_z2 =  m31 * x1 + m32 * y1 + m33,
                    est_x2 = (m11 * x1 + m12 * y1 + m13) / est_z2,
                    est_y2 = (m21 * x1 + m22 * y1 + m23) / est_z2;
        const float est_z1 =  minv31 * x2 + minv32 * y2 + minv33,
                    est_x1 = (minv11 * x2 + minv12 * y2 + minv13) / est_z1,
                    est_y1 = (minv21 * x2 + minv22 * y2 + minv23) / est_z1;
        return ((x2 - est_x2) * (x2 - est_x2) + (y2 - est_y2) * (y2 - est_y2) +
                (x1 - est_x1) * (x1 - est_x1) + (y1 - est_y1) * (y1 - est_y1)) / 2;
    }
    void computeErrors (const Mat &model) override {
        setModelParameters(model);
        for (int point_idx = 0; point_idx < points_mat->rows; point_idx++) {
            const int smpl = 4*point_idx;
            const float x1=points[smpl], y1=points[smpl+1], x2=points[smpl+2], y2=points[smpl+3];
            const float est_z2 =  m31 * x1 + m32 * y1 + m33,
                        est_x2 = (m11 * x1 + m12 * y1 + m13) / est_z2,
                        est_y2 = (m21 * x1 + m22 * y1 + m23) / est_z2;
            const float est_z1 =  minv31 * x2 + minv32 * y2 + minv33,
                        est_x1 = (minv11 * x2 + minv12 * y2 + minv13) / est_z1,
                        est_y1 = (minv21 * x2 + minv22 * y2 + minv23) / est_z1;
            errors[point_idx] = ((x2 - est_x2) * (x2 - est_x2) + (y2 - est_y2) * (y2 - est_y2) +
                                 (x1 - est_x1) * (x1 - est_x1) + (y1 - est_y1) * (y1 - est_y1))/2;
        }
    }
    const std::vector<float> &getErrors () const override { return errors; }
    Ptr<Error> clone () const override {
        return makePtr<ReprojectedErrorSymmetricImpl>(*points_mat);
    }
};
Ptr<ReprojectionErrorSymmetric>
ReprojectionErrorSymmetric::create(const Mat &points) {
    return makePtr<ReprojectedErrorSymmetricImpl>(points);
}

// Forward Reprojection Error
class ReprojectedErrorForwardImpl : public ReprojectionErrorForward {
private:
    const Mat * points_mat;
    const float * const points;
    float m11, m12, m13, m21, m22, m23, m31, m32, m33;
    std::vector<float> errors;
public:
    explicit ReprojectedErrorForwardImpl (const Mat &points_)
        : points_mat(&points_), points ((float *)points_.data), errors(points_.rows) {}

    inline void setModelParameters (const Mat &model) override {
        const auto * const m = (double *) model.data;
        m11=static_cast<float>(m[0]); m12=static_cast<float>(m[1]); m13=static_cast<float>(m[2]);
        m21=static_cast<float>(m[3]); m22=static_cast<float>(m[4]); m23=static_cast<float>(m[5]);
        m31=static_cast<float>(m[6]); m32=static_cast<float>(m[7]); m33=static_cast<float>(m[8]);
    }
    inline float getError (int point_idx) const override {
        const int smpl = 4*point_idx;
        const float x1 = points[smpl], y1 = points[smpl+1], x2 = points[smpl+2], y2 = points[smpl+3];
        const float est_z2 =  m31 * x1 + m32 * y1 + m33,
                    est_x2 = (m11 * x1 + m12 * y1 + m13) / est_z2,
                    est_y2 = (m21 * x1 + m22 * y1 + m23) / est_z2;
        return (x2 - est_x2) * (x2 - est_x2) + (y2 - est_y2) * (y2 - est_y2);
    }
    void computeErrors (const Mat &model) override {
        setModelParameters(model);
        for (int point_idx = 0; point_idx < points_mat->rows; point_idx++) {
            const int smpl = 4*point_idx;
            const float x1=points[smpl], y1=points[smpl+1], x2=points[smpl+2], y2=points[smpl+3];
            const float est_z2 =  m31 * x1 + m32 * y1 + m33,
                        est_x2 = (m11 * x1 + m12 * y1 + m13) / est_z2,
                        est_y2 = (m21 * x1 + m22 * y1 + m23) / est_z2;
            errors[point_idx] = (x2 - est_x2) * (x2 - est_x2) + (y2 - est_y2) * (y2 - est_y2);
        }
    }
    const std::vector<float> &getErrors () const override { return errors; }
    Ptr<Error> clone () const override {
        return makePtr<ReprojectedErrorForwardImpl>(*points_mat);
    }
};
Ptr<ReprojectionErrorForward>
ReprojectionErrorForward::create(const Mat &points) {
    return makePtr<ReprojectedErrorForwardImpl>(points);
}

class SampsonErrorImpl : public SampsonError {
private:
    const Mat * points_mat;
    const float * const points;
    float m11, m12, m13, m21, m22, m23, m31, m32, m33;
    std::vector<float> errors;
public:
    explicit SampsonErrorImpl (const Mat &points_) :
            points_mat(&points_), points ((float *) points_.data), errors(points_.rows) {}

    inline void setModelParameters (const Mat &model) override {
        const auto * const m = (double *) model.data;
        m11=static_cast<float>(m[0]); m12=static_cast<float>(m[1]); m13=static_cast<float>(m[2]);
        m21=static_cast<float>(m[3]); m22=static_cast<float>(m[4]); m23=static_cast<float>(m[5]);
        m31=static_cast<float>(m[6]); m32=static_cast<float>(m[7]); m33=static_cast<float>(m[8]);
    }

    /*
     *                                       (pt2^t * F * pt1)^2)
     * Sampson error = ------------------------------------------------------------------------
     *                  (((F⋅pt1)(0))^2 + ((F⋅pt1)(1))^2 + ((F^t⋅pt2)(0))^2 + ((F^t⋅pt2)(1))^2)
     *
     * [ x2 y2 1 ] * [ F(1,1)  F(1,2)  F(1,3) ]   [ x1 ]
     *               [ F(2,1)  F(2,2)  F(2,3) ] * [ y1 ]
     *               [ F(3,1)  F(3,2)  F(3,3) ]   [ 1  ]
     *
     */
    inline float getError (int point_idx) const override {
        const int smpl = 4*point_idx;
        const float x1=points[smpl], y1=points[smpl+1], x2=points[smpl+2], y2=points[smpl+3];
        const float F_pt1_x = m11 * x1 + m12 * y1 + m13,
                    F_pt1_y = m21 * x1 + m22 * y1 + m23;
        const float pt2_F_x = x2 * m11 + y2 * m21 + m31,
                    pt2_F_y = x2 * m12 + y2 * m22 + m32;
        const float pt2_F_pt1 = x2 * F_pt1_x + y2 * F_pt1_y + m31 * x1 + m32 * y1 + m33;
        return (pt2_F_pt1 * pt2_F_pt1) / (F_pt1_x * F_pt1_x + F_pt1_y * F_pt1_y +
                                          pt2_F_x * pt2_F_x + pt2_F_y * pt2_F_y);
    }
    void computeErrors (const Mat &model) override {
        setModelParameters(model);
        for (int point_idx = 0; point_idx < points_mat->rows; point_idx++) {
            const int smpl = 4*point_idx;
            const float x1=points[smpl], y1=points[smpl+1], x2=points[smpl+2], y2=points[smpl+3];
            const float F_pt1_x = m11 * x1 + m12 * y1 + m13,
                        F_pt1_y = m21 * x1 + m22 * y1 + m23;
            const float pt2_F_x = x2 * m11 + y2 * m21 + m31,
                        pt2_F_y = x2 * m12 + y2 * m22 + m32;
            const float pt2_F_pt1 = x2 * F_pt1_x + y2 * F_pt1_y + m31 * x1 + m32 * y1 + m33;
            errors[point_idx] = (pt2_F_pt1 * pt2_F_pt1) / (F_pt1_x * F_pt1_x + F_pt1_y * F_pt1_y +
                                                           pt2_F_x * pt2_F_x + pt2_F_y * pt2_F_y);
        }
    }
    const std::vector<float> &getErrors () const override { return errors; }
    Ptr<Error> clone () const override {
        return makePtr<SampsonErrorImpl>(*points_mat);
    }
};
Ptr<SampsonError>
SampsonError::create(const Mat &points) {
    return makePtr<SampsonErrorImpl>(points);
}

////////////////////////////////////// NORMALIZING TRANSFORMATION /////////////////////////
class NormTransformImpl : public NormTransform {
private:
    const float * const points;
public:
    explicit NormTransformImpl (const Mat &points_) : points((float*)points_.data) {}

    // Compute normalized points and transformation matrices.
    void getNormTransformation (Mat& norm_points, const std::vector<int> &sample,
                                int sample_size, Matx33d &T1, Matx33d &T2) const override {
        double mean_pts1_x = 0, mean_pts1_y = 0, mean_pts2_x = 0, mean_pts2_y = 0;

        // find average of each coordinate of points.
        int smpl;
        for (int i = 0; i < sample_size; i++) {
            smpl = 4 * sample[i];

            mean_pts1_x += points[smpl    ];
            mean_pts1_y += points[smpl + 1];
            mean_pts2_x += points[smpl + 2];
            mean_pts2_y += points[smpl + 3];
        }

        mean_pts1_x /= sample_size; mean_pts1_y /= sample_size;
        mean_pts2_x /= sample_size; mean_pts2_y /= sample_size;

        double avg_dist1 = 0, avg_dist2 = 0, x1_m, y1_m, x2_m, y2_m;
        for (int i = 0; i < sample_size; i++) {
            smpl = 4 * sample[i];
            /*
             * Compute a similarity transform T that takes points xi
             * to a new set of points x̃i such that the centroid of
             * the points x̃i is the coordinate origin and their
             * average distance from the origin is √2
             *
             * sqrt(x̃*x̃ + ỹ*ỹ) = sqrt(2)
             * ax*ax + by*by = 2
             */
            x1_m = points[smpl    ] - mean_pts1_x;
            y1_m = points[smpl + 1] - mean_pts1_y;
            x2_m = points[smpl + 2] - mean_pts2_x;
            y2_m = points[smpl + 3] - mean_pts2_y;

            avg_dist1 += sqrt (x1_m * x1_m + y1_m * y1_m);
            avg_dist2 += sqrt (x2_m * x2_m + y2_m * y2_m);
        }

        // scale
        avg_dist1 = M_SQRT2 / (avg_dist1 / sample_size);
        avg_dist2 = M_SQRT2 / (avg_dist2 / sample_size);

        const double transl_x1 = -mean_pts1_x * avg_dist1, transl_y1 = -mean_pts1_y * avg_dist1;
        const double transl_x2 = -mean_pts2_x * avg_dist2, transl_y2 = -mean_pts2_y * avg_dist2;

        // transformation matrices
        T1 = Matx33d (avg_dist1, 0, transl_x1,0, avg_dist1, transl_y1,0, 0, 1);
        T2 = Matx33d (avg_dist2, 0, transl_x2,0, avg_dist2, transl_y2,0, 0, 1);

        norm_points = Mat_<float>(sample_size, 4); // normalized points Nx4 matrix
        auto * norm_points_ptr = (float *) norm_points.data;

        // Normalize points: Npts = T*pts    3x3 * 3xN
        const float avg_dist1f = (float)avg_dist1, avg_dist2f = (float)avg_dist2;
        const float transl_x1f = (float)transl_x1, transl_y1f = (float)transl_y1;
        const float transl_x2f = (float)transl_x2, transl_y2f = (float)transl_y2;
        for (int i = 0; i < sample_size; i++) {
            smpl = 4 * sample[i];
            (*norm_points_ptr++) = avg_dist1f * points[smpl    ] + transl_x1f;
            (*norm_points_ptr++) = avg_dist1f * points[smpl + 1] + transl_y1f;
            (*norm_points_ptr++) = avg_dist2f * points[smpl + 2] + transl_x2f;
            (*norm_points_ptr++) = avg_dist2f * points[smpl + 3] + transl_y2f;
        }
    }
};
Ptr<NormTransform> NormTransform::create (const Mat &points) {
    return makePtr<NormTransformImpl>(points);
}
}}

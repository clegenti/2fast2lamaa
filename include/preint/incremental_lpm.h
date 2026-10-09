#ifndef UGPM_INCREMENTAL_LPM_H
#define UGPM_INCREMENTAL_LPM_H

// Incremental version of the LPM preintegration (IterativeIntegrator, type LPM of
// ImuPreintegration): the same model, fed one sample at a time as the IMU data arrives, and queried
// at any time after the start, instead of integrated in one go over a batch of data for a set of
// times given beforehand.
//
// The model, as in the batch integrator:
//   - rotation: the gyroscope data interpolated linearly, integrated with one exponential map step
//     per step of the timeline (the value at the start of the step, held over the step). The timeline
//     is the accelerometer timestamps, each interval divided into steps of at most 1/min_freq;
//   - velocity and position: the accelerometer samples rotated into the start frame (by the rotation
//     at their timestamp), interpolated linearly, and integrated in closed form;
//   - covariance: the rotation block propagated step by step with the gyroscope variance, the
//     velocity and position blocks diagonal ((t - t0)*acc_var and (t - t0)^2*acc_var);
//   - bias Jacobians: numerical for the gyroscope bias (the rotation integrated again with each axis
//     of the gyroscope data shifted by kNumGyrBiasJacobianDelta), analytic for the accelerometer bias.
// The results are those of the batch integrator given the same timeline (up to rounding). Its
// timeline also holds the times it is queried at, and a point at start + kNumDtJacobianDelta, where
// this one only holds the accelerometer timestamps: steps split differently, so the two differ by
// the discretisation error of the rotation (a bound on which is the step size, 1/min_freq).
//
// Differences with the batch integrator:
//   - the accelerometer value at the start time is interpolated between the samples around it; the
//     batch integrator takes the value of the sample before it (while it interpolates the Jacobians);
//   - the time-shift Jacobians (d_delta_*_d_t) are not computed, and are zero;
//   - get() returns the covariance as integrated, without the bias uncertainty that
//     ImuPreintegration::get adds by default (as ImuPreintegration::get(..., 0, 0)).
//
// Usage: construct at the start time with the bias prior (subtracted from the data, as by the batch
// integrator), feed the samples of each stream in increasing time order, starting with at least one
// sample of each stream before the start time. A step is integrated as soon as the next accelerometer
// sample is covered by the gyroscope data; get(t) interpolates between the integrated samples, and
// extrapolates past the last one as the batch integrator does past the end of its data.

#include "types.h"
#include "math_utils.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ugpm
{

class IncrementalLpm
{
    public:
        // `extra_times`: additional steps boundaries of the rotation integration (sorted). Only meant
        // to reproduce the timeline of the batch integrator exactly, for testing
        IncrementalLpm(const double start_t, const PreintPrior& prior, const double acc_var, const double gyr_var,
                const double min_freq = 500.0, const std::vector<double>& extra_times = std::vector<double>())
            : start_t_(start_t)
            , prior_(prior)
            , acc_var_(acc_var)
            , gyr_var_(gyr_var)
            , min_freq_(min_freq)
            , extra_times_(extra_times)
        {
        }

        void addGyr(const ImuSample& sample)
        {
            if(!gyr_.empty() && sample.t <= gyr_.back().t)
            {
                throw std::invalid_argument("IncrementalLpm::addGyr: samples must be given in increasing time order");
            }
            gyr_.push_back(debiased(sample, prior_.gyr_bias));
            advance();
        }

        void addAcc(const ImuSample& sample)
        {
            if(!acc_.empty() && sample.t <= acc_.back().t)
            {
                throw std::invalid_argument("IncrementalLpm::addAcc: samples must be given in increasing time order");
            }
            acc_.push_back(debiased(sample, prior_.acc_bias));
            advance();
        }

        double startTime() const { return start_t_; }
        const PreintPrior& prior() const { return prior_; }

        // Whether the start has been integrated (the data around it received), which get() needs
        bool initialised() const { return !knots_.empty(); }

        // Time of the last accelerometer sample integrated: get() interpolates up to it, and
        // extrapolates past it. -infinity before initialisation
        double committedTime() const { return knots_.empty() ? -std::numeric_limits<double>::infinity() : knots_.back().t; }

        PreintMeas get(const double t) const
        {
            if(knots_.empty())
            {
                throw std::logic_error("IncrementalLpm::get: not initialised (no data around the start time yet)");
            }
            if(t < start_t_)
            {
                throw std::range_error("IncrementalLpm::get: query before the start time");
            }
            // The last knot at or before t
            size_t c = std::upper_bound(knots_.begin(), knots_.end(), t, [](const double v, const Knot& k) { return v < k.t; }) - knots_.begin() - 1;
            const Knot& kc = knots_[c];
            if(kc.t == t)
            {
                return toMeas(kc.t, kc.R, kc.cov_r, kc.R_pert, kc.v, kc.p, kc.d_v_d_bf, kc.d_v_d_bw, kc.d_p_d_bf, kc.d_p_d_bw);
            }

            // Rotation from the knot to t, on the steps of the knot's interval (t closing the last one)
            const bool has_next = (c + 1) < knots_.size();
            std::vector<double> bounds = has_next ? steps(kc.t, knots_[c+1].t) : steps(kc.t, t);
            while(!bounds.empty() && bounds.back() >= t) bounds.pop_back();
            bounds.push_back(t);
            Mat3 R = kc.R;
            Mat3 cov_r = kc.cov_r;
            Mat3 R_pert[3] = {kc.R_pert[0], kc.R_pert[1], kc.R_pert[2]};
            double s = kc.t;
            for(const double b : bounds)
            {
                rotStep(gyrAt(s), b - s, R, cov_r, R_pert);
                s = b;
            }

            // Rotated accelerometer value at t: between the knot and the next one, or extrapolated
            // from the last two (the batch integrator's last linear segment)
            Vec3 a_t;
            Mat3 d_a_d_bf, d_a_d_bw;
            if(has_next)
            {
                const Knot& kn = knots_[c+1];
                interpolate(kc, kn, t, a_t, d_a_d_bf, d_a_d_bw);
            }
            else if(c > 0)
            {
                interpolate(knots_[c-1], kc, t, a_t, d_a_d_bf, d_a_d_bw);
            }
            else
            {
                a_t = kc.a;
                d_a_d_bf = kc.d_a_d_bf;
                d_a_d_bw = kc.d_a_d_bw;
            }

            Vec3 v, p;
            Mat3 d_v_d_bf, d_v_d_bw, d_p_d_bf, d_p_d_bw;
            const double dt = t - kc.t;
            for(int axis = 0; axis < 3; ++axis)
            {
                v(axis) = kc.v(axis) + ((t - kc.t)*(kc.a(axis) + a_t(axis))/2.0);
                p(axis) = kc.p(axis) + kc.v(axis)*(t - kc.t) + ((kc.t - t)*(kc.t - t)*(2.0*kc.a(axis) + a_t(axis))/6.0);
                const Vec3 d0_bf = kc.d_a_d_bf.row(axis).transpose(), d1_bf = d_a_d_bf.row(axis).transpose();
                const Vec3 d0_bw = kc.d_a_d_bw.row(axis).transpose(), d1_bw = d_a_d_bw.row(axis).transpose();
                d_v_d_bf.row(axis) = (kc.d_v_d_bf.row(axis).transpose() + dt*(d0_bf + d1_bf)/2.0).transpose();
                d_v_d_bw.row(axis) = (kc.d_v_d_bw.row(axis).transpose() + dt*(d0_bw + d1_bw)/2.0).transpose();
                d_p_d_bf.row(axis) = (kc.d_p_d_bf.row(axis).transpose() + dt*kc.d_v_d_bf.row(axis).transpose() + dt*dt*(2.0*d0_bf + d1_bf)/6.0).transpose();
                d_p_d_bw.row(axis) = (kc.d_p_d_bw.row(axis).transpose() + dt*kc.d_v_d_bw.row(axis).transpose() + dt*dt*(2.0*d0_bw + d1_bw)/6.0).transpose();
            }
            return toMeas(t, R, cov_r, R_pert, v, p, d_v_d_bf, d_v_d_bw, d_p_d_bf, d_p_d_bw);
        }

    private:
        // The state of the integration at an accelerometer timestamp (or the start time)
        struct Knot
        {
            double t;
            Mat3 R;
            Mat3 R_pert[3];     // R integrated with the gyroscope data shifted on each axis
            Mat3 cov_r;
            Vec3 a;             // accelerometer data rotated into the start frame
            Mat3 d_a_d_bf;      // row i: Jacobian of a(i)
            Mat3 d_a_d_bw;
            Vec3 v, p;
            Mat3 d_v_d_bf, d_v_d_bw, d_p_d_bf, d_p_d_bw;
            size_t next_acc;    // index of the first accelerometer sample after t
        };

        double start_t_;
        PreintPrior prior_;
        double acc_var_;
        double gyr_var_;
        double min_freq_;
        std::vector<double> extra_times_;

        std::vector<ImuSample> gyr_;
        std::vector<ImuSample> acc_;
        std::vector<Knot> knots_;

        static ImuSample debiased(const ImuSample& s, const std::vector<double>& bias)
        {
            ImuSample out = s;
            for(int i = 0; i < 3; ++i) out.data[i] = s.data[i] - bias[i];
            return out;
        }

        static Vec3 sampleVec(const ImuSample& s) { return Vec3(s.data[0], s.data[1], s.data[2]); }

        // Gyroscope data interpolated linearly at t (the first or last segment past the data)
        Vec3 gyrAt(const double t) const
        {
            size_t j = std::lower_bound(gyr_.begin(), gyr_.end(), t, [](const ImuSample& s, const double v) { return s.t < v; }) - gyr_.begin();
            size_t i = (j == 0) ? 0 : std::min(j - 1, gyr_.size() - 2);
            const ImuSample& s0 = gyr_[i];
            const ImuSample& s1 = gyr_[i+1];
            Vec3 out;
            for(int a = 0; a < 3; ++a)
            {
                const double alpha = (s1.data[a] - s0.data[a]) / (s1.t - s0.t);
                out(a) = s0.data[a] + alpha*(t - s0.t);
            }
            return out;
        }

        // Boundaries of the steps of the rotation integration over (t0, t1], t1 last: the interval
        // cut into equal steps of at most 1/min_freq, and the extra times in it
        std::vector<double> steps(const double t0, const double t1) const
        {
            std::vector<double> out;
            const int n = std::max(1, (int)std::ceil((t1 - t0)*min_freq_ - 1e-9));
            const double q = (t1 - t0)/n;
            for(int i = 1; i < n; ++i) out.push_back(t0 + i*q);
            for(const double e : extra_times_)
            {
                if(e > t0 && e < t1) out.push_back(e);
            }
            std::sort(out.begin(), out.end());
            out.push_back(t1);
            return out;
        }

        // One step of the rotation integration, as the batch integrator's (closed form of the
        // exponential map and of the right Jacobian for the main rotation, expMap for the shifted ones)
        void rotStep(const Vec3& w, const double dt, Mat3& R, Mat3& cov_r, Mat3* R_pert) const
        {
            const Vec3 gyr_dt = w*dt;
            const double gyr_norm = gyr_dt.norm();
            Mat3 e_R = Mat3::Identity();
            Mat3 j_r = Mat3::Identity();
            if(gyr_norm > 0.0000000001)
            {
                Mat3 gyr_skew_mat;
                gyr_skew_mat <<     0, -gyr_dt[2],  gyr_dt[1],
                            gyr_dt[2],         0, -gyr_dt[0],
                            -gyr_dt[1],  gyr_dt[0],         0;
                const double s_gyr_norm = std::sin(gyr_norm);
                const double gyr_norm_sq = gyr_norm * gyr_norm;
                const double scalar_2 = (1 - std::cos(gyr_norm)) / gyr_norm_sq;
                const Mat3 skew_mat_sq = gyr_skew_mat * gyr_skew_mat;
                e_R = e_R + ( (s_gyr_norm / gyr_norm ) * gyr_skew_mat ) + ( scalar_2 * skew_mat_sq);
                j_r = j_r - (scalar_2 * gyr_skew_mat) + ( ( (gyr_norm - s_gyr_norm)/ (gyr_norm_sq * gyr_norm) ) * skew_mat_sq);
            }
            const Mat3 A = e_R.transpose();
            const Mat3 B = (j_r*dt);
            const Mat3 imu_cov = Mat3::Identity()*gyr_var_;
            cov_r = A*cov_r*A.transpose() + B*imu_cov*B.transpose();
            R = R * e_R;
            for(int i = 0; i < 3; ++i)
            {
                Vec3 offset = Vec3::Zero();
                offset(i) = kNumGyrBiasJacobianDelta;
                R_pert[i] = R_pert[i] * expMap((w + offset)*dt);
            }
        }

        static Mat3 gyrBiasJacobian(const Mat3& R, const Mat3* R_pert)
        {
            Mat3 out;
            for(int i = 0; i < 3; ++i) out.col(i) = logMap(R.transpose()*R_pert[i])/kNumGyrBiasJacobianDelta;
            return out;
        }

        // The accelerometer sample rotated into the start frame, and its bias Jacobians, as
        // reprojectAccData computes them
        static void rotateAcc(const Mat3& R, const Mat3& J_R_bw, const Vec3& acc, Vec3& a, Mat3& d_a_d_bf, Mat3& d_a_d_bw)
        {
            d_a_d_bf = R;
            const Mat9_3 temp_d_R_d_bw = jacobianExpMapZeroM(J_R_bw);
            for(int r = 0; r < 3; ++r)
            {
                Row9 temp;
                temp << R(r,0)*acc(0), R(r,1)*acc(0), R(r,2)*acc(0),
                        R(r,0)*acc(1), R(r,1)*acc(1), R(r,2)*acc(1),
                        R(r,0)*acc(2), R(r,1)*acc(2), R(r,2)*acc(2);
                d_a_d_bw.row(r) = temp*temp_d_R_d_bw;
            }
            a = R*acc;
        }

        // The rotated accelerometer value and Jacobians at t, on the line through two knots
        static void interpolate(const Knot& k0, const Knot& k1, const double t, Vec3& a, Mat3& d_a_d_bf, Mat3& d_a_d_bw)
        {
            const double ratio = (t - k0.t) / (k1.t - k0.t);
            for(int axis = 0; axis < 3; ++axis)
            {
                const double alpha = (k1.a(axis) - k0.a(axis)) / (k1.t - k0.t);
                a(axis) = k0.a(axis) + alpha*(t - k0.t);
            }
            d_a_d_bf = k1.d_a_d_bf*ratio + k0.d_a_d_bf*(1 - ratio);
            d_a_d_bw = k1.d_a_d_bw*ratio + k0.d_a_d_bw*(1 - ratio);
        }

        PreintMeas toMeas(const double t, const Mat3& R, const Mat3& cov_r, const Mat3* R_pert, const Vec3& v, const Vec3& p,
                const Mat3& d_v_d_bf, const Mat3& d_v_d_bw, const Mat3& d_p_d_bf, const Mat3& d_p_d_bw) const
        {
            PreintMeas out;
            out.delta_R = R;
            out.delta_v = v;
            out.delta_p = p;
            out.dt = t - start_t_;
            out.dt_sq_half = out.dt*out.dt*0.5;
            // The batch integrator's floor on the diagonal (1e-6) applies to the rotation block only:
            // the velocity and position variances are set after it
            out.cov = Mat9::Zero();
            out.cov.block<3,3>(0,0) = cov_r;
            for(int i = 0; i < 3; ++i) out.cov(i,i) = std::max(out.cov(i,i), 1e-6);
            for(int axis = 0; axis < 3; ++axis)
            {
                const double v_var = (t - start_t_)*acc_var_;
                out.cov(3+axis, 3+axis) = v_var;
                out.cov(6+axis, 6+axis) = (t - start_t_)*v_var;
            }
            out.d_delta_R_d_bw = gyrBiasJacobian(R, R_pert);
            out.d_delta_v_d_bw = d_v_d_bw;
            out.d_delta_v_d_bf = d_v_d_bf;
            out.d_delta_p_d_bw = d_p_d_bw;
            out.d_delta_p_d_bf = d_p_d_bf;
            out.d_delta_R_d_t = Vec3::Zero();
            out.d_delta_v_d_t = Vec3::Zero();
            out.d_delta_p_d_t = Vec3::Zero();
            return out;
        }

        // Integrates everything the data received allows: the start once the data around it is
        // there, then one accelerometer interval at a time, each once the gyroscope data covers it
        void advance()
        {
            if(gyr_.size() < 2 || acc_.size() < 2)
            {
                return;
            }
            if(knots_.empty() && !initialise())
            {
                return;
            }
            while(true)
            {
                const Knot& kc = knots_.back();
                const size_t j = kc.next_acc;
                if(j >= acc_.size() || gyr_.back().t < acc_[j].t)
                {
                    return;
                }
                knots_.push_back(step(kc, j));
            }
        }

        // The knot at the start time, from the accelerometer samples around it (the one before
        // rotated back from the start)
        bool initialise()
        {
            // First accelerometer sample at or after the start, and one before it
            const size_t k = std::lower_bound(acc_.begin(), acc_.end(), start_t_, [](const ImuSample& s, const double v) { return s.t < v; }) - acc_.begin();
            if(k == acc_.size())
            {
                return false;
            }
            if(k == 0 && acc_[0].t > start_t_)
            {
                throw std::range_error("IncrementalLpm: no accelerometer data before the start time");
            }
            if(gyr_.back().t < acc_[k].t)
            {
                return false;
            }
            Knot k0;
            k0.t = start_t_;
            k0.R = Mat3::Identity();
            k0.cov_r = Mat3::Zero();
            for(int i = 0; i < 3; ++i) k0.R_pert[i] = Mat3::Identity();
            k0.v = Vec3::Zero();
            k0.p = Vec3::Zero();
            k0.d_v_d_bf = Mat3::Zero();
            k0.d_v_d_bw = Mat3::Zero();
            k0.d_p_d_bf = Mat3::Zero();
            k0.d_p_d_bw = Mat3::Zero();
            // The rotated sample at the start time, and those around it
            Vec3 a1;
            Mat3 d1_bf, d1_bw;
            if(acc_[k].t == start_t_)
            {
                rotateAcc(k0.R, Mat3::Zero(), sampleVec(acc_[k]), k0.a, k0.d_a_d_bf, k0.d_a_d_bw);
                k0.next_acc = k + 1;
            }
            else
            {
                // The sample before: its rotation relative to the start, integrated forward to the
                // start and inverted, as the batch integrator re-expresses the times before the start
                const ImuSample& s0 = acc_[k-1];
                Mat3 R_fwd = Mat3::Identity();
                Mat3 cov_unused = Mat3::Zero();
                Mat3 R_pert_fwd[3] = {Mat3::Identity(), Mat3::Identity(), Mat3::Identity()};
                double s = s0.t;
                for(const double b : steps(s0.t, start_t_))
                {
                    rotStep(gyrAt(s), b - s, R_fwd, cov_unused, R_pert_fwd);
                    s = b;
                }
                const Mat3 R0 = R_fwd.transpose();
                Mat3 R0_pert[3];
                for(int i = 0; i < 3; ++i) R0_pert[i] = R_pert_fwd[i].transpose();
                Knot kb;
                kb.t = s0.t;
                rotateAcc(R0, gyrBiasJacobian(R0, R0_pert), sampleVec(s0), kb.a, kb.d_a_d_bf, kb.d_a_d_bw);
                // The sample after: rotated forward from the start
                Knot k1;
                k1.t = acc_[k].t;
                rotateTo(k0, k1);
                rotateAcc(k1.R, gyrBiasJacobian(k1.R, k1.R_pert), sampleVec(acc_[k]), k1.a, k1.d_a_d_bf, k1.d_a_d_bw);
                interpolate(kb, k1, start_t_, k0.a, k0.d_a_d_bf, k0.d_a_d_bw);
                k0.next_acc = k;
            }
            knots_.push_back(k0);
            return true;
        }

        // The rotation (and its covariance and shifted versions) of kn, at kn.t, integrated from kc
        void rotateTo(const Knot& kc, Knot& kn) const
        {
            kn.R = kc.R;
            kn.cov_r = kc.cov_r;
            for(int i = 0; i < 3; ++i) kn.R_pert[i] = kc.R_pert[i];
            double s = kc.t;
            for(const double b : steps(kc.t, kn.t))
            {
                rotStep(gyrAt(s), b - s, kn.R, kn.cov_r, kn.R_pert);
                s = b;
            }
        }

        // The knot at accelerometer sample j, integrated from the knot kc
        Knot step(const Knot& kc, const size_t j) const
        {
            Knot kn;
            kn.t = acc_[j].t;
            rotateTo(kc, kn);
            rotateAcc(kn.R, gyrBiasJacobian(kn.R, kn.R_pert), sampleVec(acc_[j]), kn.a, kn.d_a_d_bf, kn.d_a_d_bw);
            const double t_0 = kc.t, t_1 = kn.t;
            const double dt = t_1 - t_0;
            for(int axis = 0; axis < 3; ++axis)
            {
                const double d_0 = kc.a(axis), d_1 = kn.a(axis);
                kn.p(axis) = kc.p(axis) + kc.v(axis)*(t_1 - t_0) + ((t_0 - t_1)*(t_0 - t_1)*(2.0*d_0 + d_1)/6.0);
                kn.v(axis) = kc.v(axis) + ((t_1 - t_0)*(d_0 + d_1)/2.0);
                const Vec3 d0_bf = kc.d_a_d_bf.row(axis).transpose(), d1_bf = kn.d_a_d_bf.row(axis).transpose();
                const Vec3 d0_bw = kc.d_a_d_bw.row(axis).transpose(), d1_bw = kn.d_a_d_bw.row(axis).transpose();
                const Vec3 temp_d_v_d_bf = dt*(d0_bf + d1_bf)/2.0;
                const Vec3 temp_d_v_d_bw = dt*(d0_bw + d1_bw)/2.0;
                const Vec3 temp_d_p_d_bf = dt*dt*(2.0*d0_bf + d1_bf)/6.0;
                const Vec3 temp_d_p_d_bw = dt*dt*(2.0*d0_bw + d1_bw)/6.0;
                kn.d_p_d_bf.row(axis) = (kc.d_p_d_bf.row(axis).transpose() + dt*kc.d_v_d_bf.row(axis).transpose() + temp_d_p_d_bf).transpose();
                kn.d_p_d_bw.row(axis) = (kc.d_p_d_bw.row(axis).transpose() + dt*kc.d_v_d_bw.row(axis).transpose() + temp_d_p_d_bw).transpose();
                kn.d_v_d_bf.row(axis) = (kc.d_v_d_bf.row(axis).transpose() + temp_d_v_d_bf).transpose();
                kn.d_v_d_bw.row(axis) = (kc.d_v_d_bw.row(axis).transpose() + temp_d_v_d_bw).transpose();
            }
            kn.next_acc = j + 1;
            return kn;
        }
};

} // namespace ugpm

#endif

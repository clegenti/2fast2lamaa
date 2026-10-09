#include "lice/imu_window_estimator.h"

#include <ceres/ceres.h>
#include <ceres/manifold.h>
#include <ceres/rotation.h>
#include <ceres/sphere_manifold.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>

namespace
{
// A pose (in the log) as position and rotation vector
Eigen::Matrix<double, 6, 1> poseVec(const Mat4& T)
{
    Eigen::Matrix<double, 6, 1> out;
    out.head<3>() = T.block<3,1>(0,3);
    out.tail<3>() = ugpm::logMap(T.block<3,3>(0,0));
    return out;
}

Mat4 toMat4(const Eigen::Quaterniond& q, const Vec3& p)
{
    Mat4 T = Mat4::Identity();
    T.block<3,3>(0,0) = q.toRotationMatrix();
    T.block<3,1>(0,3) = p;
    return T;
}

// Orthonormal basis of the plane orthogonal to g (the tangent space of the gravity's sphere at g)
Eigen::Matrix<double, 3, 2> tangentBasis(const Vec3& g)
{
    const Vec3 n = g.normalized();
    const Vec3 a = (std::abs(n.x()) < 0.9) ? Vec3::UnitX() : Vec3::UnitY();
    const Vec3 b1 = a.cross(n).normalized();
    const Vec3 b2 = n.cross(b1);
    Eigen::Matrix<double, 3, 2> B;
    B.col(0) = b1;
    B.col(1) = b2;
    return B;
}

// Preintegrated IMU between nodes i and j: rotation, velocity and position residuals in node i's
// frame, the measurement corrected to first order for the biases' difference with the ones it was
// integrated with, whitened by its covariance (W: inverse of its Cholesky factor)
struct LpmResidual
{
    LpmResidual(const ugpm::PreintMeas& m, const Vec3& lin_ba, const Vec3& lin_bg)
        : dR(m.delta_R), dv(m.delta_v), dp(m.delta_p), J_R_bw(m.d_delta_R_d_bw), J_v_bw(m.d_delta_v_d_bw), J_v_bf(m.d_delta_v_d_bf),
          J_p_bw(m.d_delta_p_d_bw), J_p_bf(m.d_delta_p_d_bf), dt(m.dt), dt_sq_half(m.dt_sq_half), lin_ba(lin_ba), lin_bg(lin_bg)
    {
        const Eigen::LLT<Mat9> llt(m.cov);
        W = llt.matrixL().solve(Mat9::Identity());
    }

    template <typename T>
    bool operator()(const T* q_i, const T* p_i, const T* v_i, const T* ba_i, const T* bg_i,
            const T* q_j, const T* p_j, const T* v_j, const T* g, T* res) const
    {
        using M3 = Eigen::Matrix<T, 3, 3>;
        using V3 = Eigen::Matrix<T, 3, 1>;
        const Eigen::Map<const Eigen::Quaternion<T>> Qi(q_i), Qj(q_j);
        const Eigen::Map<const V3> Pi(p_i), Vi(v_i), BAi(ba_i), BGi(bg_i), Pj(p_j), Vj(v_j), G(g);
        // The preintegration's bias correction: the samples were integrated with lin_* subtracted
        const V3 d_bg = lin_bg.cast<T>() - BGi;
        const V3 d_ba = lin_ba.cast<T>() - BAi;
        const V3 phi = J_R_bw.cast<T>()*d_bg;
        M3 dR_corr;
        ceres::AngleAxisToRotationMatrix(phi.data(), dR_corr.data());
        const M3 dR_c = dR.cast<T>()*dR_corr;
        const V3 dv_c = dv.cast<T>() + J_v_bf.cast<T>()*d_ba + J_v_bw.cast<T>()*d_bg;
        const V3 dp_c = dp.cast<T>() + J_p_bf.cast<T>()*d_ba + J_p_bw.cast<T>()*d_bg;

        const M3 Ri = Qi.toRotationMatrix();
        const M3 E = dR_c.transpose()*Ri.transpose()*Qj.toRotationMatrix();
        V3 r_R;
        ceres::RotationMatrixToAngleAxis(E.data(), r_R.data());
        const T t_dt(dt), t_dt2(dt_sq_half);
        const V3 r_v = Ri.transpose()*(Vj - Vi - G*t_dt) - dv_c;
        const V3 r_p = Ri.transpose()*(Pj - Pi - Vi*t_dt - G*t_dt2) - dp_c;
        Eigen::Matrix<T, 9, 1> r;
        r << r_R, r_v, r_p;
        Eigen::Map<Eigen::Matrix<T, 9, 1>> out(res);
        out = W.cast<T>()*r;
        return true;
    }

    Mat3 dR;
    Vec3 dv, dp;
    Mat3 J_R_bw, J_v_bw, J_v_bf, J_p_bw, J_p_bf;
    double dt, dt_sq_half;
    Vec3 lin_ba, lin_bg;
    Mat9 W;
};

ceres::CostFunction* lpmCost(const ugpm::PreintMeas& m, const Vec3& lin_ba, const Vec3& lin_bg)
{
    return new ceres::AutoDiffCostFunction<LpmResidual, 9, 4, 3, 3, 3, 3, 4, 3, 3, 3>(new LpmResidual(m, lin_ba, lin_bg));
}

// Registered pose: position error and rotation error in the measured frame, whitened (W). The
// registration optimises a correction [dt, dr] applied on the right of its prior pose, so this is the
// same parameterisation as its covariance, to first order
struct PoseResidual
{
    PoseResidual(const Mat4& meas, const Mat6& W)
        : R_m(meas.block<3,3>(0,0)), p_m(meas.block<3,1>(0,3)), W(W)
    {}

    template <typename T>
    bool operator()(const T* q, const T* p, T* res) const
    {
        using M3 = Eigen::Matrix<T, 3, 3>;
        using V3 = Eigen::Matrix<T, 3, 1>;
        const Eigen::Map<const Eigen::Quaternion<T>> Q(q);
        const Eigen::Map<const V3> P(p);
        const M3 Rm = R_m.cast<T>();
        const V3 r_p = Rm.transpose()*(P - p_m.cast<T>());
        const M3 E = Rm.transpose()*Q.toRotationMatrix();
        V3 r_r;
        ceres::RotationMatrixToAngleAxis(E.data(), r_r.data());
        Eigen::Matrix<T, 6, 1> r;
        r << r_p, r_r;
        Eigen::Map<Eigen::Matrix<T, 6, 1>> out(res);
        out = W.cast<T>()*r;
        return true;
    }

    Mat3 R_m;
    Vec3 p_m;
    Mat6 W;
};

// Random walk of the biases between two nodes dt apart
struct BiasWalkResidual
{
    BiasWalkResidual(const double acc_walk_std, const double gyr_walk_std, const double dt)
        : w_a(1.0/(acc_walk_std*std::sqrt(dt))), w_g(1.0/(gyr_walk_std*std::sqrt(dt)))
    {}

    template <typename T>
    bool operator()(const T* ba_i, const T* bg_i, const T* ba_j, const T* bg_j, T* res) const
    {
        for(int i = 0; i < 3; ++i)
        {
            res[i] = (ba_j[i] - ba_i[i])*T(w_a);
            res[3+i] = (bg_j[i] - bg_i[i])*T(w_g);
        }
        return true;
    }

    double w_a, w_g;
};

ceres::CostFunction* biasWalkCost(const double acc_walk_std, const double gyr_walk_std, const double dt)
{
    return new ceres::AutoDiffCostFunction<BiasWalkResidual, 6, 3, 3, 3, 3>(new BiasWalkResidual(acc_walk_std, gyr_walk_std, dt));
}

// Zero-mean prior on the accelerometer bias
struct AccBiasPriorResidual
{
    explicit AccBiasPriorResidual(const double std) : w(1.0/std) {}

    template <typename T>
    bool operator()(const T* ba, T* res) const
    {
        for(int i = 0; i < 3; ++i) res[i] = ba[i]*T(w);
        return true;
    }

    double w;
};

ceres::CostFunction* accBiasPriorCost(const double std)
{
    return new ceres::AutoDiffCostFunction<AccBiasPriorResidual, 3, 3>(new AccBiasPriorResidual(std));
}

// Linear prior on (v, ba, bg, g): J*[v - v0; ba - ba0; bg - bg0; B^T(g - g0)] + r
class MarginalPriorCost : public ceres::CostFunction
{
    public:
        MarginalPriorCost(const Eigen::MatrixXd& J, const Eigen::VectorXd& r, const Vec3& v0, const Vec3& ba0, const Vec3& bg0,
                const Vec3& g0, const Eigen::Matrix<double, 3, 2>& B)
            : J_(J), r_(r), v0_(v0), ba0_(ba0), bg0_(bg0), g0_(g0), B_(B)
        {
            set_num_residuals((int)J.rows());
            mutable_parameter_block_sizes()->assign({3, 3, 3, 3});
        }

        bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override
        {
            Eigen::Matrix<double, 11, 1> dx;
            dx.segment<3>(0) = Eigen::Map<const Vec3>(parameters[0]) - v0_;
            dx.segment<3>(3) = Eigen::Map<const Vec3>(parameters[1]) - ba0_;
            dx.segment<3>(6) = Eigen::Map<const Vec3>(parameters[2]) - bg0_;
            dx.segment<2>(9) = B_.transpose()*(Eigen::Map<const Vec3>(parameters[3]) - g0_);
            Eigen::Map<Eigen::VectorXd>(residuals, J_.rows()) = J_*dx + r_;
            if(jacobians)
            {
                using RowMat = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;
                for(int b = 0; b < 3; ++b)
                {
                    if(jacobians[b]) Eigen::Map<RowMat>(jacobians[b], J_.rows(), 3) = J_.middleCols(3*b, 3);
                }
                if(jacobians[3]) Eigen::Map<RowMat>(jacobians[3], J_.rows(), 3) = J_.rightCols(2)*B_.transpose();
            }
            return true;
        }

    private:
        Eigen::MatrixXd J_;
        Eigen::VectorXd r_;
        Vec3 v0_, ba0_, bg0_, g0_;
        Eigen::Matrix<double, 3, 2> B_;
};

// Biases moved further than this from the ones an interval was integrated with: integrated again
constexpr double kRebiasAcc = 0.05;     // m/s^2
constexpr double kRebiasGyr = 0.005;    // rad/s
// IMU data kept before the oldest node (s)
constexpr double kImuMargin = 0.5;
// Poses waiting for IMU data beyond this are dropped (the IMU stream is missing)
constexpr size_t kMaxPending = 100;
} // namespace


ImuWindowEstimator::ImuWindowEstimator(const ImuWindowEstimatorOptions& options)
    : opt_(options)
{
    if(!opt_.log_path.empty())
    {
        log_.open(opt_.log_path);
        log_ << "t_ns,reg_x,reg_y,reg_z,reg_rx,reg_ry,reg_rz,pred_x,pred_y,pred_z,pred_rx,pred_ry,pred_rz,"
             << "opt_x,opt_y,opt_z,opt_rx,opt_ry,opt_rz,vx,vy,vz,bax,bay,baz,bgx,bgy,bgz,gx,gy,gz,nodes,iterations,solve_ms,rebiased,"
             << "std_x,std_y,std_z,std_rx,std_ry,std_rz\n";
        log_ << std::setprecision(10);
    }
    if(!opt_.trajectory_path.empty())
    {
        trajectory_.open(opt_.trajectory_path, std::ios::out | std::ios::trunc);
        trajectory_ << "timestamp, x, y, z, r0, r1, r2" << std::endl;
    }
    worker_ = std::thread(&ImuWindowEstimator::run, this);
}

ImuWindowEstimator::~ImuWindowEstimator()
{
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        stop_ = true;
    }
    queue_cv_.notify_all();
    worker_.join();
}

void ImuWindowEstimator::addGyr(const Vec3& gyr, const int64_t t_ns)
{
    {
        std::lock_guard<std::mutex> lock(imu_mutex_);
        if(t_ref_ns_ < 0) t_ref_ns_ = t_ns;
        ugpm::ImuSample s;
        s.t = toSec(t_ns);
        for(int i = 0; i < 3; ++i) s.data[i] = gyr[i];
        if(!gyr_buf_.empty() && s.t <= gyr_buf_.back().t) return;
        gyr_buf_.push_back(s);
        if(open_) open_->addGyr(s);
    }
    queue_cv_.notify_one();
}

void ImuWindowEstimator::addAcc(const Vec3& acc, const int64_t t_ns)
{
    {
        std::lock_guard<std::mutex> lock(imu_mutex_);
        if(t_ref_ns_ < 0) t_ref_ns_ = t_ns;
        ugpm::ImuSample s;
        s.t = toSec(t_ns);
        for(int i = 0; i < 3; ++i) s.data[i] = acc[i];
        if(!acc_buf_.empty() && s.t <= acc_buf_.back().t) return;
        acc_buf_.push_back(s);
        if(open_) open_->addAcc(s);
    }
    queue_cv_.notify_one();
}

void ImuWindowEstimator::poseWeight(const Mat6* registration_cov, Mat6& sqrt_info, Vec6& std)
{
    Mat6 cov = Mat6::Zero();
    cov.diagonal() << Vec3::Constant(opt_.pose_pos_std*opt_.pose_pos_std), Vec3::Constant(opt_.pose_rot_std*opt_.pose_rot_std);
    if(opt_.use_registration_covariance && registration_cov && registration_cov->allFinite())
    {
        // Each block scaled so that its smallest eigenvalue is the configured variance: D*C*D with D
        // diagonal keeps the correlations, and the matrix positive definite
        const Mat6 C = 0.5*(*registration_cov + registration_cov->transpose());
        const double l_t = Eigen::SelfAdjointEigenSolver<Mat3>(C.topLeftCorner<3,3>()).eigenvalues().minCoeff();
        const double l_r = Eigen::SelfAdjointEigenSolver<Mat3>(C.bottomRightCorner<3,3>()).eigenvalues().minCoeff();
        if(l_t > 0.0 && l_r > 0.0)
        {
            Vec6 d;
            d << Vec3::Constant(opt_.pose_pos_std/std::sqrt(l_t)), Vec3::Constant(opt_.pose_rot_std/std::sqrt(l_r));
            const Mat6 scaled = d.asDiagonal()*C*d.asDiagonal();
            const Eigen::LLT<Mat6> llt(scaled);
            if(llt.info() == Eigen::Success)
            {
                cov = scaled;
            }
            else
            {
                ++cov_fallbacks_;
            }
        }
        else
        {
            ++cov_fallbacks_;
        }
    }
    else if(opt_.use_registration_covariance)
    {
        ++cov_fallbacks_;
    }
    sqrt_info = Eigen::LLT<Mat6>(cov).matrixL().solve(Mat6::Identity());
    std = cov.diagonal().cwiseSqrt();
}

void ImuWindowEstimator::addPose(const Mat4& pose, const int64_t t_ns, const Mat6* registration_cov)
{
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        if(t_ns <= last_queued_ns_) return;     // already processed (or older)
        PendingPose p;
        p.t_ns = t_ns;
        p.pose = pose;
        poseWeight(registration_cov, p.sqrt_info, p.std);
        if(!pending_.empty() && t_ns <= pending_.back().t_ns)
        {
            // A scan registered again: the later pose replaces the earlier one (and a pose replaces
            // a trajectory time)
            if(t_ns == pending_.back().t_ns) pending_.back() = p;
            return;
        }
        pending_.push_back(p);
        if(pending_.size() > kMaxPending)
        {
            std::cout << "ImuWindowEstimator: no IMU data for the poses received, dropping the oldest" << std::endl;
            pending_.pop_front();
        }
    }
    queue_cv_.notify_one();
}

void ImuWindowEstimator::addTrajectoryTime(const int64_t t_ns)
{
    if(!trajectory_.is_open()) return;
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        if(!pending_.empty() && t_ns <= pending_.back().t_ns) return;
        if(t_ns <= last_queued_ns_) return;
        PendingPose p;
        p.t_ns = t_ns;
        p.is_pose = false;
        pending_.push_back(p);
        if(pending_.size() > kMaxPending) pending_.pop_front();
    }
    queue_cv_.notify_one();
}

void ImuWindowEstimator::writeTrajectory(const int64_t t_ns, const Mat4& pose)
{
    if(!trajectory_.is_open()) return;
    const Vec3 r = ugpm::logMap(pose.block<3,3>(0,0));
    trajectory_ << std::fixed << t_ns << ", " << pose(0,3) << ", " << pose(1,3) << ", " << pose(2,3) << ", "
                << r(0) << ", " << r(1) << ", " << r(2) << std::endl;
}

bool ImuWindowEstimator::latestState(State& state) const
{
    std::lock_guard<std::mutex> lock(state_mutex_);
    if(!has_latest_) return false;
    state = latest_;
    return true;
}

bool ImuWindowEstimator::predict(const int64_t t_ns, Mat4& pose, Vec3& vel) const
{
    std::lock_guard<std::mutex> lock(state_mutex_);
    if(!has_latest_) return false;
    std::lock_guard<std::mutex> imu_lock(imu_mutex_);
    return predictLocked(latest_, t_ns, pose, vel);
}

bool ImuWindowEstimator::biasEstimate(BiasEstimate& estimate) const
{
    std::lock_guard<std::mutex> lock(state_mutex_);
    if(!has_bias_estimate_) return false;
    estimate = bias_estimate_;
    return true;
}

ImuWindowEstimator::BiasEstimate ImuWindowEstimator::computeBiasEstimate()
{
    const Node& newest = nodes_.back();
    BiasEstimate e;
    e.t_ns = newest.t_ns;
    e.acc_bias = newest.ba;
    e.gyr_bias = newest.bg;

    // Covariance from the marginalised information on [v, ba, bg, dg] of the oldest node, bounded by a
    // very loose regularisation where it has none (the pseudo-inverse would say zero variance there)
    e.cov.setZero();
    e.cov.diagonal() << Vec3::Constant(opt_.acc_bias_prior_std*opt_.acc_bias_prior_std), Vec3::Constant(0.1*0.1);
    if(prior_.marginal && prior_.J.rows() > 0)
    {
        Eigen::Matrix<double, 11, 11> H = prior_.J.transpose()*prior_.J;
        Eigen::Matrix<double, 11, 1> reg;
        reg << Vec3::Constant(1.0/(100.0*100.0)), Vec3::Constant(1.0/(10.0*10.0)), Vec3::Constant(1.0/(1.0*1.0)), Eigen::Vector2d::Constant(1.0/(10.0*10.0));
        H.diagonal() += reg;
        const Eigen::Matrix<double, 11, 11> C = H.inverse();
        e.cov = C.block<6,6>(3,3);
        // The biases of the newest node wander from the oldest's over the window
        const double span = newest.t - nodes_.front().t;
        e.cov.diagonal() += (Vec6() << Vec3::Constant(opt_.acc_bias_walk_std*opt_.acc_bias_walk_std*span),
                                       Vec3::Constant(opt_.gyr_bias_walk_std*opt_.gyr_bias_walk_std*span)).finished();
    }

    // Stable: optimised long enough, and not moving any more
    if(first_solve_ns_ < 0) first_solve_ns_ = newest.t_ns;
    Vec6 b;
    b << newest.ba, newest.bg;
    bias_history_.push_back({newest.t_ns, b});
    while(!bias_history_.empty() && (newest.t_ns - bias_history_.front().first)*1e-9 > opt_.bias_stable_window) bias_history_.pop_front();
    double max_da = 0.0, max_dg = 0.0;
    for(const auto& [t, h] : bias_history_)
    {
        max_da = std::max(max_da, (h.head<3>() - b.head<3>()).norm());
        max_dg = std::max(max_dg, (h.tail<3>() - b.tail<3>()).norm());
    }
    e.stable = prior_.marginal && (newest.t_ns - first_solve_ns_)*1e-9 >= opt_.bias_stable_time
            && max_da <= opt_.bias_stable_acc && max_dg <= opt_.bias_stable_gyr;
    return e;
}

bool ImuWindowEstimator::predictLatest(Prediction& prediction) const
{
    std::lock_guard<std::mutex> lock(state_mutex_);
    if(!has_latest_) return false;
    std::lock_guard<std::mutex> imu_lock(imu_mutex_);
    if(!open_ || !open_->initialised() || gyr_buf_.empty()) return false;
    const int64_t t_ns = t_ref_ns_ + (int64_t)std::llround(open_->committedTime()*1e9);
    if(!predictLocked(latest_, t_ns, prediction.pose, prediction.vel)) return false;
    applyCorrection(t_ns, prediction.pose, prediction.vel);
    prediction.t_ns = t_ns;
    const ugpm::ImuSample& g = gyr_buf_.back();
    prediction.ang_vel = Vec3(g.data[0], g.data[1], g.data[2]) - latest_.gyr_bias;
    return true;
}

void ImuWindowEstimator::applyCorrection(const int64_t t_ns, Mat4& pose, Vec3& vel) const
{
    if(correction_.t0_ns < 0 || opt_.output_smoothing <= 0.0) return;
    const double alpha = 1.0 - (t_ns - correction_.t0_ns)*1e-9/opt_.output_smoothing;
    if(alpha <= 0.0) return;
    const double a = std::min(alpha, 1.0);
    pose.block<3,1>(0,3) += a*correction_.dp;
    pose.block<3,3>(0,0) = ugpm::expMap(a*correction_.dR)*pose.block<3,3>(0,0);
    vel += a*correction_.dv;
}

bool ImuWindowEstimator::predictLocked(const State& s, const int64_t t_ns, Mat4& pose, Vec3& vel) const
{
    if(!open_ || open_start_ns_ != s.t_ns || t_ns < s.t_ns || !open_->initialised())
    {
        return false;
    }
    const ugpm::PreintMeas m = open_->get(toSec(t_ns));
    const Vec3 d_ba = open_ba_ - s.acc_bias;
    const Vec3 d_bg = open_bg_ - s.gyr_bias;
    const Mat3 R = s.pose.block<3,3>(0,0);
    const Vec3 p = s.pose.block<3,1>(0,3);
    pose = Mat4::Identity();
    pose.block<3,3>(0,0) = R*m.delta_R*ugpm::expMap(m.d_delta_R_d_bw*d_bg);
    vel = s.vel + s.gravity*m.dt + R*(m.delta_v + m.d_delta_v_d_bf*d_ba + m.d_delta_v_d_bw*d_bg);
    pose.block<3,1>(0,3) = p + s.vel*m.dt + s.gravity*m.dt_sq_half + R*(m.delta_p + m.d_delta_p_d_bf*d_ba + m.d_delta_p_d_bw*d_bg);
    return true;
}

void ImuWindowEstimator::flush()
{
    std::unique_lock<std::mutex> lock(queue_mutex_);
    queue_cv_.wait(lock, [this]() { return stop_ || (pending_.empty() && !busy_); });
}

bool ImuWindowEstimator::imuCovers(const int64_t t_ns) const
{
    std::lock_guard<std::mutex> lock(imu_mutex_);
    if(t_ref_ns_ < 0) return false;
    const double t = toSec(t_ns);
    if(open_) return open_->initialised() && open_->committedTime() >= t;
    // First node: data of both streams before it, and an accelerometer sample after it that the
    // gyroscope data covers
    if(acc_buf_.empty() || gyr_buf_.empty() || acc_buf_.front().t >= t || gyr_buf_.front().t >= t) return false;
    const auto it = std::lower_bound(acc_buf_.begin(), acc_buf_.end(), t, [](const ugpm::ImuSample& s, const double v) { return s.t < v; });
    return it != acc_buf_.end() && gyr_buf_.back().t >= it->t;
}

void ImuWindowEstimator::run()
{
    while(true)
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        queue_cv_.wait(lock, [this]() {
            if(stop_) return true;
            while(!pending_.empty())
            {
                const int64_t t_ns = pending_.front().t_ns;
                if(imuCovers(t_ns)) return true;
                // The first pose needs IMU data before it: a pose before the IMU stream starts cannot be one
                bool drop = false;
                {
                    std::lock_guard<std::mutex> imu_lock(imu_mutex_);
                    drop = !open_ && t_ref_ns_ >= 0 && ((!acc_buf_.empty() && acc_buf_.front().t >= toSec(t_ns)) || (!gyr_buf_.empty() && gyr_buf_.front().t >= toSec(t_ns)));
                }
                if(!drop) return false;
                pending_.pop_front();
            }
            return false;
        });
        if(stop_) return;
        const auto item = pending_.front();
        pending_.pop_front();
        busy_ = true;
        last_queued_ns_ = std::max(last_queued_ns_, item.t_ns);
        lock.unlock();

        if(item.is_pose)
        {
            process(item);
        }
        else
        {
            // A trajectory time: the pose predicted from the newest optimised node
            Mat4 pose;
            Vec3 vel;
            bool ok = false;
            {
                std::lock_guard<std::mutex> state_lock(state_mutex_);
                if(has_latest_)
                {
                    std::lock_guard<std::mutex> imu_lock(imu_mutex_);
                    ok = predictLocked(latest_, item.t_ns, pose, vel);
                }
            }
            if(ok) writeTrajectory(item.t_ns, pose);
        }

        lock.lock();
        busy_ = false;
        lock.unlock();
        queue_cv_.notify_all();
    }
}

std::unique_ptr<ugpm::IncrementalLpm> ImuWindowEstimator::makeLpm(const double start, const Vec3& ba, const Vec3& bg) const
{
    ugpm::PreintPrior prior;
    prior.acc_bias = {ba[0], ba[1], ba[2]};
    prior.gyr_bias = {bg[0], bg[1], bg[2]};
    auto lpm = std::make_unique<ugpm::IncrementalLpm>(start, prior, opt_.acc_std*opt_.acc_std, opt_.gyr_std*opt_.gyr_std, opt_.min_freq);
    // From the sample before the start of each stream
    auto from = [start](const std::vector<ugpm::ImuSample>& buf) {
        const size_t k = std::lower_bound(buf.begin(), buf.end(), start, [](const ugpm::ImuSample& s, const double v) { return s.t < v; }) - buf.begin();
        return (k == 0) ? size_t(0) : k - 1;
    };
    for(size_t i = from(gyr_buf_); i < gyr_buf_.size(); ++i) lpm->addGyr(gyr_buf_[i]);
    for(size_t i = from(acc_buf_); i < acc_buf_.size(); ++i) lpm->addAcc(acc_buf_[i]);
    return lpm;
}

void ImuWindowEstimator::process(const PendingPose& pending)
{
    const int64_t t_ns = pending.t_ns;
    const Mat4& pose = pending.pose;
    Node n;
    n.t_ns = t_ns;
    n.meas = pose;
    n.meas_sqrt_info = pending.sqrt_info;
    n.q = Eigen::Quaterniond(Mat3(pose.block<3,3>(0,0))).normalized();
    n.p = pose.block<3,1>(0,3);
    n.ba = Vec3::Zero();
    n.bg = Vec3::Zero();
    n.v = Vec3::Zero();

    bool has_pred = false;
    Mat4 pred = Mat4::Identity();
    Vec3 pred_vel;
    double solve_ms = 0.0;

    if(nodes_.empty())
    {
        std::lock_guard<std::mutex> imu_lock(imu_mutex_);
        n.t = toSec(t_ns);
        // Gravity from the accelerometer mean over the last 0.2 s: at rest it measures -R^T g
        Vec3 mean = Vec3::Zero();
        int count = 0;
        for(const auto& s : acc_buf_)
        {
            if(s.t <= n.t && s.t >= n.t - 0.2)
            {
                mean += Vec3(s.data[0], s.data[1], s.data[2]);
                ++count;
            }
        }
        if(count == 0) mean = Vec3(acc_buf_.front().data[0], acc_buf_.front().data[1], acc_buf_.front().data[2]);
        gravity_ = -(n.q.toRotationMatrix()*mean).normalized()*opt_.gravity_norm;
        prior_ = Prior();
        nodes_.push_back(n);
    }
    else
    {
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            if(has_latest_)
            {
                std::lock_guard<std::mutex> imu_lock(imu_mutex_);
                has_pred = predictLocked(latest_, t_ns, pred, pred_vel);
            }
        }
        Interval iv;
        {
            std::lock_guard<std::mutex> imu_lock(imu_mutex_);
            n.t = toSec(t_ns);
            iv.meas = open_->get(n.t);
            iv.lin_ba = open_ba_;
            iv.lin_bg = open_bg_;
        }
        Node& last = nodes_.back();
        // Initial guess: the biases of the last node, its velocity propagated (for the second node,
        // before any solve, the velocity between the first two poses)
        n.ba = last.ba;
        n.bg = last.bg;
        if(nodes_.size() == 1)
        {
            last.v = (n.p - last.p)/iv.meas.dt;
            n.v = last.v;
        }
        else
        {
            const Vec3 d_ba = iv.lin_ba - last.ba, d_bg = iv.lin_bg - last.bg;
            n.v = last.v + gravity_*iv.meas.dt + last.q.toRotationMatrix()*(iv.meas.delta_v + iv.meas.d_delta_v_d_bf*d_ba + iv.meas.d_delta_v_d_bw*d_bg);
        }
        nodes_.push_back(n);
        intervals_.push_back(iv);

        if((int)nodes_.size() >= opt_.min_nodes)
        {
            const auto t0 = std::chrono::steady_clock::now();
            solve();
            rebias();
            solve_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            solved_once_ = true;
        }
    }

    // The integration from the new newest node, with its biases, and the state it starts from
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        std::lock_guard<std::mutex> imu_lock(imu_mutex_);
        // What predictLatest gives now, from the previous state, for the correction to start from
        bool has_old = false;
        int64_t t_now_ns = -1;
        Mat4 old_out;
        Vec3 old_vel;
        if(has_latest_ && opt_.output_smoothing > 0.0 && open_ && open_->initialised())
        {
            t_now_ns = t_ref_ns_ + (int64_t)std::llround(open_->committedTime()*1e9);
            has_old = predictLocked(latest_, t_now_ns, old_out, old_vel);
            if(has_old) applyCorrection(t_now_ns, old_out, old_vel);
        }
        const Node& newest = nodes_.back();
        open_ = makeLpm(newest.t, newest.ba, newest.bg);
        open_start_ns_ = newest.t_ns;
        open_ba_ = newest.ba;
        open_bg_ = newest.bg;
        if(solved_once_)
        {
            latest_.t_ns = newest.t_ns;
            latest_.pose = toMat4(newest.q, newest.p);
            latest_.vel = newest.v;
            latest_.acc_bias = newest.ba;
            latest_.gyr_bias = newest.bg;
            latest_.gravity = gravity_;
            bias_estimate_ = computeBiasEstimate();
            has_bias_estimate_ = true;
            has_latest_ = true;
            Mat4 new_out;
            Vec3 new_vel;
            if(has_old && predictLocked(latest_, t_now_ns, new_out, new_vel))
            {
                correction_.t0_ns = t_now_ns;
                correction_.dp = old_out.block<3,1>(0,3) - new_out.block<3,1>(0,3);
                correction_.dR = ugpm::logMap(old_out.block<3,3>(0,0)*new_out.block<3,3>(0,0).transpose());
                correction_.dv = old_vel - new_vel;
            }
        }
    }

    if(log_.is_open())
    {
        const Node& newest = nodes_.back();
        const auto reg = poseVec(pose);
        Eigen::Matrix<double, 6, 1> pr;
        pr.setConstant(std::numeric_limits<double>::quiet_NaN());
        if(has_pred) pr = poseVec(pred);
        const auto op = poseVec(toMat4(newest.q, newest.p));
        log_ << t_ns;
        for(int i = 0; i < 6; ++i) log_ << "," << reg(i);
        for(int i = 0; i < 6; ++i) log_ << "," << pr(i);
        for(int i = 0; i < 6; ++i) log_ << "," << op(i);
        const Vec3* state_vecs[4] = {&newest.v, &newest.ba, &newest.bg, &gravity_};
        for(const Vec3* v : state_vecs) for(int i = 0; i < 3; ++i) log_ << "," << (*v)(i);
        log_ << "," << nodes_.size() << "," << last_iterations_ << "," << solve_ms << "," << last_rebiased_;
        for(int i = 0; i < 6; ++i) log_ << "," << pending.std(i);
        log_ << "\n";
    }

    writeTrajectory(t_ns, solved_once_ ? toMat4(nodes_.back().q, nodes_.back().p) : pose);

    if(solved_once_)
    {
        slide();
    }
    // Drop the IMU data the window no longer needs
    {
        std::lock_guard<std::mutex> imu_lock(imu_mutex_);
        const double keep_from = nodes_.front().t - kImuMargin;
        auto trim = [keep_from](std::vector<ugpm::ImuSample>& buf) {
            const auto it = std::lower_bound(buf.begin(), buf.end(), keep_from, [](const ugpm::ImuSample& s, const double v) { return s.t < v; });
            buf.erase(buf.begin(), it);
        };
        trim(acc_buf_);
        trim(gyr_buf_);
    }
}

void ImuWindowEstimator::solve()
{
    ceres::Problem problem;
    for(size_t i = 0; i < nodes_.size(); ++i)
    {
        Node& n = nodes_[i];
        problem.AddParameterBlock(n.q.coeffs().data(), 4, new ceres::EigenQuaternionManifold());
        problem.AddParameterBlock(n.p.data(), 3);
        problem.AddParameterBlock(n.v.data(), 3);
        problem.AddParameterBlock(n.ba.data(), 3);
        problem.AddParameterBlock(n.bg.data(), 3);
        if(i == 0)
        {
            // The oldest pose is held (see slide())
            problem.SetParameterBlockConstant(n.q.coeffs().data());
            problem.SetParameterBlockConstant(n.p.data());
        }
        else
        {
            problem.AddResidualBlock(new ceres::AutoDiffCostFunction<PoseResidual, 6, 4, 3>(new PoseResidual(n.meas, n.meas_sqrt_info)),
                    new ceres::CauchyLoss(opt_.pose_loss_scale), n.q.coeffs().data(), n.p.data());
        }
    }
    problem.AddParameterBlock(gravity_.data(), 3, new ceres::SphereManifold<3>());
    for(size_t i = 0; i < intervals_.size(); ++i)
    {
        Node& a = nodes_[i];
        Node& b = nodes_[i+1];
        const Interval& iv = intervals_[i];
        problem.AddResidualBlock(lpmCost(iv.meas, iv.lin_ba, iv.lin_bg), nullptr,
                a.q.coeffs().data(), a.p.data(), a.v.data(), a.ba.data(), a.bg.data(), b.q.coeffs().data(), b.p.data(), b.v.data(), gravity_.data());
        problem.AddResidualBlock(biasWalkCost(opt_.acc_bias_walk_std, opt_.gyr_bias_walk_std, b.t - a.t), nullptr, a.ba.data(), a.bg.data(), b.ba.data(), b.bg.data());
    }
    Node& first = nodes_.front();
    if(!prior_.marginal)
    {
        problem.AddResidualBlock(accBiasPriorCost(opt_.acc_bias_prior_std), nullptr, first.ba.data());
    }
    else if(prior_.J.rows() > 0)
    {
        problem.AddResidualBlock(new MarginalPriorCost(prior_.J, prior_.r, prior_.v0, prior_.ba0, prior_.bg0, prior_.g0, prior_.B), nullptr,
                first.v.data(), first.ba.data(), first.bg.data(), gravity_.data());
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations = opt_.max_iterations;
    options.num_threads = 1;
    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    last_iterations_ = (int)summary.iterations.size();
}

void ImuWindowEstimator::rebias()
{
    last_rebiased_ = 0;
    for(size_t i = 0; i < intervals_.size(); ++i)
    {
        Interval& iv = intervals_[i];
        const Node& a = nodes_[i];
        if((a.ba - iv.lin_ba).norm() <= kRebiasAcc && (a.bg - iv.lin_bg).norm() <= kRebiasGyr) continue;
        std::lock_guard<std::mutex> imu_lock(imu_mutex_);
        const auto lpm = makeLpm(a.t, a.ba, a.bg);
        if(!lpm->initialised() || lpm->committedTime() < nodes_[i+1].t) continue;
        iv.meas = lpm->get(nodes_[i+1].t);
        iv.lin_ba = a.ba;
        iv.lin_bg = a.bg;
        ++last_rebiased_;
    }
    if(last_rebiased_ > 0)
    {
        solve();
    }
}

void ImuWindowEstimator::slide()
{
    while((int)nodes_.size() > opt_.min_nodes && (nodes_.back().t - nodes_.front().t) > opt_.window_duration)
    {
        Node& n0 = nodes_[0];
        Node& n1 = nodes_[1];
        const Interval& iv = intervals_[0];
        const Eigen::Matrix<double, 3, 2> B = tangentBasis(gravity_);

        // The factors leaving with n0, linearised: columns [v0 ba0 bg0 | v1 ba1 bg1 dg], the poses
        // of n0 and n1 held constant
        constexpr int kCols = 20;
        std::vector<Eigen::MatrixXd> Js;
        std::vector<Eigen::VectorXd> rs;
        if(!prior_.marginal)
        {
            Eigen::MatrixXd J = Eigen::MatrixXd::Zero(3, kCols);
            J.block<3,3>(0,3) = Mat3::Identity()/opt_.acc_bias_prior_std;
            Js.push_back(J);
            rs.push_back(n0.ba/opt_.acc_bias_prior_std);
        }
        else if(prior_.J.rows() > 0)
        {
            const int m = (int)prior_.J.rows();
            Eigen::Matrix<double, 11, 1> dx;
            dx << n0.v - prior_.v0, n0.ba - prior_.ba0, n0.bg - prior_.bg0, prior_.B.transpose()*(gravity_ - prior_.g0);
            Eigen::MatrixXd J = Eigen::MatrixXd::Zero(m, kCols);
            J.leftCols(9) = prior_.J.leftCols(9);
            J.rightCols(2) = prior_.J.rightCols(2)*(prior_.B.transpose()*B);
            Js.push_back(J);
            rs.push_back(prior_.J*dx + prior_.r);
        }
        {
            using RowMat3 = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;
            std::unique_ptr<ceres::CostFunction> cost(lpmCost(iv.meas, iv.lin_ba, iv.lin_bg));
            const double* params[9] = {n0.q.coeffs().data(), n0.p.data(), n0.v.data(), n0.ba.data(), n0.bg.data(), n1.q.coeffs().data(), n1.p.data(), n1.v.data(), gravity_.data()};
            RowMat3 jv0(9, 3), jba0(9, 3), jbg0(9, 3), jv1(9, 3), jg(9, 3);
            double* jac[9] = {nullptr, nullptr, jv0.data(), jba0.data(), jbg0.data(), nullptr, nullptr, jv1.data(), jg.data()};
            Eigen::VectorXd r(9);
            cost->Evaluate(params, r.data(), jac);
            Eigen::MatrixXd J = Eigen::MatrixXd::Zero(9, kCols);
            J.block(0, 0, 9, 3) = jv0;
            J.block(0, 3, 9, 3) = jba0;
            J.block(0, 6, 9, 3) = jbg0;
            J.block(0, 9, 9, 3) = jv1;
            J.block(0, 18, 9, 2) = jg*B;
            Js.push_back(J);
            rs.push_back(r);
        }
        {
            using RowMat3 = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;
            std::unique_ptr<ceres::CostFunction> cost(biasWalkCost(opt_.acc_bias_walk_std, opt_.gyr_bias_walk_std, n1.t - n0.t));
            const double* params[4] = {n0.ba.data(), n0.bg.data(), n1.ba.data(), n1.bg.data()};
            RowMat3 j0(6, 3), j1(6, 3), j2(6, 3), j3(6, 3);
            double* jac[4] = {j0.data(), j1.data(), j2.data(), j3.data()};
            Eigen::VectorXd r(6);
            cost->Evaluate(params, r.data(), jac);
            Eigen::MatrixXd J = Eigen::MatrixXd::Zero(6, kCols);
            J.block(0, 3, 6, 3) = j0;
            J.block(0, 6, 6, 3) = j1;
            J.block(0, 12, 6, 3) = j2;
            J.block(0, 15, 6, 3) = j3;
            Js.push_back(J);
            rs.push_back(r);
        }
        int rows = 0;
        for(const auto& J : Js) rows += (int)J.rows();
        Eigen::MatrixXd J(rows, kCols);
        Eigen::VectorXd r(rows);
        int row = 0;
        for(size_t i = 0; i < Js.size(); ++i)
        {
            J.middleRows(row, Js[i].rows()) = Js[i];
            r.segment(row, rs[i].size()) = rs[i];
            row += (int)Js[i].rows();
        }

        // Schur complement on n0's velocity and biases (pseudo-inverse: they may be unobserved)
        const Eigen::MatrixXd H = J.transpose()*J;
        const Eigen::VectorXd b = J.transpose()*r;
        const Eigen::MatrixXd Hmm = H.topLeftCorner(9, 9);
        const Eigen::MatrixXd Hmk = H.topRightCorner(9, 11);
        const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es_m(Hmm);
        const double tol_m = 1e-10*std::max(1.0, es_m.eigenvalues().maxCoeff());
        Eigen::VectorXd inv_m = es_m.eigenvalues();
        for(int i = 0; i < inv_m.size(); ++i) inv_m(i) = (inv_m(i) > tol_m) ? 1.0/inv_m(i) : 0.0;
        const Eigen::MatrixXd Hmm_inv = es_m.eigenvectors()*inv_m.asDiagonal()*es_m.eigenvectors().transpose();
        Eigen::MatrixXd Hk = H.bottomRightCorner(11, 11) - Hmk.transpose()*Hmm_inv*Hmk;
        Eigen::VectorXd bk = b.tail(11) - Hmk.transpose()*Hmm_inv*b.head(9);

        // Random walk of the gravity direction over the interval: the prior on g becomes one on
        // g' = g + w, w ~ N(0, q I), with g marginalised out
        const double q = std::pow(opt_.gravity_walk_std*opt_.gravity_norm, 2)*(n1.t - n0.t);
        if(q > 0.0)
        {
            const Eigen::Matrix2d Qi = Eigen::Matrix2d::Identity()/q;
            const Eigen::Matrix2d S_inv = (Hk.bottomRightCorner<2,2>() + Qi).inverse();
            const Eigen::MatrixXd Hyg = Hk.topRightCorner(9, 2);
            Eigen::MatrixXd Hn(11, 11);
            Hn.topLeftCorner(9, 9) = Hk.topLeftCorner(9, 9) - Hyg*S_inv*Hyg.transpose();
            Hn.topRightCorner(9, 2) = Hyg*S_inv*Qi;
            Hn.bottomLeftCorner(2, 9) = Hn.topRightCorner(9, 2).transpose();
            Hn.bottomRightCorner<2,2>() = Qi - Qi*S_inv*Qi;
            Eigen::VectorXd bn(11);
            bn.head(9) = bk.head(9) - Hyg*S_inv*bk.tail(2);
            bn.tail(2) = Qi*S_inv*bk.tail(2);
            Hk = Hn;
            bk = bn;
        }

        // Back to a residual: J_p^T J_p = Hk, J_p^T r_p = bk, on the directions with information
        const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es_k(0.5*(Hk + Hk.transpose()));
        const double tol_k = 1e-10*std::max(1.0, es_k.eigenvalues().maxCoeff());
        std::vector<int> keep;
        for(int i = 0; i < 11; ++i) if(es_k.eigenvalues()(i) > tol_k) keep.push_back(i);
        Prior prior;
        prior.marginal = true;
        prior.J.resize(keep.size(), 11);
        prior.r.resize(keep.size());
        for(size_t k = 0; k < keep.size(); ++k)
        {
            const double s = std::sqrt(es_k.eigenvalues()(keep[k]));
            const Eigen::VectorXd u = es_k.eigenvectors().col(keep[k]);
            prior.J.row(k) = s*u.transpose();
            prior.r(k) = u.dot(bk)/s;
        }
        prior.v0 = n1.v;
        prior.ba0 = n1.ba;
        prior.bg0 = n1.bg;
        prior.g0 = gravity_;
        prior.B = B;
        prior_ = prior;

        nodes_.pop_front();
        intervals_.pop_front();
    }
}

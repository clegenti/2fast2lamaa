#pragma once

// Sliding-window estimator of the IMU state (pose, velocity, biases, gravity) from the poses of the GP
// map registration and the IMU samples between them.
//
// One node per registered pose (the IMU/body frame in the map frame, at the scan time): rotation,
// position, velocity (map frame) and the two biases (the usual convention: measurement = true value +
// bias). The gravity vector (map frame, fixed norm) is estimated with the window. Factors:
//   - the registered pose of each node, robust loss, weighted either by the configured standard
//     deviations or, with `use_registration_covariance`, by the covariance of the registration re-scaled
//     so that the smallest eigenvalue of its translation block is pose_pos_std^2 and that of its
//     rotation block pose_rot_std^2 (the correlations and the ratios between directions are kept:
//     directions the registration constrains poorly stay loose);
//   - between consecutive nodes, the IMU preintegrated between them (incremental LPM, integrated as
//     the samples arrive), with the first-order bias correction of its Jacobians, weighted by its
//     covariance; re-integrated when a bias moves too far from the one it was integrated with;
//   - between consecutive nodes, a random walk of the biases;
//   - on the first node, a zero-mean prior on the accelerometer bias (observability before the
//     motion excites it).
// The window covers `window_duration` seconds (at least `min_nodes` nodes). When a node leaves it, the
// next node's pose is held constant, and the factors that leave with the node are replaced by a
// Gaussian prior on the next node's velocity and biases and on the gravity: the Schur complement of
// those factors (linearised at the current estimate, the two poses constant), on the removed node's
// velocity and biases. The prior is in information form, so directions that the removed factors do
// not constrain stay unconstrained, and the gravity part is loosened by a random walk
// (`gravity_walk_std`) so that the estimate can follow a slow drift of the map frame.
//
// Threads: the IMU samples and the poses may be given from any thread. The window is optimised by
// the estimator's own thread, once the IMU data covers the pose's time (a pose waits for the IMU data,
// through a dropout too). latestState() and predict() may be called from any thread.

#include "lice/types.h"
#include "preint/incremental_lpm.h"

#include <condition_variable>
#include <deque>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <Eigen/Geometry>

struct ImuWindowEstimatorOptions
{
    double window_duration = 1.0;       // s
    int min_nodes = 3;
    double acc_std = 0.02;              // IMU noise, as for the odometry's preintegration
    double gyr_std = 0.005;
    double acc_bias_walk_std = 1e-3;    // m/s^2/sqrt(s)
    double gyr_bias_walk_std = 1e-4;    // rad/s/sqrt(s)
    double acc_bias_prior_std = 0.2;    // m/s^2, zero-mean prior on the first node
    double gravity_norm = 9.81;         // m/s^2
    double gravity_walk_std = 1e-3;     // rad/sqrt(s), random walk of the gravity direction
    double pose_pos_std = 0.02;         // m, registered poses (see use_registration_covariance)
    double pose_rot_std = 0.002;        // rad
    bool use_registration_covariance = false;
    double pose_loss_scale = 3.0;       // Cauchy scale on the whitened pose residual (standard deviations)
    int max_iterations = 10;
    double min_freq = 500.0;            // step rate of the rotation integration (Hz)
    // predictLatest() blends each correction of a new optimisation in over this time (s) instead of
    // jumping to it: the offset between what it gave and the new prediction decays linearly to zero.
    // 0: no smoothing
    double output_smoothing = 0.1;
    std::string log_path = "";          // csv of every processed pose, when not empty
    // Trajectory file (the format of the GP map's trajectory.csv), when not empty: one line per pose
    // given (the node's pose right after its optimisation, the registered pose before the first
    // one) and per trajectory time given (the pose predicted there from the newest optimised node)
    std::string trajectory_path = "";
    // The bias estimate is flagged stable (biasEstimate) once the window has been optimised for
    // bias_stable_time seconds and the biases have moved less than bias_stable_acc / bias_stable_gyr
    // over the last bias_stable_window seconds
    double bias_stable_time = 10.0;     // s
    double bias_stable_window = 3.0;    // s
    double bias_stable_acc = 0.02;      // m/s^2
    double bias_stable_gyr = 0.002;     // rad/s
};

class ImuWindowEstimator
{
    public:
        struct State
        {
            int64_t t_ns = -1;
            Mat4 pose = Mat4::Identity();
            Vec3 vel = Vec3::Zero();
            Vec3 acc_bias = Vec3::Zero();
            Vec3 gyr_bias = Vec3::Zero();
            Vec3 gravity = Vec3::Zero();
        };

        struct Prediction
        {
            int64_t t_ns = -1;
            Mat4 pose = Mat4::Identity();
            Vec3 vel = Vec3::Zero();        // map frame
            Vec3 ang_vel = Vec3::Zero();    // body frame, bias removed
        };

        explicit ImuWindowEstimator(const ImuWindowEstimatorOptions& options);
        ~ImuWindowEstimator();

        ImuWindowEstimator(const ImuWindowEstimator&) = delete;
        ImuWindowEstimator& operator=(const ImuWindowEstimator&) = delete;

        // IMU samples, in the IMU/body frame of the poses. A sample not newer than the last one of
        // its stream is dropped
        void addGyr(const Vec3& gyr, const int64_t t_ns);
        void addAcc(const Vec3& acc, const int64_t t_ns);

        // A registered pose (IMU/body frame in the map frame) at t_ns, in increasing time order: queued,
        // and processed once the IMU data covers t_ns. `registration_cov`: covariance of the
        // registration's correction [translation, rotation] (see use_registration_covariance), null or
        // not finite to use the configured standard deviations
        void addPose(const Mat4& pose, const int64_t t_ns, const Mat6* registration_cov = nullptr);

        // A time at which to write the trajectory (trajectory_path), for the scans that are not given
        // as poses (ignored at the time of a pose). Processed in order with the poses
        void addTrajectoryTime(const int64_t t_ns);

        // The newest node of the last optimisation. False before the first one
        bool latestState(State& state) const;

        // The pose and velocity at t_ns (at or after the newest node's time), integrated from the newest
        // node of the last optimisation with the IMU data received (extrapolated past it)
        bool predict(const int64_t t_ns, Mat4& pose, Vec3& vel) const;

        // The same at the newest time both IMU streams cover, with the angular velocity there, and
        // smoothed (output_smoothing). For publishing at the IMU rate: its t_ns increases as the
        // samples arrive, and it is continuous across the optimisations
        bool predictLatest(Prediction& prediction) const;

        struct BiasEstimate
        {
            int64_t t_ns = -1;              // time of the node it is the estimate of
            Vec3 acc_bias = Vec3::Zero();   // measurement = true value + bias
            Vec3 gyr_bias = Vec3::Zero();
            // Covariance of [acc_bias, gyr_bias]: from the information the window has marginalised
            // (data older than the window, so barely any of the samples a consumer of the estimate
            // integrates now), plus the random walk of the biases over the window
            Mat6 cov = Mat6::Identity();
            bool stable = false;            // see bias_stable_*
        };
        // The biases of the newest node of the last optimisation. False before the first one
        bool biasEstimate(BiasEstimate& estimate) const;

        // Waits until every pose given so far has been processed (or dropped), for tests
        void flush();

    private:
        struct Node
        {
            int64_t t_ns;
            double t;
            Eigen::Quaterniond q;
            Vec3 p, v, ba, bg;
            Mat4 meas;          // registered pose
            Mat6 meas_sqrt_info;    // whitening of its residual [translation, rotation]
        };
        // The IMU preintegrated between two consecutive nodes, and the biases it was integrated with
        struct Interval
        {
            ugpm::PreintMeas meas;
            Vec3 lin_ba, lin_bg;
        };
        // Gaussian prior on the oldest node's velocity and biases and on the gravity, in information
        // form: residual J*[v - v0; ba - ba0; bg - bg0; B^T(g - g0)] + r (B: tangent basis at g0)
        struct Prior
        {
            bool marginal = false;      // false: the zero-mean accelerometer bias prior only
            Eigen::MatrixXd J;
            Eigen::VectorXd r;
            Vec3 v0, ba0, bg0, g0;
            Eigen::Matrix<double, 3, 2> B;
        };

        ImuWindowEstimatorOptions opt_;

        // IMU data (imu_mutex_): the samples since shortly before the oldest node, in seconds since
        // the first sample, and the integration from the newest node onwards
        mutable std::mutex imu_mutex_;
        int64_t t_ref_ns_ = -1;
        std::vector<ugpm::ImuSample> acc_buf_, gyr_buf_;
        std::unique_ptr<ugpm::IncrementalLpm> open_;
        int64_t open_start_ns_ = -1;
        Vec3 open_ba_ = Vec3::Zero(), open_bg_ = Vec3::Zero();

        // Poses waiting for the IMU data (queue_mutex_)
        std::mutex queue_mutex_;
        std::condition_variable queue_cv_;
        struct PendingPose
        {
            int64_t t_ns;
            Mat4 pose;
            Mat6 sqrt_info;
            Vec6 std;       // standard deviations of the weighting, for the log
            bool is_pose = true;    // false: a trajectory time only
        };
        std::deque<PendingPose> pending_;
        size_t cov_fallbacks_ = 0;
        bool busy_ = false;
        int64_t last_queued_ns_ = -1;   // time of the last item taken off the queue
        bool stop_ = false;
        std::thread worker_;

        // The window (only the worker thread touches it)
        std::deque<Node> nodes_;
        std::deque<Interval> intervals_;    // intervals_[i] between nodes_[i] and nodes_[i+1]
        Vec3 gravity_ = Vec3::Zero();
        Prior prior_;
        bool solved_once_ = false;
        int last_iterations_ = 0;       // of the last solve, for the log
        int last_rebiased_ = 0;         // intervals integrated again after the last solve

        // Snapshot of the newest optimised node (state_mutex_)
        mutable std::mutex state_mutex_;
        State latest_;
        bool has_latest_ = false;
        // The correction being blended in by predictLatest (state_mutex_): at t0_ns, the smoothed
        // output was the raw prediction + dp, Exp(dR)*R, + dv
        struct Correction
        {
            int64_t t0_ns = -1;
            Vec3 dp = Vec3::Zero();
            Vec3 dR = Vec3::Zero();
            Vec3 dv = Vec3::Zero();
        };
        Correction correction_;
        BiasEstimate bias_estimate_;    // state_mutex_
        bool has_bias_estimate_ = false;
        // For the stability of the biases (worker thread only)
        int64_t first_solve_ns_ = -1;
        std::deque<std::pair<int64_t, Vec6>> bias_history_;
        // The bias estimate of the newest node, after an optimisation
        BiasEstimate computeBiasEstimate();
        // The raw prediction with the part of the correction still to blend in at t_ns
        void applyCorrection(const int64_t t_ns, Mat4& pose, Vec3& vel) const;

        std::ofstream log_;
        std::ofstream trajectory_;
        void writeTrajectory(const int64_t t_ns, const Mat4& pose);

        double toSec(const int64_t t_ns) const { return (t_ns - t_ref_ns_)*1e-9; }

        void run();
        bool imuCovers(const int64_t t_ns) const;
        void process(const PendingPose& pending);
        // The whitening of a registered pose's residual, and its standard deviations
        void poseWeight(const Mat6* registration_cov, Mat6& sqrt_info, Vec6& std);
        void solve();
        void slide();
        void rebias();
        // An integration from `start` with the given biases, over the buffered samples (imu_mutex_ held)
        std::unique_ptr<ugpm::IncrementalLpm> makeLpm(const double start, const Vec3& ba, const Vec3& bg) const;
        bool predictLocked(const State& s, const int64_t t_ns, Mat4& pose, Vec3& vel) const;
};

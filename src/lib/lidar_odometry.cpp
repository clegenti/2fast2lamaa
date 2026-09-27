#include "lice/lidar_odometry.h"
#include <cstdlib>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <array>
#include <memory>
#include "nanoflann.hpp"
#include "ankerl/unordered_dense.h"
#include <ctime>


// ---- Conditioning of the target triplets/pairs a data association is built on -------------------
//
// The neighbours of a feature are overwhelmingly its own scanline: along a ring the points are
// centimetres apart, while the next ring is half a metre away at range. Taking the closest ones
// therefore gives three (or two) collinear points, and a plane or a line fitted through those says
// nothing about the motion across them. The residual is not merely noisy, it is blind: for a triplet
// on one scanline `v1 x v2` is a vanishing, arbitrarily oriented vector, so the point-to-plane
// distance stays near zero whatever the pose does.

// Two targets count as coming from different scanlines when their ring differs. Sensors that report
// no ring fall back on the sampling time: more than this apart is a different part of the sweep.
const int64_t kMinTargetTimeSpread = 5000000; // 5 ms in nanoseconds

// Smallest |v1 x v2|/(|v1||v2|) accepted for a planar triplet, which is the sine of the angle at
// their common vertex and exactly the quantity that normalises the normal in
// DataAssociation::computeResidual. 0.2 keeps the triplet at least ~11.5 degrees off collinear.
const double kMinPlanarSine = 0.2;

// How many neighbours to pull out of the kd tree as candidates. Enough of them are needed for a
// second scanline to be within reach at all: on a 16 beam sensor the six closest neighbours of a
// point are all its own ring.
const int kPlanarCandidates = 16;
const int kEdgeCandidates = 12;

// Leaf size of the kd tree of the target features: 32 was the fastest to build and search against
// 10 and 16, on target clouds of 2.6k to 118k points
const size_t kFeatureTreeLeafSize = 32;

namespace
{

// Target features of one type, contiguous as nanoflann reads them, with their indexes in the
// feature cloud. The coordinates go through float, as the jk tree used before got them (vec3f), so
// the neighbours found are the same
struct FeatureCloud
{
    std::vector<std::array<double, 3> > pts;
    std::vector<int> ids;
    size_t kdtree_get_point_count() const { return pts.size(); }
    double kdtree_get_pt(const size_t idx, const size_t dim) const { return pts[idx][dim]; }
    template<class BBox> bool kdtree_get_bbox(BBox&) const { return false; }
};

typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double, FeatureCloud>, FeatureCloud, 3, uint32_t> FeatureIndex;

// The index keeps a reference to its cloud, so the two live together (and must not move once built)
struct FeatureTree
{
    FeatureCloud cloud;
    std::unique_ptr<FeatureIndex> index;
};

} // namespace




LidarOdometry::LidarOdometry(const LidarOdometryParams& params, LidarOdometryPublisher* node)
    : params_(params)
    , node_(node)
{
    params_.association_filter_lin_quantum = 1.5*params_.feature_voxel_size;

    // Initialize the state variables and current position and rotation
    state_blocks_.resize(4, Vec3::Zero());
    current_pos_ = Vec3::Zero();
    current_rot_ = Vec3::Zero();


    // Initialize the state calibration vector
    Vec3 temp;
    temp << params_.calib_rx, params_.calib_ry, params_.calib_rz;
    ceres::AngleAxisToQuaternion<double>(temp.data(), state_calib_.data());
    state_calib_[4] = params_.calib_px;
    state_calib_[5] = params_.calib_py;
    state_calib_[6] = params_.calib_pz;

    // Initialize the loss function
    loss_function_ = new ceres::CauchyLoss(params_.loss_function_scale);

    // Initialize the sensor noise
    imu_data_.acc_var = params_.acc_std*params_.acc_std;
    imu_data_.gyr_var = params_.gyr_std*params_.gyr_std;
    lidar_weight_ = 1.0/params_.lidar_std;
}


void LidarOdometry::addPc(const std::shared_ptr<std::vector<Pointd>>& pc, const int64_t t)
{
    bool has_imu_data = true;
    imu_mutex_.lock();
    if(params_.mode == LidarOdometryMode::IMU)
    {
        if((imu_data_.acc.size() == 0) || (imu_data_.gyr.size() == 0) || (imu_data_.acc[0].t > nanosToImuTime(t)) || (imu_data_.gyr[0].t > nanosToImuTime(t)))
        {
            has_imu_data = false;
        }
    }
    else if(params_.mode == LidarOdometryMode::GYR)
    {
        if((imu_data_.gyr.size() == 0) || (imu_data_.gyr[0].t > nanosToImuTime(t)))
        {
            has_imu_data = false;
        }
    }
    else
    {
        first_ = false; // In NO_IMU mode, we don't care about IMU data
    }
    imu_mutex_.unlock();

    mutex_.lock();
    // Keep track of a time offset to deal with ugpm::ImuData coded with doubles
    if(first_ || !has_imu_data)
    {
        mutex_.unlock();
        return;
    }
    if(last_pc_time_ >= 0)
    {
        scan_time_sum_ += (t - last_pc_time_);
        scan_count_++;
    }
    last_pc_time_ = t;
    mutex_.unlock();

    splitAndFeatureExtraction(pc, t);
}

void LidarOdometry::addAccSample(const Vec3& acc, const int64_t t)
{
    // Keep track of a time offset to deal with ugpm::ImuData coded with doubles
    mutex_.lock();
    if(first_)
    {
        first_ = false;
        imu_time_offset_ = t;
    }
    mutex_.unlock();
    ugpm::ImuSample imu_sample;
    imu_sample.data[0] = acc[0];
    imu_sample.data[1] = acc[1];
    imu_sample.data[2] = acc[2];
    imu_sample.t = nanosToImuTime(t);

    imu_mutex_.lock();
    imu_data_.acc.push_back(imu_sample);
    imu_mutex_.unlock();
    notifyNewData(true);
}

void LidarOdometry::addGyroSample(const Vec3& gyro, const int64_t t)
{
    mutex_.lock();
    // Keep track of a time offset to deal with ugpm::ImuData coded with doubles
    if(first_)
    {
        first_ = false;
        imu_time_offset_ = t;
    }
    mutex_.unlock();
    ugpm::ImuSample imu_sample;
    imu_sample.data[0] = gyro[0];
    imu_sample.data[1] = gyro[1];
    imu_sample.data[2] = gyro[2];
    imu_sample.t = nanosToImuTime(t);

    imu_mutex_.lock();
    imu_data_.gyr.push_back(imu_sample);
    imu_mutex_.unlock();
    notifyNewData(true);
}

void LidarOdometry::stop()
{
    mutex_.lock();
    running_ = false;
    mutex_.unlock();
    notifyNewData(false);
}

void LidarOdometry::notifyNewData(const bool imu_sample)
{
    bool notify = true;
    {
        std::lock_guard<std::mutex> lock(data_cv_mutex_);
        data_version_++;
        if(imu_sample)
        {
            notify = wake_on_imu_;
        }
    }
    if(notify)
    {
        data_cv_.notify_one();
    }
}

std::shared_ptr<std::thread> LidarOdometry::runThread()
{
    return std::make_shared<std::thread>(&LidarOdometry::run, this);
}



void LidarOdometry::run()
{
    std::cout << "Starting lidar odometry thread" << std::endl;
    running_ = true;
    while(running_)
    {
        // Read before checking the data: whatever changes after this, even before the wait below,
        // makes the wait return at once
        uint64_t seen_version;
        {
            std::lock_guard<std::mutex> lock(data_cv_mutex_);
            seen_version = data_version_;
        }

        // Check if there is enough point clouds chunks and enough IMU data to run the optimisation
        mutex_.lock();
        int64_t last_time_t = imu_time_offset_;
        int64_t scan_period = (scan_count_ > 0) ? (int64_t)(scan_time_sum_ / scan_count_) : 100000000; 
        mutex_.unlock();
        imu_mutex_.lock();
        if( (params_.mode == LidarOdometryMode::IMU) && (imu_data_.acc.size() > 0) && (imu_data_.gyr.size() > 0))
        {
             last_time_t += std::min((int64_t)(imu_data_.acc.back().t * 1e9), (int64_t)(imu_data_.gyr.back().t * 1e9));
        }
        else if( (params_.mode == LidarOdometryMode::GYR) && (imu_data_.gyr.size() > 0))
        {
             last_time_t += (int64_t)(imu_data_.gyr.back().t * 1e9);
        }
        imu_mutex_.unlock();
        size_t id_to_run = 1;
        pc_mutex_.lock();
        bool has_data = false;
        if(params_.mode == LidarOdometryMode::IMU)
        {
            has_data = (imu_data_.acc.size() > 0) && (imu_data_.gyr.size() > 0) && (pc_chunk_features_.size() > id_to_run) && (pc_chunks_t_[id_to_run] + scan_period) <= last_time_t;
        }
        else if(params_.mode == LidarOdometryMode::GYR)
        {
            has_data = (imu_data_.gyr.size() > 0) && (pc_chunk_features_.size() > id_to_run) && (pc_chunks_t_[id_to_run] + scan_period) <= last_time_t;
        }
        else // NO_IMU
        {
            has_data = (pc_chunk_features_.size() > id_to_run);
        }
        // A chunk is there, only the IMU data to cover it is missing
        const bool waiting_for_imu = !has_data && (pc_chunk_features_.size() > id_to_run);
        pc_mutex_.unlock();
        if(has_data)
        {
            StopWatch sw;
            sw.start();
            optimise();
            sw.stop();
            sw.print("Lidar odometry overall optimisation time");
        }
        else
        {
            std::unique_lock<std::mutex> lock(data_cv_mutex_);
            wake_on_imu_ = waiting_for_imu;
            data_cv_.wait(lock, [&]{ return data_version_ != seen_version; });
            wake_on_imu_ = false;
        }
    }
    std::cout << "Stopping lidar odometry thread" << std::endl;
}


double LidarOdometry::nanosToImuTime(const int64_t nanos) const
{
    return nanosToSeconds(nanos, imu_time_offset_);
}


void LidarOdometry::splitAndFeatureExtraction(std::shared_ptr<std::vector<Pointd> > pc, const int64_t t)
{
    // The extraction itself is shared with the other nodes that publish undistorted scans (see
    // lice/lidar_feature_extraction.h). `median_dt_` is a member so that it is only estimated once.
    std::shared_ptr<std::vector<Pointd> > output(new std::vector<Pointd>());
    std::shared_ptr<std::vector<Pointd> > sparse_features(new std::vector<Pointd>());
    if(!extractLidarFeatures(*pc, toFeatureExtractionParams(params_, is_2d_), median_dt_, *output, sparse_features.get()))
    {
        return;
    }


    pc_mutex_.lock();
    if( pc_chunks_t_.size() > 0 
        && (t <= pc_chunks_t_.back()))
    {
        throw std::runtime_error("LidarOdometry::extractEdgeFeatures: New point cloud chunk has a timestamp older than the last one.");
    }
    pc_chunk_features_sparse_.push_back(sparse_features);
    pc_chunk_features_.push_back(output);
    pc_chunks_.push_back(pc);
    pc_chunks_t_.push_back(t);
    pc_mutex_.unlock();
    notifyNewData(false);
    return;
}




std::tuple<std::vector<std::shared_ptr<std::vector<Pointd> > >, std::vector<std::shared_ptr<std::vector<Pointd> > >, ugpm::ImuData, int64_t, int64_t> LidarOdometry::getDataForOptimisation()
{
    pc_mutex_.lock();
    int64_t t0 = pc_chunks_t_.at(0);
    int64_t t1 = pc_chunks_t_.at(1);
    // Get the first and third chunks of point clouds as features
    std::vector<std::shared_ptr<std::vector<Pointd> > > features;
    std::vector<std::shared_ptr<std::vector<Pointd> > > sparse_features;
    
    features.push_back(pc_chunk_features_.at(0));
    sparse_features.push_back(pc_chunk_features_sparse_.at(0));
    features.push_back(pc_chunk_features_.at(1));
    sparse_features.push_back(pc_chunk_features_sparse_.at(1));
    if(features.at(0)->size() == 0 || features.at(1)->size() == 0)
    {
        throw std::runtime_error("LidarOdometry::getDataForOptimisation: No features available for optimisation.");
    }
    pc_mutex_.unlock();

    // Only get the data from the IMU that is relevant for the optimisation
    mutex_.lock();
    int64_t margin = (int64_t)(scan_time_sum_ / (2*scan_count_));
    mutex_.unlock();
    ugpm::ImuData imu_data;
    if(params_.mode != LidarOdometryMode::NO_IMU)
    {
        imu_mutex_.lock();
        if(params_.mode == LidarOdometryMode::GYR)
        {
            imu_data = imu_data_.get(nanosToImuTime(t0 - margin), std::min(imu_data_.gyr.back().t, nanosToImuTime(t1 + 3*margin)));
        }
        else // IMU
        {
            imu_data = imu_data_.get(nanosToImuTime(t0 - margin), std::min(std::max(imu_data_.acc.back().t, imu_data_.gyr.back().t), nanosToImuTime(t1 + 3*margin)));
        }
        imu_mutex_.unlock();
    }

    if(is_2d_)
    {
        for(size_t i = 0; i < imu_data.gyr.size(); ++i)
        {
            imu_data.gyr[i].data[0] = 0.0;
            imu_data.gyr[i].data[1] = 0.0;
        }
    }

    return {features, sparse_features, imu_data, t0, t1};
}





void LidarOdometry::optimise()
{   

    // Get the data for optimisation
    auto [features, sparse_features, imu_data, t0, t1] = getDataForOptimisation();


    State state(imu_data, nanosToImuTime(t0), 200.0, params_.mode);

    // Initialize the state on the first optimisation
    if(first_optimisation_)
    {
        initState(imu_data);
        current_time_ = t0;
    }
    else
    {
        updateCurrentPose(t0, t1);
    }


    std::set<int> types = kTypes;
    if(params_.planar_only)
    {
        types = {1};
    }
    // Create the problem and optimise
    if(first_optimisation_)
    {
        double save_max_feature_dist = params_.max_feature_dist;
        ceres::LossFunction* save_loss_function = loss_function_;
        params_.max_feature_dist = 5.0;
        loss_function_ = new ceres::CauchyLoss(5.0);
        createProblemAssociateAndOptimise(features, sparse_features, state, types, 5, true);
        createProblemAssociateAndOptimise(features, sparse_features, state, types, 5, true);
        createProblemAssociateAndOptimise(features, sparse_features, state, types, 5, true);
        createProblemAssociateAndOptimise(features, sparse_features, state, types, 5, true);
        createProblemAssociateAndOptimise(features, sparse_features, state, types, 5, true);
        createProblemAssociateAndOptimise(features, sparse_features, state, types, 5, true);
        params_.max_feature_dist = save_max_feature_dist;
        delete loss_function_;
        loss_function_ = save_loss_function;
    }

    createProblemAssociateAndOptimise(features, sparse_features, state, types, 25);

    publishResults(state);

    //printState();

    prepareNextState(state);

    
    first_optimisation_ = false;
}


void LidarOdometry::initState(const ugpm::ImuData& imu_data)
{
    if(params_.mode != LidarOdometryMode::IMU)
    {
        return;
    }

    // Create the state times and the K_inv matrix
    Vec3 acc_temp;
    acc_temp[0] = imu_data.acc[0].data[0];
    acc_temp[1] = imu_data.acc[0].data[1];
    acc_temp[2] = imu_data.acc[0].data[2];

    acc_temp.normalize();
    acc_temp *= -params_.g;

    state_blocks_[2] = acc_temp;

}



namespace
{

// Whether two target points were measured by different scanlines: a different ring when the sensor
// reports one, and otherwise a sampling time far enough apart to be a different part of the sweep
// (see kMinTargetTimeSpread)
inline bool differentScanlines(const Pointd& a, const Pointd& b)
{
    if((a.channel != kNoChannel) && (b.channel != kNoChannel))
    {
        return a.channel != b.channel;
    }
    return std::abs(a.t - b.t) > kMinTargetTimeSpread;
}


// How well conditioned the plane through three points is: the sine of the angle at `a`, between the
// two edges leaving it. Zero when the three are collinear, one when they are perpendicular. This is
// the same v1/v2 pair DataAssociation::computeResidual builds its normal from.
inline double planarSine(const Vec3& a, const Vec3& b, const Vec3& c)
{
    const Vec3 v1 = b - a;
    const Vec3 v2 = c - a;
    const double norms = v1.norm()*v2.norm();
    if(norms < 1e-9)
    {
        return 0.0;
    }
    return v1.cross(v2).norm()/norms;
}


} // namespace


std::vector<DataAssociation> LidarOdometry::createProblemAssociateAndOptimise(
        const std::vector<std::shared_ptr<std::vector<Pointd> > >& pts
        , const std::vector<std::shared_ptr<std::vector<Pointd> > >& sparse_pts
        , State& state
        , const std::set<int>& types
        , const int nb_iter
        , bool vel_only)
{
    // Perform the optimisation
    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = false;
    options.max_num_iterations = nb_iter;
    options.num_threads = params_.num_threads;
    options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
    options.function_tolerance = 1e-4;


    // Project the features to the state times. Only the targets of the first point cloud and the
    // sources of the last one are used, by the association and by publishAssociations alike
    std::vector<std::shared_ptr<std::vector<Pointd> > > projected_features = projectPoints(
        pts,
        state,
        state_blocks_,
        state_calib_,
        0);
    std::vector<std::shared_ptr<std::vector<Pointd> > > projected_sparse_features = projectPoints(
        sparse_pts,
        state,
        state_blocks_,
        state_calib_,
        (int)sparse_pts.size() - 1);



    std::vector<DataAssociation> data_associations = getDataAssociations(types, projected_features, projected_sparse_features);

    // What the solver is about to be fed, for the eyes
    if(params_.publish_associations)
    {
        if((node_ != nullptr) && !pts.empty() && !pts.back()->empty())
        {
            node_->publishAssociations(pts.back()->back().t, projected_features,
                    projected_sparse_features, data_associations);
        }
    }

    ceres::Problem::Options pb_options;
    pb_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    // With the state cache, the lidar residuals are evaluated from the state data the callback
    // gathers once per evaluation (see LidarResiduals). Declared before the problem, whose cost
    // functions refer to it, so that it outlives them.
    std::unique_ptr<LidarResiduals> lidar_residuals;
    std::unique_ptr<StateCacheCallback> state_cache_callback;
    if(params_.mode != LidarOdometryMode::NO_IMU)
    {
        lidar_residuals = std::make_unique<LidarResiduals>(state);
        state_cache_callback = std::make_unique<StateCacheCallback>(state, state_blocks_[0], state_blocks_[1], state_blocks_[2], state_blocks_[3], lidar_residuals.get());
        state_cache_callback->PrepareForEvaluation(true, true);
        pb_options.evaluation_callback = state_cache_callback.get();
    }



    ceres::Problem problem(pb_options);
    addBlocks(problem, vel_only);
    addLidarResiduals(problem, data_associations, pts, sparse_pts, state, lidar_residuals.get());

    if(params_.mode == LidarOdometryMode::IMU)
    {
        ZeroPrior* prior = new ZeroPrior(3, 1.0);
        problem.AddResidualBlock(prior, NULL, state_blocks_[0].data());
    }

    // Solve the problem
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    //std::cout << summary.FullReport() << std::endl;

    return data_associations;
}



std::vector<std::shared_ptr<std::vector<Pointd> > > LidarOdometry::projectPoints(
        const std::vector<std::shared_ptr<std::vector<Pointd> > >& pts,
        const State& state,
        const std::vector<Vec3>& state_blocks,
        const Vec7& state_calib,
        const int only_chunk) const
{
    // Prepare the output vector
    std::vector<std::shared_ptr<std::vector<Pointd> > > output(pts.size());

    // Convert the extrinsic calibration state to R and t. The adapter is required: the raw pointer
    // overload of QuaternionToRotation writes row-major, unlike the other ceres rotation functions,
    // and R_calib is column-major, so without it this was the transpose of the calibration. That
    // went unnoticed because both launch file calibrations are rotations by (nearly) pi, which are
    // (nearly) symmetric: exact for os1, 0.03 degrees for Boreas; a 90 degree mount would be 180
    // degrees off. The residuals rotate with the quaternion directly, and were not affected.
    Mat3 R_calib;
    ceres::QuaternionToRotation<double>(state_calib.data(), ceres::ColumnMajorAdapter3x3(R_calib.data()));
    Vec3 t_calib = state_calib.segment<3>(4);

    // The points are only projected for the data association, so each one takes the pose of the
    // state time closest to it rather than one interpolated at its own time: at most half a state
    // period away (2.5 ms at 200 Hz), a few centimetres at range while rotating fast, against a
    // neighbour search of max_feature_dist. That leaves one matrix product per point instead of
    // an interpolation and an angle-axis rotation (a sincos). The residuals and the published
    // clouds keep the interpolated poses. Each state pose is combined with the calibration once:
    // p_W = R_k (R_calib p_L + t_calib) + p_k
    const std::vector<std::pair<Vec3, Mat3> > state_poses = state.statePoses(state_blocks[0], state_blocks[1], state_blocks[2], state_blocks[3]);
    std::vector<Mat3> R_W_L(state_poses.size());
    std::vector<Vec3> t_W_L(state_poses.size());
    for(size_t k = 0; k < state_poses.size(); ++k)
    {
        R_W_L[k] = state_poses[k].second * R_calib;
        t_W_L[k] = state_poses[k].second * t_calib + state_poses[k].first;
    }

    // Project each point cloud to the state times
    for(size_t i = 0; i < pts.size(); ++i)
    {
        output[i] = std::make_shared<std::vector<Pointd> >();
        if((only_chunk >= 0) && ((int)i != only_chunk))
        {
            continue;
        }
        output[i]->resize(pts[i]->size());

        for(size_t j = 0; j < pts[i]->size(); ++j)
        {
            const Pointd& pt = pts[i]->at(j);
            const int k = state.closestStateId(nanosToImuTime(pt.t));
            const Vec3 p_W = R_W_L[k] * pt.vec3() + t_W_L[k];
            output[i]->at(j) = Pointd(p_W, pt.t, pt.i, pt.channel, pt.type);
        }
    }
    return output;
}



std::vector<DataAssociation> LidarOdometry::getDataAssociations(
        const std::set<int>& types
        , const std::vector<std::shared_ptr<std::vector<Pointd> > >& pts
        , const std::vector<std::shared_ptr<std::vector<Pointd> > >& sparse_pts)
{
    // Prepare the output vector and precompute the maximum distance parameter
    std::vector<DataAssociation> data_associations;
    
    std::shared_ptr<std::vector<Pointd> > source_features = sparse_pts.back();
    std::shared_ptr<std::vector<Pointd> > target_features = pts.front();
    int target_id = 0;
    int pc_id = pts.size() - 1;
    data_associations.reserve(source_features->size());

    // Precompute the maximum distance parameter
    double max_dist2 = params_.max_feature_dist*params_.max_feature_dist;

    // Create the kd tree for each feature type. Sized once: a tree must not move after it is built
    std::vector<FeatureTree> feature_trees(types.size());
    std::map<int, int> tree_types;
    int counter = 0;
    for(const auto type: types)
    {
        tree_types[type] = counter;
        counter++;
    }

    // Do the kd tree creation in parallel threads
    std::vector<std::thread> threads;
    // Anonymous function to create a kd tree for a given type
    auto createKdTree = [&](int type, FeatureTree& tree, std::shared_ptr<std::vector<Pointd> > features)
    {
        for(size_t j = 0; j < features->size(); ++j)
        {
            if(features->at(j).type == type)
            {
                Vec3f p = features->at(j).vec3f();
                tree.cloud.pts.push_back({p[0], p[1], p[2]});
                tree.cloud.ids.push_back((int)j);
            }
        }
        if(!tree.cloud.pts.empty())
        {
            tree.index = std::make_unique<FeatureIndex>(3, tree.cloud, nanoflann::KDTreeSingleIndexAdaptorParams(kFeatureTreeLeafSize));
        }
    };
    // Launch the threads
    for(const auto type: types)
    {
        if(type != 3)
        {
            threads.push_back(std::thread(createKdTree, type, std::ref(feature_trees[tree_types[type]]), target_features));
        }
    }
    // Join the threads
    for(auto& thread: threads)
    {
        thread.join();
    }


    std::map<int, std::vector<std::vector<int>>> temp_type_to_ids;
    std::map<int, std::vector<int>> source_downsampled_ids;
    for(size_t i = 0; i < source_features->size(); ++i)
    {
        if(types.find(source_features->at(i).type) == types.end())
        {
            continue;
        }
        if(temp_type_to_ids.find(source_features->at(i).type) == temp_type_to_ids.end())
        {
            temp_type_to_ids[source_features->at(i).type] = std::vector<std::vector<int>>(8);
            source_downsampled_ids[source_features->at(i).type] = std::vector<int>();
        }
        Vec3 p = source_features->at(i).vec3();
        int quadrant = (p[0] >= 0)*4 + (p[1] >= 0)*2 + (p[2] >= 0);
        temp_type_to_ids[source_features->at(i).type][quadrant].push_back(i);
    }

    // Sort the quadrants by number of points, smallest first: each one is capped at its share of the
    // remaining budget, so the small ones take all they have and leave the rest to the larger ones.
    // Largest first capped every quadrant at about an eighth of the budget before the small ones
    // turned out not to use theirs, and that budget was lost
    for(auto& [type, ids]: temp_type_to_ids)
    {
        std::sort(ids.begin(), ids.end(), [](const std::vector<int>& a, const std::vector<int>& b) { return a.size() < b.size(); });
    }


    // For each type, cap the number of candidate sources to 2*params_.max_associations_per_type
    for(const auto& [type, quadrants]: temp_type_to_ids)
    {
        for(size_t i = 0; i < quadrants.size(); i++)
        {
            int cap = (int)((2*params_.max_associations_per_type - source_downsampled_ids[type].size()) / (quadrants.size() - i));
            if(cap <= 0)
                break;
            if(quadrants[i].size() > (size_t)(cap))
            {
                std::vector<int> sampled_ids;
                sampled_ids.reserve(cap);
                std::sample(quadrants[i].begin(), quadrants[i].end(), std::back_inserter(sampled_ids), cap, std::mt19937{std::random_device{}()});
                source_downsampled_ids[type].insert(source_downsampled_ids[type].end(), sampled_ids.begin(), sampled_ids.end());
            }
            else
            {
                source_downsampled_ids[type].insert(source_downsampled_ids[type].end(), quadrants[i].begin(), quadrants[i].end());
            }
        }
        // The search below stops at max_associations_per_type successes: in a random order, the
        // first ones found are a uniform random subset of all the successes, as the shuffle and cap
        // of the associations used to give, without searching for the candidates dropped after.
        // Without the shuffle, the candidates are grouped by quadrant and the first ones would win
        std::shuffle(source_downsampled_ids[type].begin(), source_downsampled_ids[type].end(), std::mt19937{std::random_device{}()});
    }


    
    // Create a hashmap to store the previous associations
    std::map<int, std::vector<DataAssociation>> associations_per_type;
    for(const auto& type: types)
    {
        associations_per_type[type] = std::vector<DataAssociation>();
    }
    // Anonymous function to find the associations for a given type
    auto findAssociations = [&](int type, const std::vector<int>& ids, int pc_id, int target_id, std::vector<DataAssociation>& associations, const FeatureTree& tree, std::shared_ptr<std::vector<Pointd> > source, std::shared_ptr<std::vector<Pointd> > target)
    {
        // The k closest targets within max_feature_dist, closest first, as indexes in the target
        // features. Buffers reused for every search: nn is only valid until the next one
        std::array<uint32_t, std::max(kPlanarCandidates, kEdgeCandidates)> nn_idx;
        std::array<double, std::max(kPlanarCandidates, kEdgeCandidates)> nn_dist2;
        std::vector<int> nn;
        nn.reserve(nn_idx.size());
        auto search = [&](const Vec3& p, const size_t k)
        {
            const size_t nb_found = tree.index->rknnSearch(p.data(), k, nn_idx.data(), nn_dist2.data(), max_dist2);
            nn.clear();
            for(size_t m = 0; m < nb_found; ++m)
            {
                nn.push_back(tree.cloud.ids[nn_idx[m]]);
            }
        };
        for(size_t i = 0; i < ids.size(); ++i)
        {
            if(associations.size() >= params_.max_associations_per_type)
            {
                break;
            }
            Vec3 temp_feature = source->at(ids[i]).vec3();

            if(type == 1)
            {
                search(temp_feature, kPlanarCandidates);

                if(nn.size() < 3)
                    continue;


                int target_feature_id = nn[0];

                const Pointd& point_1 = target->at(target_feature_id);
                Vec3 candidate_1 = point_1.vec3();
                int candidate_2_id = 1;
                // Get the second candidate at a distance greater than params_.min_feature_dist from
                // the first
                while(((size_t)(candidate_2_id) < nn.size())&&
                    ((candidate_1 - target->at(nn[candidate_2_id]).vec3()).norm() < params_.min_feature_dist))
                {
                    candidate_2_id++;
                }

                if((size_t)(candidate_2_id) >= nn.size())
                    continue;

                const Pointd& point_2 = target->at(nn[candidate_2_id]);
                Vec3 candidate_2 = point_2.vec3();

                // Get the third candidate: far enough from BOTH of the first two, spanning a second
                // scanline together with them, and far enough off their line for the normal of the
                // plane to mean anything. Any one of those failing rules the candidate out, hence the
                // disjunction: a triplet on a single scanline is collinear, and its point-to-plane
                // residual is blind to the motion (see kMinPlanarSine).
                int candidate_3_id = candidate_2_id + 1;
                while((size_t)(candidate_3_id) < nn.size())
                {
                    const Pointd& point_3 = target->at(nn[candidate_3_id]);
                    const Vec3 candidate_3 = point_3.vec3();
                    const bool far_enough = ((candidate_1 - candidate_3).norm() >= params_.min_feature_dist)
                            && ((candidate_2 - candidate_3).norm() >= params_.min_feature_dist);
                    const bool spans_scanlines = differentScanlines(point_1, point_2)
                            || differentScanlines(point_1, point_3)
                            || differentScanlines(point_2, point_3);
                    const bool well_conditioned =
                            planarSine(candidate_1, candidate_2, candidate_3) >= kMinPlanarSine;
                    if(far_enough && spans_scanlines && well_conditioned)
                    {
                        break;
                    }
                    candidate_3_id++;
                }

                if((size_t)(candidate_3_id) < nn.size())
                {
                    DataAssociation data_association;
                    data_association.pc_id = pc_id;
                    data_association.feature_id = ids[i];
                    data_association.type = type;
                    data_association.target_ids.push_back(std::make_pair(target_id, nn[0]));
                    data_association.target_ids.push_back(std::make_pair(target_id, nn[candidate_2_id]));
                    data_association.target_ids.push_back(std::make_pair(target_id, nn[candidate_3_id]));

                    associations.push_back(data_association);
                }

            }
            else if((type == 2))
            {
                search(temp_feature, kEdgeCandidates);

                if(nn.size() < 2)
                    continue;


                int target_feature_id = nn[0];

                const Pointd& point_1 = target->at(target_feature_id);
                Vec3 candidate_1 = point_1.vec3();
                // Get the second candidate at a distance greater than params_.min_feature_dist from
                // the first AND off its scanline: two edge points of one scanline lie along that
                // scanline, not along the physical edge, so the line they define is the wrong one and
                // the point-to-line residual measures nothing.
                int candidate_2_id = 1;
                while(((size_t)(candidate_2_id) < nn.size())&&
                    (((candidate_1 - target->at(nn[candidate_2_id]).vec3()).norm() < params_.min_feature_dist)
                     || !differentScanlines(point_1, target->at(nn[candidate_2_id]))))
                {
                    candidate_2_id++;
                }

                if((size_t)(candidate_2_id) < nn.size())
                {
                    DataAssociation data_association;
                    data_association.pc_id = pc_id;
                    data_association.feature_id = ids[i];
                    data_association.type = type;
                    data_association.target_ids.push_back(std::make_pair(target_id, nn[0]));
                    data_association.target_ids.push_back(std::make_pair(target_id, nn[candidate_2_id]));
                    associations.push_back(data_association);
                }
            }
        }
    };

    // Launch the threads to find the associations for each type
    std::vector<std::thread> assoc_threads;
    for(const auto& [type, ids]: source_downsampled_ids)
    {
        if(!feature_trees[tree_types[type]].index)
            continue;
        assoc_threads.push_back(std::thread(findAssociations, type, ids, pc_id, target_id, std::ref(associations_per_type[type]), std::cref(feature_trees[tree_types[type]]), source_features, target_features));
    }
    // Join the threads
    for(auto& thread: assoc_threads)
    {
        thread.join();
    }


    // Already capped at max_associations_per_type per type by the search
    for(const auto& pair: associations_per_type)
    {
        const auto& assocs = pair.second;
        data_associations.insert(data_associations.end(), assocs.begin(), assocs.end());
    }

    return data_associations;
}






void LidarOdometry::addBlocks(ceres::Problem& problem, bool vel_only)
{
    // Add the state variables
    problem.AddParameterBlock(state_blocks_[0].data(), 3);
    ceres::SphereManifold<3>* sphere = new ceres::SphereManifold<3>();
    problem.AddParameterBlock(state_blocks_[2].data(), 3, sphere);
    if(is_2d_)
    {
        // Lock the z component of velocity
        std::vector<int> constant_dims = {2};
        ceres::SubsetManifold* vel_manifold = new ceres::SubsetManifold(3, constant_dims);
        problem.AddParameterBlock(state_blocks_[3].data(), 3, vel_manifold);
        // Lock the x and y component of gyro bias
        constant_dims = {0, 1};
        ceres::SubsetManifold* gyr_manifold = new ceres::SubsetManifold(3, constant_dims);
        problem.AddParameterBlock(state_blocks_[1].data(), 3, gyr_manifold);
    }
    else
    {
        problem.AddParameterBlock(state_blocks_[1].data(), 3);
        problem.AddParameterBlock(state_blocks_[3].data(), 3);
    }

    if(vel_only)
    {
        problem.SetParameterBlockConstant(state_blocks_[0].data());
        problem.SetParameterBlockConstant(state_blocks_[1].data());
        problem.SetParameterBlockConstant(state_blocks_[2].data());
    }

    if(params_.mode != LidarOdometryMode::IMU)
    {
        problem.SetParameterBlockConstant(state_blocks_[0].data());
        problem.SetParameterBlockConstant(state_blocks_[2].data());
    }
}



void LidarOdometry::addLidarResiduals(ceres::Problem& problem
        , const std::vector<DataAssociation>& data_associations
        , const std::vector<std::shared_ptr<std::vector<Pointd> > >& pts
        , const std::vector<std::shared_ptr<std::vector<Pointd> > >& sparse_pts
        , const State& state
        , LidarResiduals* lidar_residuals
        )
{
    // Add the residuals
    for(size_t i = 0; i < data_associations.size(); ++i)
    {
        ceres::CostFunction* cost_function;
        if(lidar_residuals != nullptr)
        {
            const size_t id = lidar_residuals->addResidual(data_associations[i], pts, sparse_pts, lidar_weight_, imu_time_offset_, state_calib_);
            cost_function = new LidarResidualCostFunction(*lidar_residuals, id);
        }
        else
        {
            cost_function = new LidarNoCalCostFunction(state, data_associations[i], pts, sparse_pts, lidar_weight_, imu_time_offset_, state_calib_);
        }

        problem.AddResidualBlock(cost_function, loss_function_, state_blocks_[0].data(), state_blocks_[1].data(), state_blocks_[2].data(), state_blocks_[3].data());
    }

}





void LidarOdometry::printState()
{
    if(params_.mode == LidarOdometryMode::IMU)
    {
        std::cout << "State: " << std::endl;
        std::cout << "    acc_bias: " << state_blocks_[0].transpose() << std::endl;
        std::cout << "    gyr_bias: " << state_blocks_[1].transpose() << std::endl;
        std::cout << "    gravity: " << state_blocks_[2].transpose() << std::endl;
        std::cout << "    vel: " << state_blocks_[3].transpose() << std::endl;
    }
    else if(params_.mode == LidarOdometryMode::GYR)
    {
        std::cout << "State: " << std::endl;
        std::cout << "    gyr_bias: " << state_blocks_[1].transpose() << std::endl;
        std::cout << "    vel: " << state_blocks_[3].transpose() << std::endl;
    }
    else // NO_IMU
    {
        std::cout << "State: " << std::endl;
        std::cout << "    ang_vel: " << state_blocks_[1].transpose() << std::endl;
        std::cout << "    vel: " << state_blocks_[3].transpose() << std::endl;
    }
}



void LidarOdometry::prepareNextState(const State& state)
{
    pc_mutex_.lock();
    int64_t next_query_time = pc_chunks_t_.at(1);
    pc_mutex_.unlock();

    auto [next_pos, next_rot] = state.query(nanosToImuTime(next_query_time), state_blocks_[0], state_blocks_[1], state_blocks_[2], state_blocks_[3]);
    Mat3 R = ugpm::expMap(next_rot);

    if(params_.mode != LidarOdometryMode::NO_IMU)
    {


        acc_bias_sum_ += state_blocks_[0];
        gyr_bias_sum_ += state_blocks_[1];
        bias_count_++;

        if(bias_count_ >= 10)
        {
            state_blocks_[0] = acc_bias_sum_ / bias_count_;
            state_blocks_[1] = gyr_bias_sum_ / bias_count_;
        }
        else
        {
            state_blocks_[0] = Vec3::Zero();
            state_blocks_[1] = Vec3::Zero();
        }

        if(params_.mode == LidarOdometryMode::IMU)
        {
            state_blocks_[2] = R.transpose()*state_blocks_[2];
            state_blocks_[3] = R.transpose()*state_blocks_[3];
        }

        mutex_.lock();
        int64_t margin = (int64_t)(scan_time_sum_ / (2*scan_count_));
        mutex_.unlock();
        imu_mutex_.lock();
        imu_data_ = imu_data_.get(nanosToImuTime(pc_chunks_t_.at(0) - margin), std::numeric_limits<double>::max());
        imu_mutex_.unlock();
    }
    else
    {
        state_blocks_[3] = R.transpose()*state_blocks_[3];
    }

    // Remove the first 2 chunks of point clouds and features
    int n_to_remove = 1;
    pc_mutex_.lock();
    pc_chunks_.erase(pc_chunks_.begin(), pc_chunks_.begin() + n_to_remove);
    pc_chunk_features_.erase(pc_chunk_features_.begin(), pc_chunk_features_.begin() + n_to_remove);
    pc_chunk_features_sparse_.erase(pc_chunk_features_sparse_.begin(), pc_chunk_features_sparse_.begin() + n_to_remove);
    pc_chunks_t_.erase(pc_chunks_t_.begin(), pc_chunks_t_.begin() + n_to_remove);
    pc_mutex_.unlock();

}


void LidarOdometry::updateCurrentPose(const int64_t t0, const int64_t t1)
{
    auto [pos_t0, rot_t0] = prev_state_.query(nanosToImuTime(t0), prev_state_blocks_[0], prev_state_blocks_[1], prev_state_blocks_[2], prev_state_blocks_[3]);
    auto [pos_t1, rot_t1] = prev_state_.query(nanosToImuTime(t1), prev_state_blocks_[0], prev_state_blocks_[1], prev_state_blocks_[2], prev_state_blocks_[3]);
    auto [inv_pos_t0, inv_rot_t0] = invertTransform(pos_t0, rot_t0);
    auto [increment_pos, increment_rot] = combineTransforms(inv_pos_t0, inv_rot_t0, pos_t1, rot_t1);
    current_pose_mutex_.lock();
    std::tie(current_pos_, current_rot_) = combineTransforms(current_pos_, current_rot_, increment_pos, increment_rot);
    current_time_ = t1;
    current_pose_mutex_.unlock();
}

void LidarOdometry::publishResults(const State& state)
{
    int id_to_run = 1;
    pc_mutex_.lock();
    int64_t anchor_t = pc_chunks_t_.at(id_to_run);
    int64_t t1 = pc_chunks_t_.at(1);
    pc_mutex_.unlock();

    auto [pos_t1, rot_t1] = state.query(nanosToImuTime(t1), state_blocks_[0], state_blocks_[1], state_blocks_[2], state_blocks_[3]);
    auto [inv_pos_t1, inv_rot_t1] = invertTransform(pos_t1, rot_t1);


    if(node_ != nullptr) node_->publishTransform(current_time_, current_pos_, current_rot_);

    if(first_optimisation_)
    {

        current_pose_mutex_.lock();
        std::tie(current_pos_, current_rot_) = combineTransforms(current_pos_, current_rot_, inv_pos_t1, inv_rot_t1);
        current_time_ = t1;
        if(node_ != nullptr) node_->publishTransform(current_time_, current_pos_, current_rot_);
        current_pose_mutex_.unlock();
    }

    // Publish the velocity at the current time
    auto [current_twist_linear, current_twist_angular] = state.queryTwist(nanosToImuTime(t1), state_blocks_[0], state_blocks_[1], state_blocks_[2], state_blocks_[3]);
    if(node_ != nullptr) node_->publishTwist(t1, current_twist_linear, current_twist_angular);
    
    // Publish the odometry at the end of the scan
    int64_t end_t = anchor_t + (int64_t)(scan_time_sum_ / scan_count_);
    auto [pos_end, rot_end] = state.query(nanosToImuTime(end_t), state_blocks_[0], state_blocks_[1], state_blocks_[2], state_blocks_[3]);
    auto [increment_pos, increment_rot] = combineTransforms(inv_pos_t1, inv_rot_t1, pos_end, rot_end);
    auto [current_end_pos, current_end_rot] = combineTransforms(current_pos_, current_rot_, increment_pos, increment_rot);
    auto [twist_linear, twist_angular] = state.queryTwist(nanosToImuTime(end_t), state_blocks_[0], state_blocks_[1], state_blocks_[2], state_blocks_[3]);
    if(node_ != nullptr) node_->publishGlobalOdom(end_t, current_end_pos, current_end_rot, twist_linear, twist_angular);


    pc_mutex_.lock();
    // Launch the correction of the point clouds in a separate thread
    if(params_.dense_pc_output)
    {
        std::thread dense_correction_thread(&LidarOdometry::correctAndPublishPc, this, pc_chunks_,pc_chunks_t_, state, state_blocks_, state_calib_, true);
        dense_correction_thread.detach();
    }
    std::thread correction_thread(&LidarOdometry::correctAndPublishPc, this, pc_chunk_features_, pc_chunks_t_, state, state_blocks_, state_calib_, false);
    correction_thread.detach();


    pc_mutex_.unlock();

    prev_state_ = state;
    prev_state_blocks_ = state_blocks_;
    prev_state_calib_ = state_calib_;

}



void LidarOdometry::correctAndPublishPc(
        const std::vector<std::shared_ptr<std::vector<Pointd> > > pts,
        const std::vector<int64_t> pc_chunks_t,
        const State state,
        const std::vector<Vec3> state_blocks,
        const Vec7 state_calib,
        const bool dense)
{
    int64_t current_t;
    current_pose_mutex_.lock();
    current_t = current_time_;
    current_pose_mutex_.unlock();

    int64_t end_t = pc_chunks_t.at(1);
    end_t += (int64_t)(scan_time_sum_ / scan_count_);
    if(end_t < current_t)
    {
        std::cout << "Skipping point cloud correction, current time is " << current_t << " and end time is " << end_t << std::endl;
        return;
    }
    

    auto [pos_t1, rot_t1] = state.query(nanosToImuTime(pc_chunks_t.at(1)), state_blocks[0], state_blocks[1], state_blocks[2], state_blocks[3]);

    auto [inv_pos_t1, inv_rot_t1] = invertTransform(pos_t1, rot_t1);

    Vec3 r_calib;
    ceres::QuaternionToAngleAxis<double>(state_calib.data(), r_calib.data());
    Vec3 t_calib = state_calib.segment<3>(4);

    // Both are the same for every point of the scan, so they are turned into rotation matrices once
    const Mat3 R_calib = expMap(r_calib);
    const Mat3 R_inv_t1 = expMap(inv_rot_t1);

    // Correct the points of chuncks 1 and 2 and publish them
    std::vector<Pointd> pc_corrected;
    pc_corrected.reserve(pts.at(1)->size());
    int nb_to_run = 2;
    for(int i = 1; i < nb_to_run; ++i)
    {
        std::vector<double> chunk_t;
        chunk_t.reserve(pts.at(i)->size());
        for(size_t j = 0; j < pts.at(i)->size(); ++j)
        {
            chunk_t.push_back(nanosToImuTime(pts.at(i)->at(j).t));
        }
        std::vector<std::pair<Vec3, Vec3>> poses = state.queryApprox(chunk_t, state_blocks[0], state_blocks[1], state_blocks[2], state_blocks[3]);

        for(size_t j = 0; j < pts.at(i)->size(); ++j)
        {
            // The point in the frame at t1, R_inv_t1 * (R_j * (R_calib * p_L + t_calib) + pos_j) +
            // inv_pos_t1: the three transforms applied in turn. Composing them as angle-axis vectors
            // instead cost six conversions to and from rotation matrices per point, two of them of
            // the per-scan constants above, and was a tenth of the node's cpu.
            const Vec3 p_I = R_calib * pts.at(i)->at(j).vec3() + t_calib;
            Vec3 p_W;
            ceres::AngleAxisRotatePoint<double>(poses[j].second.data(), p_I.data(), p_W.data());
            p_W += poses[j].first;
            const Vec3 p_t1 = R_inv_t1 * p_W + inv_pos_t1;
            pc_corrected.push_back(Pointd(p_t1, pts.at(i)->at(j).t, pts.at(i)->at(j).i, pts.at(i)->at(j).channel, pts.at(i)->at(j).type));
        }

    }

                


    if(dense)
    {
        if(node_ != nullptr) node_->publishPcDense(pc_chunks_t.at(1), pc_corrected);
    }
    else
    {
        // Sort the point cloud by time
        std::sort(pc_corrected.begin(), pc_corrected.end(), [](const Pointd& a, const Pointd& b) {
            return a.t < b.t;
        });
        if(node_ != nullptr) node_->publishPc(pc_chunks_t.at(1), pc_corrected);
    }

}



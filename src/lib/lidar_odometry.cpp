#include "lice/lidar_odometry.h"
#include <iostream>
#include <random>
#include "KDTree.h"
#include "lice/lidar_odometry_node.h"
#include "ankerl/unordered_dense.h"
#include <ctime>

typedef jk::tree::KDTree<std::pair<int,int>, 3, 16> KDTree3;
typedef jk::tree::KDTree<int, 3, 16> KDTree3Simple;




LidarOdometry::LidarOdometry(const LidarOdometryParams& params, LidarOdometryNode* node)
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

}

void LidarOdometry::stop()
{
    mutex_.lock();
    running_ = false;
    mutex_.unlock();
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
        size_t id_to_run = (params_.low_latency) ? 1 : 2;
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
            std::this_thread::sleep_for(std::chrono::milliseconds(int(kPullPeriod*1000)));
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
    if(pc->size() == 0)
    {
        return;
    }


    // Sort the points by time
    if(params_.unsorted_pc)
    {
        std::sort(pc->begin(), pc->end(), [](const Pointd& a, const Pointd& b) { return a.t < b.t; });
    }

    double threshold = params_.feature_voxel_size;

    pc_mutex_.lock();
    bool is_odd = (pc_chunks_.size() % 2 == 1);
    if((!params_.low_latency) && is_odd)
    {
        pc_chunk_features_.push_back(std::make_shared<std::vector<Pointd> >());
        pc_chunk_features_sparse_.push_back(std::make_shared<std::vector<Pointd> >());
        pc_chunks_.push_back(pc);
        pc_chunks_t_.push_back(t);
        pc_mutex_.unlock();
        return;
    }
    pc_mutex_.unlock();

    // Split the point cloud into channels and remove points too close
    std::vector<std::vector<Pointd> > channels;
    if(is_2d_)
    {
        channels.push_back(*pc);
    }
    else
    {
        bool has_channel = pc->at(0).channel != kNoChannel;
        if(!has_channel)
        {
            throw std::runtime_error("LidarOdometry::extractEdgeFeatures: Point cloud does not have a channel. Current implementation requires point clouds to be split by channel.");
        }
        else
        {
            channels = splitChannels(*pc, params_.min_range, params_.max_feature_range);
        }
    }


    // Get the median time between points in each channel
    if(median_dt_ < 0.0)
    {
        // Get the median Dt for all channels
        std::vector<int64_t> dt;
        for(size_t i = 0; i < channels.size(); ++i)
        {
            if(channels[i].size() < 3)
            {
                continue;
            }
            dt.push_back(getMedianDt(channels[i]));
        }
        std::sort(dt.begin(), dt.end());
        median_dt_ = dt[dt.size()/2];
        if(median_dt_ <= 0.0)
        {
            dt.clear();
            for(size_t i = 0; i < channels.size(); ++i)
            {
                if(channels[i].size() < 3)
                {
                    continue;
                }
                dt.push_back(getMeanDt(channels[i]));
            }
            std::sort(dt.begin(), dt.end());
            median_dt_ = dt[dt.size()/2];

        }
        if(median_dt_ <= 0.0)
        {
            std::cout << "ERROR !!!!!!!!!!!!!!!!!!!!! LidarOdometry::extractEdgeFeatures: Median dt is zero or negative, cannot compute features." << std::endl;
            return;
        }
    }


    std::shared_ptr<std::vector<Pointd> > output(new std::vector<Pointd>());
    std::shared_ptr<std::vector<Pointd> > downsample(new std::vector<Pointd>());
    int64_t time_thr = (int64_t)(median_dt_ * 1.5); // 100 times the median dt
    int64_t time_thr_far = (int64_t)(median_dt_ * 100); // 100 times the median dt
    double min_range = 1.1* params_.min_range;

    // For each channel, extract the edge features
    for(size_t i = 0; i < channels.size(); ++i)
    {
        if(channels[i].size() < 3)
        {
            continue; // Not enough points to extract features
        }

        std::vector<double> ranges(channels[i].size());
        for(size_t j = 0; j < channels[i].size(); ++j)
        {
            ranges[j] = std::sqrt(channels[i][j].x * channels[i][j].x + 
                                  channels[i][j].y * channels[i][j].y +
                                  channels[i][j].z * channels[i][j].z);
        }

        // Loop through the points in the channel
        Pointd last_point = channels[i].front();
        downsample->push_back(last_point);
        downsample->back().type = 1; // Set the type to 1 (planar)
        for(size_t j = 1; j < channels[i].size() - 1; ++j)
        {
            if(ranges[j] < min_range || ranges[j] > params_.max_feature_range)
            {
                continue; // Skip points that are too close
            }


            if((!params_.planar_only) && (!is_2d_))
            {
                int64_t dt_1 = channels[i][j].nanos() - channels[i][j-1].nanos();
                int64_t dt_2 = channels[i][j+1].nanos() - channels[i][j].nanos();
                if(dt_1 > time_thr && dt_2 > time_thr)
                {
                    continue;
                }
                double delta_1 = ranges[j] - ranges[j-1];
                double delta_2 = ranges[j] - ranges[j+1];
                double min_delta = std::min(std::abs(delta_1), std::abs(delta_2));
                if(min_delta > threshold)
                {
                    continue;
                }
                double local_thr = std::max(5*min_delta, threshold);
                
                if( (dt_1 < time_thr)
                        && (std::abs(delta_1) < threshold)
                        && ((dt_2 > time_thr_far) || (delta_2 < -local_thr)) )
                {
                    output->push_back(channels[i][j]);
                    output->back().type = 2; // Set the type to 2 (edge)
                    continue;
                }
                else if( (dt_2 < time_thr)
                        && (std::abs(delta_2) < threshold)
                        && ((dt_1 > time_thr_far) || (delta_1 < -local_thr)) )
                {
                    output->push_back(channels[i][j]);
                    output->back().type = 2; // Set the type to 2 (edge)
                    continue;
                }
            }

            // Check the distance with the last point
            double distance = std::sqrt(
                (channels[i][j].x - last_point.x) * (channels[i][j].x - last_point.x) +
                (channels[i][j].y - last_point.y) * (channels[i][j].y - last_point.y) +
                (channels[i][j].z - last_point.z) * (channels[i][j].z - last_point.z));
            double rand_val = ((double) rand() / (RAND_MAX));
            if(distance > params_.feature_voxel_size*(0.5 + rand_val))
            //if(distance > params_.feature_voxel_size)
            {
                last_point = channels[i][j];
                downsample->push_back(channels[i][j]);
                downsample->back().type = is_2d_ ? 2 : 1; // Set the type to 1 (planar) or 2 (edge) depending on is_2d_
            }
        }
    }

    *downsample = downsamplePointCloudSubset(*downsample, params_.feature_voxel_size);
    
    std::shared_ptr<std::vector<Pointd> > sparse_features(new std::vector<Pointd>());
    *sparse_features = downsamplePointCloudSubset(*downsample, 2*params_.feature_voxel_size);
    // Concatenate the edge features to the sparse features
    if((!params_.planar_only) && (!is_2d_))
    {
        std::vector<Pointd> edge_sparse_features = downsamplePointCloudSubset(*output, params_.feature_voxel_size);
        sparse_features->insert(sparse_features->end(), edge_sparse_features.begin(), edge_sparse_features.end());
    }

    // Concatenate the downsampled features and the edge features
    output->insert(output->end(), downsample->begin(), downsample->end());


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
    return;
}




std::tuple<std::vector<std::shared_ptr<std::vector<Pointd> > >, std::vector<std::shared_ptr<std::vector<Pointd> > >, ugpm::ImuData, int64_t, int64_t> LidarOdometry::getDataForOptimisation()
{
    pc_mutex_.lock();
    int64_t t0 = pc_chunks_t_.at(0);
    int64_t t1 = pc_chunks_t_.at( (params_.low_latency) ? 1 : 2);
    // Get the first and third chunks of point clouds as features
    std::vector<std::shared_ptr<std::vector<Pointd> > > features;
    std::vector<std::shared_ptr<std::vector<Pointd> > > sparse_features;
    
    features.push_back(pc_chunk_features_.at(0));
    sparse_features.push_back(pc_chunk_features_sparse_.at(0));
    if(params_.low_latency)
    {
        features.push_back(pc_chunk_features_.at(1));
        sparse_features.push_back(pc_chunk_features_sparse_.at(1));
    }
    else
    {
        features.push_back(pc_chunk_features_.at(2));
        sparse_features.push_back(pc_chunk_features_sparse_.at(2));
    }
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
    StopWatch sw;
    sw.start();

    // Get the data for optimisation
    auto [features, sparse_features, imu_data, t0, t1] = getDataForOptimisation();


    State state(imu_data, nanosToImuTime(t0), params_.state_frequency, params_.mode);

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


    sw.stop();
    sw.print("Precomputations: ");


    sw.reset();
    sw.start();

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

    printState();
    logState();

    prepareNextState(state);

    sw.stop();
    sw.print("Optimisation and launch publisher: ");

    
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



std::vector<DataAssociation> LidarOdometry::createProblemAssociateAndOptimise(
        const std::vector<std::shared_ptr<std::vector<Pointd> > >& pts
        , const std::vector<std::shared_ptr<std::vector<Pointd> > >& sparse_pts
        , const State& state
        , const std::set<int>& types
        , const int nb_iter
        , bool vel_only)
{
    // Perform the optimisation
    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = false;
    options.max_num_iterations = nb_iter;
    options.num_threads = 8;
    options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
    options.function_tolerance = 1e-4;


    // Project the features to the state times
    std::vector<std::shared_ptr<std::vector<Pointd> > > projected_features = projectPoints(
        pts,
        state,
        state_blocks_,
        time_offset_,
        state_calib_);
    std::vector<std::shared_ptr<std::vector<Pointd> > > projected_sparse_features = projectPoints(
        sparse_pts,
        state,
        state_blocks_,
        time_offset_,
        state_calib_);



    std::vector<DataAssociation> data_associations = getDataAssociations(types, projected_features, projected_sparse_features);


    ceres::Problem::Options pb_options;
    pb_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;

    ceres::Problem problem(pb_options);
    addBlocks(problem, vel_only);
    addLidarResiduals(problem, data_associations, pts, sparse_pts, state);

    ZeroPrior* prior = new ZeroPrior(3, 1.0);
    problem.AddResidualBlock(prior, NULL, state_blocks_[0].data());

    // Solve the problem
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << std::endl;


    return data_associations;
}



std::vector<std::shared_ptr<std::vector<Pointd> > > LidarOdometry::projectPoints(
        const std::vector<std::shared_ptr<std::vector<Pointd> > >& pts,
        const State& state,
        const std::vector<Vec3>& state_blocks,
        const double time_offset,
        const Vec7& state_calib) const
{
    // Prepare the output vector
    std::vector<std::shared_ptr<std::vector<Pointd> > > output(pts.size());

    // Convert the extrinsic calibration state to R and t
    Mat3 R_calib;
    ceres::QuaternionToRotation<double>(state_calib.data(), R_calib.data());
    Vec3 t_calib = state_calib.segment<3>(4);

    // Project each point cloud to the state times
    for(size_t i = 0; i < pts.size(); ++i)
    {
        output[i] = std::make_shared<std::vector<Pointd> >();
        output[i]->resize(pts[i]->size());

        // Collect the point times to perform a single query to the state
        std::vector<double> temp_times;
        temp_times.reserve(pts[i]->size());
        for(size_t j = 0; j < pts[i]->size(); ++j)
        {
            temp_times.push_back(nanosToImuTime(pts[i]->at(j).t));
        }
        std::vector<std::pair<Vec3, Vec3>> poses = state.queryApprox(temp_times, state_blocks[0], state_blocks[1], state_blocks[2], state_blocks[3], time_offset);

        // Apply the transformations to each point
        for(size_t j = 0; j < pts[i]->size(); ++j)
        {
            Vec3& pos = poses[j].first;
            Vec3& rot = poses[j].second;

            Vec3 p_L = pts[i]->at(j).vec3();
            Vec3 p_I = R_calib * p_L + t_calib;
            Vec3 p_W;
            ceres::AngleAxisRotatePoint<double>(rot.data(), p_I.data(), p_W.data());
            p_W += pos;

            output[i]->at(j) = Pointd(p_W, pts[i]->at(j).t, pts[i]->at(j).i, pts[i]->at(j).channel, pts[i]->at(j).type);
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

    // Create the kd tree for each feature type
    std::vector<std::shared_ptr<KDTree3Simple>> feature_kd_trees;
    std::map<int, int> tree_types;
    int counter = 0;
    for(const auto type: types)
    {
        tree_types[type] = counter;
        feature_kd_trees.push_back(std::make_shared<KDTree3Simple>());
        counter++;
    }

    // Do the kd tree creation in parallel threads
    std::vector<std::thread> threads;
    // Anonymous function to create a kd tree for a given type
    auto createKdTree = [&](int type, std::shared_ptr<KDTree3Simple> tree, std::shared_ptr<std::vector<Pointd> > features)
    {
        for(size_t j = 0; j < features->size(); ++j)
        {
            if(features->at(j).type == type)
            {
                Vec3f p = features->at(j).vec3f();
                tree->addPoint({p[0], p[1], p[2]}, j);
            }
        }
    };
    // Launch the threads
    for(const auto type: types)
    {
        if(type != 3)
        {
            threads.push_back(std::thread(createKdTree, type, feature_kd_trees[tree_types[type]], target_features));
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

    // Sort the by number of points in each quadrant
    for(auto& [type, ids]: temp_type_to_ids)
    {
        std::sort(ids.begin(), ids.end(), [](const std::vector<int>& a, const std::vector<int>& b) { return a.size() > b.size(); });
    }


    // For each type, cap the number of associations to 2*params_.max_associations_per_type
    for(const auto& [type, quadrants]: temp_type_to_ids)
    {
        for(size_t i = 0; i < quadrants.size(); i++)
        {
            int cap = std::ceil(2*params_.max_associations_per_type - source_downsampled_ids[type].size()) / (quadrants.size() - i);
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
    }
    

    
    // Create a hashmap to store the previous associations
    std::map<int, std::vector<DataAssociation>> associations_per_type;
    for(const auto& type: types)
    {
        associations_per_type[type] = std::vector<DataAssociation>();
    }
    // Anonymous function to find the associations for a given type
    auto findAssociations = [&](int type, const std::vector<int>& ids, int pc_id, int target_id, std::vector<DataAssociation>& associations, KDTree3Simple& tree, std::shared_ptr<std::vector<Pointd> > source, std::shared_ptr<std::vector<Pointd> > target)
    {
        for(size_t i = 0; i < ids.size(); ++i)
        {
            Vec3 temp_feature = source->at(ids[i]).vec3();

            if(type == 1)
            {
                auto nn = tree.searchCapacityLimitedBall({temp_feature(0), temp_feature(1), temp_feature(2)}, max_dist2, 6);

                if(nn.size() < 3)
                    continue;


                int target_feature_id = nn[0].payload;

                Vec3 candidate_1 = target->at(target_feature_id).vec3();
                int candidate_2_id = 1;
                // Get the second cadidate with at distance greater that params_.min_feature_dist between the first and the second


                while(((size_t)(candidate_2_id) < nn.size())&&
                    ((candidate_1 - target->at(nn[candidate_2_id].payload).vec3()).norm() < params_.min_feature_dist))
                {
                    candidate_2_id++;
                }

                if((size_t)(candidate_2_id) >= nn.size())
                    continue;


                int candidate_3_id = candidate_2_id + 1;
                Vec3 candidate_2 = target->at(nn[candidate_2_id].payload).vec3();

                Vec3 v1 = candidate_2 - candidate_1;
                v1.normalize();
                // Get the third candidate with at distance greater that params_.min_feature_dist between to the first, the second and the third, also check that the three points are not aligned
                while(((size_t)(candidate_3_id) < nn.size())&&
                    ((candidate_1 - target->at(nn[candidate_3_id].payload).vec3()).norm() < params_.min_feature_dist)&&
                    ((candidate_2 - target->at(nn[candidate_3_id].payload).vec3()).norm() < params_.min_feature_dist)&&
                    (std::abs(v1.dot((target->at(nn[candidate_3_id].payload).vec3() - candidate_2).normalized())) > 0.2))
                {
                    candidate_3_id++;
                }

                if((size_t)(candidate_3_id) < nn.size())
                {
                    DataAssociation data_association;
                    data_association.pc_id = pc_id;
                    data_association.feature_id = ids[i];
                    data_association.type = type;
                    data_association.target_ids.push_back(std::make_pair(target_id, nn[0].payload));
                    data_association.target_ids.push_back(std::make_pair(target_id, nn[candidate_2_id].payload));
                    data_association.target_ids.push_back(std::make_pair(target_id, nn[candidate_3_id].payload));

                    associations.push_back(data_association);
                }

            }
            else if((type == 2))
            {
                auto nn = tree.searchCapacityLimitedBall({temp_feature(0), temp_feature(1), temp_feature(2)}, max_dist2, 5);

                if(nn.size() < 2) 
                    continue;


                int target_feature_id = nn[0].payload;

                Vec3 candidate_1 = target->at(target_feature_id).vec3();
                int candidate_2_id = 1;
                // Get the second cadidate with at distance greater that params_.min_feature_dist between the first and the second


                while(((size_t)(candidate_2_id) < nn.size())&&
                    ((candidate_1 - target->at(nn[candidate_2_id].payload).vec3()).norm() < params_.min_feature_dist))
                {
                    candidate_2_id++;
                }

                if((size_t)(candidate_2_id) < nn.size())
                {
                    DataAssociation data_association;
                    data_association.pc_id = pc_id;
                    data_association.feature_id = ids[i];
                    data_association.type = type;
                    data_association.target_ids.push_back(std::make_pair(target_id, nn[0].payload));
                    data_association.target_ids.push_back(std::make_pair(target_id, nn[candidate_2_id].payload));
                    associations.push_back(data_association);
                }
            }
        }
    };

    // Launch the threads to find the associations for each type
    std::vector<std::thread> assoc_threads;
    for(const auto& [type, ids]: source_downsampled_ids)
    {
        if(feature_kd_trees[tree_types[type]]->size() == 0)
            continue;
        assoc_threads.push_back(std::thread(findAssociations, type, ids, pc_id, target_id, std::ref(associations_per_type[type]), std::ref(*(feature_kd_trees[tree_types[type]])), source_features, target_features));
    }
    // Join the threads
    for(auto& thread: assoc_threads)
    {
        thread.join();
    }


    for(const auto& pair: associations_per_type)
    {
        const auto& assocs = pair.second;

        if(assocs.size() > params_.max_associations_per_type)
        {
            std::vector<size_t> indices(assocs.size());
            std::iota(indices.begin(), indices.end(), 0);
            std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});
            for(size_t i = 0; i < params_.max_associations_per_type; ++i)
            {
                data_associations.push_back(assocs[indices[i]]);
            }
        }
        else
        {
            data_associations.insert(data_associations.end(), assocs.begin(), assocs.end());
        }
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

    problem.AddParameterBlock(&time_offset_, 1);

    problem.SetParameterBlockConstant(&time_offset_);

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
        )
{
    // Add the residuals
    for(size_t i = 0; i < data_associations.size(); ++i)
    {
        LidarNoCalCostFunction* cost_function = new LidarNoCalCostFunction(state, data_associations[i], pts, sparse_pts, lidar_weight_, imu_time_offset_, state_calib_);

        problem.AddResidualBlock(cost_function, loss_function_, state_blocks_[0].data(), state_blocks_[1].data(), state_blocks_[2].data(), state_blocks_[3].data(), &time_offset_);
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

void LidarOdometry::logState()
{
    std::string log_path = "/home/ced/ros2_ws/install/ffastllamaa/share/ffastllamaa/maps/lidar_odometry_state.csv";

    std::ofstream log_file;
    if(first_optimisation_)
    {
        log_file.open(log_path, std::ios::out);
    }
    else
    {
        log_file.open(log_path, std::ios::app);
    }
    if(!log_file.is_open())
    {
        std::cerr << "LidarOdometry::logState: Unable to open log file: " << log_path << std::endl;
        return;
    }
    if(first_optimisation_)
    {
        log_file << "time,x,y,z,rx,ry,rz,acc_bias_x,acc_bias_y,acc_bias_z,gyr_bias_x,gyr_bias_y,gyr_bias_z,gravity_x,gravity_y,gravity_z,vel_x,vel_y,vel_z,time_offset,calib_qw,calib_qx,calib_qy,calib_qz,calib_tx,calib_ty,calib_tz\n";
    }

    log_file << std::fixed << current_time_ << ","
                << current_pos_[0] << "," << current_pos_[1] << "," << current_pos_[2] << ","
                << current_rot_[0] << "," << current_rot_[1] << "," << current_rot_[2] << ","
                << state_blocks_[0][0] << "," << state_blocks_[0][1] << "," << state_blocks_[0][2] << ","
                << state_blocks_[1][0] << "," << state_blocks_[1][1] << "," << state_blocks_[1][2] << ","
                << state_blocks_[2][0] << "," << state_blocks_[2][1] << "," << state_blocks_[2][2] << ","
                << state_blocks_[3][0] << "," << state_blocks_[3][1] << "," << state_blocks_[3][2] << ","
                << time_offset_ << ","
                << state_calib_[0] << "," << state_calib_[1] << "," << state_calib_[2] << "," << state_calib_[3] << ","
                << state_calib_[4] << "," << state_calib_[5] << "," << state_calib_[6] << "\n";
    log_file.flush();
    log_file.close();
}




void LidarOdometry::prepareNextState(const State& state)
{
    pc_mutex_.lock();
    int64_t next_query_time = (params_.low_latency) ? pc_chunks_t_.at(1) : pc_chunks_t_.at(2);
    pc_mutex_.unlock();

    auto [next_pos, next_rot] = state.query(nanosToImuTime(next_query_time), state_blocks_[0], state_blocks_[1], state_blocks_[2], state_blocks_[3], time_offset_);
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
    int n_to_remove = (params_.low_latency) ? 1 : 2;
    pc_mutex_.lock();
    pc_chunks_.erase(pc_chunks_.begin(), pc_chunks_.begin() + n_to_remove);
    pc_chunk_features_.erase(pc_chunk_features_.begin(), pc_chunk_features_.begin() + n_to_remove);
    pc_chunk_features_sparse_.erase(pc_chunk_features_sparse_.begin(), pc_chunk_features_sparse_.begin() + n_to_remove);
    pc_chunks_t_.erase(pc_chunks_t_.begin(), pc_chunks_t_.begin() + n_to_remove);
    pc_mutex_.unlock();

}


void LidarOdometry::updateCurrentPose(const int64_t t0, const int64_t t1)
{
    auto [pos_t0, rot_t0] = prev_state_.query(nanosToImuTime(t0), prev_state_blocks_[0], prev_state_blocks_[1], prev_state_blocks_[2], prev_state_blocks_[3], prev_time_offset_);
    auto [pos_t1, rot_t1] = prev_state_.query(nanosToImuTime(t1), prev_state_blocks_[0], prev_state_blocks_[1], prev_state_blocks_[2], prev_state_blocks_[3], prev_time_offset_);
    auto [inv_pos_t0, inv_rot_t0] = invertTransform(pos_t0, rot_t0);
    auto [increment_pos, increment_rot] = combineTransforms(inv_pos_t0, inv_rot_t0, pos_t1, rot_t1);
    current_pose_mutex_.lock();
    std::tie(current_pos_, current_rot_) = combineTransforms(current_pos_, current_rot_, increment_pos, increment_rot);
    current_time_ = t1;
    current_pose_mutex_.unlock();
}

void LidarOdometry::publishResults(const State& state)
{
    StopWatch sw;
    sw.start();

    int id_to_run = (params_.low_latency) ? 1 : 2;
    pc_mutex_.lock();
    int64_t anchor_t = pc_chunks_t_.at(id_to_run);
    int64_t t1 = pc_chunks_t_.at(1);
    pc_mutex_.unlock();

    auto [pos_t1, rot_t1] = state.query(nanosToImuTime(t1), state_blocks_[0], state_blocks_[1], state_blocks_[2], state_blocks_[3], time_offset_);
    auto [inv_pos_t1, inv_rot_t1] = invertTransform(pos_t1, rot_t1);


    node_->publishTransform(current_time_, current_pos_, current_rot_);

    if(first_optimisation_)
    {

        current_pose_mutex_.lock();
        std::tie(current_pos_, current_rot_) = combineTransforms(current_pos_, current_rot_, inv_pos_t1, inv_rot_t1);
        current_time_ = t1;
        node_->publishTransform(current_time_, current_pos_, current_rot_);
        current_pose_mutex_.unlock();
    }

    
    // Publish the odometry at the end of the scan
    int64_t end_t = anchor_t + (int64_t)(scan_time_sum_ / scan_count_);
    auto [pos_end, rot_end] = state.query(nanosToImuTime(end_t), state_blocks_[0], state_blocks_[1], state_blocks_[2], state_blocks_[3], time_offset_);
    auto [increment_pos, increment_rot] = combineTransforms(inv_pos_t1, inv_rot_t1, pos_end, rot_end);
    auto [current_end_pos, current_end_rot] = combineTransforms(current_pos_, current_rot_, increment_pos, increment_rot);
    auto [twist_linear, twist_angular] = state.queryTwist(nanosToImuTime(end_t), state_blocks_[0], state_blocks_[1], state_blocks_[2], state_blocks_[3], time_offset_);
    node_->publishGlobalOdom(end_t, current_end_pos, current_end_rot, twist_linear, twist_angular);


    pc_mutex_.lock();
    // Launch the correction of the point clouds in a separate thread
    if(params_.dense_pc_output)
    {
        std::thread dense_correction_thread(&LidarOdometry::correctAndPublishPc, this, pc_chunks_,pc_chunks_t_, state, state_blocks_, state_calib_, time_offset_, true);
        dense_correction_thread.detach();
    }
    std::thread correction_thread(&LidarOdometry::correctAndPublishPc, this, pc_chunk_features_, pc_chunks_t_, state, state_blocks_, state_calib_, time_offset_, false);
    correction_thread.detach();


    pc_mutex_.unlock();

    prev_state_ = state;
    prev_state_blocks_ = state_blocks_;
    prev_state_calib_ = state_calib_;
    prev_time_offset_ = time_offset_;


    sw.stop();
    sw.print("Querying and publishing odom results: ");
}



void LidarOdometry::correctAndPublishPc(
        const std::vector<std::shared_ptr<std::vector<Pointd> > > pts,
        const std::vector<int64_t> pc_chunks_t,
        const State state,
        const std::vector<Vec3> state_blocks,
        const Vec7 state_calib,
        const double time_offset,
        const bool dense)
{
    StopWatch sw;
    sw.start();

    int64_t current_t;
    current_pose_mutex_.lock();
    current_t = current_time_;
    current_pose_mutex_.unlock();

    int64_t end_t = (params_.low_latency) ? pc_chunks_t.at(1) : pc_chunks_t.at(2);
    end_t += (int64_t)(scan_time_sum_ / scan_count_);
    if(end_t < current_t)
    {
        std::cout << "Skipping point cloud correction, current time is " << current_t << " and end time is " << end_t << std::endl;
        return;
    }
    

    auto [pos_t1, rot_t1] = state.query(nanosToImuTime(pc_chunks_t.at(1)), state_blocks[0], state_blocks[1], state_blocks[2], state_blocks[3], time_offset);

    auto [inv_pos_t1, inv_rot_t1] = invertTransform(pos_t1, rot_t1);

    Vec3 r_calib;
    ceres::QuaternionToAngleAxis<double>(state_calib.data(), r_calib.data());
    Vec3 t_calib = state_calib.segment<3>(4);

    // Correct the points of chuncks 1 and 2 and publish them
    std::vector<Pointd> pc_corrected;
    if(params_.low_latency)
    {
        pc_corrected.reserve(pts.at(1)->size());
    }
    else
    {
        pc_corrected.reserve(pts.at(1)->size() + pts.at(2)->size());
    }
    int nb_to_run = (params_.low_latency) ? 2 : 3;
    for(int i = 1; i < nb_to_run; ++i)
    {
        std::vector<double> chunk_t;
        chunk_t.reserve(pts.at(i)->size());
        for(size_t j = 0; j < pts.at(i)->size(); ++j)
        {
            chunk_t.push_back(nanosToImuTime(pts.at(i)->at(j).t));
        }
        std::vector<std::pair<Vec3, Vec3>> poses = state.queryApprox(chunk_t, state_blocks[0], state_blocks[1], state_blocks[2], state_blocks[3], time_offset);

        for(size_t j = 0; j < pts.at(i)->size(); ++j)
        {
            auto[pos, rot] = combineTransforms(poses[j].first, poses[j].second, t_calib, r_calib);
            std::tie(pos, rot) = combineTransforms(inv_pos_t1, inv_rot_t1, pos, rot);
            Vec3 p_L = pts.at(i)->at(j).vec3();
            Vec3 p_t1;
            ceres::AngleAxisRotatePoint<double>(rot.data(), p_L.data(), p_t1.data());
            p_t1 += pos;
            pc_corrected.push_back(Pointd(p_t1, pts.at(i)->at(j).t, pts.at(i)->at(j).i, pts.at(i)->at(j).channel, pts.at(i)->at(j).type));
        }

    }

                

    sw.stop();
    sw.print("Correcting point clouds of " + std::to_string(pc_corrected.size()) + " points: ");

    sw.reset();
    sw.start();

    if(dense)
    {
        node_->publishPcDense(pc_chunks_t.at(1), pc_corrected);
    }
    else
    {
        // Sort the point cloud by time
        std::sort(pc_corrected.begin(), pc_corrected.end(), [](const Pointd& a, const Pointd& b) {
            return a.t < b.t;
        });
        node_->publishPc(pc_chunks_t.at(1), pc_corrected);
    }

    sw.stop();
    sw.print("Publishing point cloud: ");

}



#pragma once

#include "lice/types.h"
#include "lice/pointcloud_utils.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>


// Multiples of the median time between two consecutive points of a channel, used to detect the
// scanline discontinuities and the occlusions in the edge extraction
const double kFeatureTimeThrFactor = 1.5;
const double kFeatureTimeThrFarFactor = 100.0;
// The points closer than that (times the minimum range) are not considered as features
const double kFeatureMinRangeFactor = 1.1;


// Parameters of the lidar feature extraction. Subset of LidarOdometryParams with the same meaning
// and the same ROS parameter names, kept in a separate structure so that the nodes that do not run
// the lidar odometry can share the extraction.
struct FeatureExtractionParams
{
    double min_range = 1.0;
    double max_feature_range = 150.0;
    double feature_voxel_size = 0.3;
    double intensity_threshold = -1.0; // If > 0, only keep the points with an intensity above it
    bool planar_only = false;          // If true, do not extract the edge features
    bool unsorted_pc = false;          // If true, the incoming point cloud is not sorted by time
    bool is_2d = false;                // If true, the whole cloud is treated as a single channel
};


// Extract the lidar features of a raw point cloud, in the lidar frame (the edge detection relies on
// the per-channel scanline ordering and on the ranges as measured by the sensor).
// `pc` is sorted in place by timestamp if `params.unsorted_pc` is set.
// `median_dt` caches the median time between two consecutive points of a channel: it must be
// initialised to -1 by the caller and kept across the calls (it is only estimated once).
// `features` is filled with the edge (type 2) and the downsampled planar (type 1) features.
// `sparse_features`, if not null, is filled with a coarser subset (used as ICP source/target).
// Returns false if nothing could be extracted (empty cloud, or unusable point timestamps).
inline bool extractLidarFeatures(
        std::vector<Pointd>& pc
        , const FeatureExtractionParams& params
        , int64_t& median_dt
        , std::vector<Pointd>& features
        , std::vector<Pointd>* sparse_features = nullptr)
{
    features.clear();
    if(pc.size() == 0)
    {
        return false;
    }


    // Sort the points by time
    if(params.unsorted_pc)
    {
        std::sort(pc.begin(), pc.end(), [](const Pointd& a, const Pointd& b) { return a.t < b.t; });
    }

    double threshold = params.feature_voxel_size;


    // Split the point cloud into channels and remove points too close
    std::vector<std::vector<Pointd> > channels;
    if(params.is_2d)
    {
        channels.push_back(pc);
    }
    else
    {
        bool has_channel = pc.at(0).channel != kNoChannel;
        if(!has_channel)
        {
            throw std::runtime_error("extractLidarFeatures: Point cloud does not have a channel. Current implementation requires point clouds to be split by channel.");
        }
        else
        {
            channels = splitChannels(pc, params.min_range, params.max_feature_range);
        }
    }


    // Get the median time between points in each channel
    if(median_dt < 0.0)
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
        if(dt.empty())
        {
            return false;
        }
        std::sort(dt.begin(), dt.end());
        median_dt = dt[dt.size()/2];
        if(median_dt <= 0.0)
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
            if(dt.empty())
            {
                return false;
            }
            std::sort(dt.begin(), dt.end());
            median_dt = dt[dt.size()/2];

        }
        if(median_dt <= 0.0)
        {
            std::cout << "ERROR !!!!!!!!!!!!!!!!!!!!! extractLidarFeatures: Median dt is zero or negative, cannot compute features." << std::endl;
            return false;
        }
    }


    std::vector<Pointd> downsample;
    int64_t time_thr = (int64_t)(median_dt * kFeatureTimeThrFactor);
    int64_t time_thr_far = (int64_t)(median_dt * kFeatureTimeThrFarFactor);
    double min_range = kFeatureMinRangeFactor * params.min_range;

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
        downsample.push_back(last_point);
        downsample.back().type = 1; // Set the type to 1 (planar)
        for(size_t j = 1; j < channels[i].size() - 1; ++j)
        {
            if(ranges[j] < min_range || ranges[j] > params.max_feature_range)
            {
                continue; // Skip points that are too close
            }


            if((!params.planar_only) && (!params.is_2d))
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
                    if(params.intensity_threshold > 0.0 && channels[i][j].i < params.intensity_threshold)
                    {
                        continue; // Skip points that do not meet the intensity threshold
                    }
                    features.push_back(channels[i][j]);
                    features.back().type = 2; // Set the type to 2 (edge)
                    continue;
                }
                else if( (dt_2 < time_thr)
                        && (std::abs(delta_2) < threshold)
                        && ((dt_1 > time_thr_far) || (delta_1 < -local_thr)) )
                {
                    if(params.intensity_threshold > 0.0 && channels[i][j].i < params.intensity_threshold)
                    {
                        continue; // Skip points that do not meet the intensity threshold
                    }
                    features.push_back(channels[i][j]);
                    features.back().type = 2; // Set the type to 2 (edge)
                    continue;
                }
            }

            // Check the distance with the last point
            double distance = std::sqrt(
                (channels[i][j].x - last_point.x) * (channels[i][j].x - last_point.x) +
                (channels[i][j].y - last_point.y) * (channels[i][j].y - last_point.y) +
                (channels[i][j].z - last_point.z) * (channels[i][j].z - last_point.z));
            double rand_val = ((double) rand() / (RAND_MAX));
            if(distance > params.feature_voxel_size*(0.5 + rand_val))
            {
                if(params.intensity_threshold > 0.0 && channels[i][j].i < params.intensity_threshold)
                {
                    continue; // Skip points that do not meet the intensity threshold
                }
                last_point = channels[i][j];
                downsample.push_back(channels[i][j]);
                downsample.back().type = params.is_2d ? 2 : 1; // Set the type to 1 (planar) or 2 (edge) depending on is_2d
            }
        }
    }

    downsample = downsamplePointCloudSubset(downsample, params.feature_voxel_size);

    if(sparse_features != nullptr)
    {
        *sparse_features = downsamplePointCloudSubset(downsample, 2*params.feature_voxel_size);
        // Concatenate the edge features to the sparse features
        if((!params.planar_only) && (!params.is_2d))
        {
            std::vector<Pointd> edge_sparse_features = downsamplePointCloudSubset(features, params.feature_voxel_size);
            sparse_features->insert(sparse_features->end(), edge_sparse_features.begin(), edge_sparse_features.end());
        }
    }

    // Concatenate the downsampled features and the edge features
    features.insert(features.end(), downsample.begin(), downsample.end());

    return true;
}

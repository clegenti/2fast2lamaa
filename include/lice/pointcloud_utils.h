#pragma once
#include <lice/types.h>
#include "ankerl/unordered_dense.h"
#include <vector>
#include <cmath>
#include <limits>
#include <map>
#include <random>
#include "happly/happly.h"
#include "lice/utils.h"


template<typename T>
inline GridIndex getGridIndex(const PointTemplated<T>& pt, T cell_size)
{
    return GridIndex(
        static_cast<int>(std::floor(pt.x / cell_size)),
        static_cast<int>(std::floor(pt.y / cell_size)),
        static_cast<int>(std::floor(pt.z / cell_size)));
}

// Indexes of the points kept by the density filter. Split out of filterPointsDensity so that a
// parallel per-point array (the covariances) can be permuted exactly like the points: the filter both
// drops points and reorders them.
template<typename T>
inline std::vector<size_t> filterPointsDensityIndexes(const std::vector<PointTemplated<T> >& input, T cell_size)
{
    ankerl::unordered_dense::map<GridIndex, std::vector<size_t> > occupied_cells;
    occupied_cells.reserve(input.size());
    for(size_t i = 0; i < input.size(); ++i)
    {
        occupied_cells[getGridIndex(input[i], cell_size)].push_back(i);
    }
    std::vector<size_t> output;
    output.reserve(input.size());

    // If the 8 neighboring cells are all occupied, ignore the point in the cell
    for(const auto& [index, indexes] : occupied_cells)
    {
        int num_neighbors = 0;
        for(int dx = -1; dx <= 1; ++dx)
        {
            for(int dy = -1; dy <= 1; ++dy)
            {
                for(int dz = -1; dz <= 1; ++dz)
                {
                    if(dx == 0 && dy == 0 && dz == 0)
                    {
                        continue;
                    }
                    GridIndex neighbor_index(std::get<0>(index) + dx, std::get<1>(index) + dy, std::get<2>(index) + dz);
                    if(occupied_cells.find(neighbor_index) != occupied_cells.end())
                    {
                        num_neighbors++;
                    }
                }
            }
        }
        if(num_neighbors <= 12)
        {
            output.insert(output.end(), indexes.begin(), indexes.end());
        }
    }
    return output;
}

template<typename T>
inline std::vector<PointTemplated<T> > filterPointsDensity(const std::vector<PointTemplated<T> >& input, T cell_size)
{
    std::vector<size_t> indexes = filterPointsDensityIndexes(input, cell_size);
    std::vector<PointTemplated<T> > output;
    output.reserve(indexes.size());
    for(const size_t index : indexes)
    {
        output.push_back(input[index]);
    }
    return output;
}

// Same filter, keeping the per-point covariances aligned with the points that are kept
template<typename T>
inline std::pair<std::vector<PointTemplated<T> >, std::vector<Mat3> > filterPointsDensity(const std::vector<PointTemplated<T> >& input, const std::vector<Mat3>& covariances, T cell_size)
{
    std::vector<size_t> indexes = filterPointsDensityIndexes(input, cell_size);
    std::vector<PointTemplated<T> > output;
    std::vector<Mat3> output_covariances;
    output.reserve(indexes.size());
    output_covariances.reserve(indexes.size());
    for(const size_t index : indexes)
    {
        output.push_back(input[index]);
        output_covariances.push_back(covariances[index]);
    }
    return {output, output_covariances};
}


// Voxel downsampling of a point cloud, one centroid per occupied voxel.
// `covariances_in`/`covariances_out`, when given, carry a per-point position covariance through the
// downsampling. The covariance of a centroid of n independent points is sum(Sigma_i)/n^2, and the
// covariances follow the points through the point count capping. Both default to nullptr, in which
// case not a single extra operation is performed.
template<typename T>
inline std::vector<PointTemplated<T>> downsamplePointCloud(const std::vector<PointTemplated<T> >& input, T cell_size, int max_points = -1, bool quadrant_balanced = false, const std::vector<Mat3>* covariances_in = nullptr, std::vector<Mat3>* covariances_out = nullptr)
{
    const bool with_covariances = (covariances_in != nullptr) && (covariances_out != nullptr);
    std::vector<PointTemplated<T>> output;
    std::vector<Mat3> output_covariances;
    ankerl::unordered_dense::map<GridIndex, std::pair<Vec3, int>, ankerl::unordered_dense::hash<GridIndex>> grid_map;
    // Kept in its own map so that the voxel accumulator is untouched when no covariance is carried
    ankerl::unordered_dense::map<GridIndex, Mat3, ankerl::unordered_dense::hash<GridIndex>> covariance_map;
    for(size_t i = 0; i < input.size(); ++i)
    {
        const auto& pt = input[i];
        if(pt.type == kInvalidPoint)
        {
            continue;
        }
        GridIndex index = getGridIndex(pt, cell_size);
        if(grid_map.find(index) == grid_map.end())
        {
            grid_map[index] = std::make_pair(pt.vec3d(), 1);
            if(with_covariances)
            {
                covariance_map[index] = (*covariances_in)[i];
            }
        }
        else
        {
            auto& [sum, count] = grid_map[index];
            sum += pt.vec3d();
            count++;
            if(with_covariances)
            {
                covariance_map[index] += (*covariances_in)[i];
            }
        }
    }
    output.reserve(grid_map.size());
    output_covariances.reserve(with_covariances ? grid_map.size() : 0);
    for(const auto& [index, pair] : grid_map)
    {
        const auto& [centroid, count] = pair;
        output.push_back(Pointd(centroid / static_cast<T>(count),0));
        if(with_covariances)
        {
            output_covariances.push_back(covariance_map[index] / (static_cast<double>(count)*static_cast<double>(count)));
        }
    }

    if(max_points > 0 && output.size() > static_cast<size_t>(max_points))
    {
        // The points are capped through their indexes, so that the covariances can be permuted the
        // very same way
        std::vector<size_t> kept_indexes;
        if(quadrant_balanced)
        {
            std::vector<std::vector<size_t>> quadrants(4);
            for(size_t i = 0; i < output.size(); ++i)
            {
                Vec3 pt = output[i].vec3();
                if(pt[0] >= 0 && pt[1] >= 0)
                {
                    quadrants[0].push_back(i);
                }
                else if(pt[0] < 0 && pt[1] >= 0)
                {
                    quadrants[1].push_back(i);
                }
                else if(pt[0] < 0 && pt[1] < 0)
                {
                    quadrants[2].push_back(i);
                }
                else
                {
                    quadrants[3].push_back(i);
                }
            }
            // Sort the quadrants by size
            std::sort(quadrants.begin(), quadrants.end(), [](const std::vector<size_t>& a, const std::vector<size_t>& b) {
                return a.size() < b.size();
            });
            for(int i = 0; i < 4; ++i)
            {
                int num_pts_available = (max_points - kept_indexes.size()) / (4 - i);
                if(quadrants[i].size() <= static_cast<size_t>(num_pts_available))
                {
                    kept_indexes.insert(kept_indexes.end(), quadrants[i].begin(), quadrants[i].end());
                }
                else
                {
                    std::vector<int> indexes = generateRandomIndexes(0, quadrants[i].size(), num_pts_available);
                    for(int idx: indexes)
                    {
                        kept_indexes.push_back(quadrants[i][idx]);
                    }
                }
            }
        }
        else
        {
            std::vector<int> indexes = generateRandomIndexes(0, output.size(), max_points);
            for(int idx: indexes)
            {
                kept_indexes.push_back(idx);
            }
        }

        std::vector<PointTemplated<T>> temp_pts;
        std::vector<Mat3> temp_covariances;
        temp_pts.reserve(kept_indexes.size());
        temp_covariances.reserve(with_covariances ? kept_indexes.size() : 0);
        for(const size_t index : kept_indexes)
        {
            temp_pts.push_back(output[index]);
            if(with_covariances)
            {
                temp_covariances.push_back(output_covariances[index]);
            }
        }
        output = temp_pts;
        output_covariances = temp_covariances;
    }

    if(with_covariances)
    {
        *covariances_out = output_covariances;
    }

    return output;
}

// Same as above, downsampling each point type separately so that the rarer types are not swallowed by
// the dominant one. The optional per-point covariances are carried through, as in downsamplePointCloud.
template<typename T>
inline std::vector<PointTemplated<T>> downsamplePointCloudPerType(
    const std::vector<PointTemplated<T>>& input, double cell_size, int max_points = -1, const std::vector<Mat3>* covariances_in = nullptr, std::vector<Mat3>* covariances_out = nullptr)
{
    const bool with_covariances = (covariances_in != nullptr) && (covariances_out != nullptr);
    std::map<int, std::vector<PointTemplated<T>>> type_to_points;
    std::map<int, std::vector<Mat3> > type_to_covariances;
    for(size_t i = 0; i < input.size(); ++i)
    {
        const auto& pt = input[i];
        if(type_to_points.find(pt.type) == type_to_points.end())
        {
            type_to_points[pt.type] = std::vector<PointTemplated<T>>();
            type_to_points[pt.type].reserve(input.size());
        }
        type_to_points[pt.type].push_back(pt);
        if(with_covariances)
        {
            type_to_covariances[pt.type].push_back((*covariances_in)[i]);
        }
    }
    std::vector<PointTemplated<T>> output;
    std::vector<Mat3> output_covariances;
    for(const auto& [type, pts] : type_to_points)
    {
        std::vector<Mat3> downsampled_covariances;
        std::vector<PointTemplated<T>> downsampled_pts = downsamplePointCloud<T>(pts, cell_size, max_points*((double)pts.size())/input.size(), false,
                with_covariances ? &type_to_covariances[type] : nullptr,
                with_covariances ? &downsampled_covariances : nullptr);
        for(auto& pt : downsampled_pts)
        {
            pt.type = type;
        }
        output.insert(output.end(), downsampled_pts.begin(), downsampled_pts.end());
        output_covariances.insert(output_covariances.end(), downsampled_covariances.begin(), downsampled_covariances.end());
    }
    if(with_covariances)
    {
        *covariances_out = output_covariances;
    }
    return output;
}



// Random generator for the point selection, in place of rand(), which takes a global lock on
// every call. One per thread; fixed seed, as rand() was never seeded, so runs stay repeatable.
inline std::mt19937_64& pointSelectionRng()
{
    thread_local std::mt19937_64 rng(1);
    return rng;
}

// Keeps one random point per occupied cell. Cells get a slot in order of first appearance, so each
// point needs a single lookup and no per-cell point list is stored: the rank of the kept point in
// its cell is drawn once the cell counts are known, and a second pass picks it.
template<typename T>
inline std::vector<PointTemplated<T> > downsamplePointCloudSubset(
    const std::vector<PointTemplated<T> >& input, T cell_size)
{
    ankerl::unordered_dense::map<GridIndex, uint32_t> slots;
    slots.reserve(input.size());
    std::vector<uint32_t> slot_of_point(input.size());
    std::vector<uint32_t> count;
    for(size_t i = 0; i < input.size(); ++i)
    {
        const auto [it, inserted] = slots.try_emplace(getGridIndex(input[i], cell_size), (uint32_t)count.size());
        if(inserted)
        {
            count.push_back(0);
        }
        slot_of_point[i] = it->second;
        count[it->second]++;
    }

    std::vector<uint32_t> rank(count.size(), 0);
    for(size_t s = 0; s < count.size(); ++s)
    {
        if(count[s] > 1)
        {
            rank[s] = std::uniform_int_distribution<uint32_t>(0, count[s] - 1)(pointSelectionRng());
        }
        count[s] = 0;
    }

    std::vector<uint32_t> kept(rank.size());
    for(size_t i = 0; i < input.size(); ++i)
    {
        const uint32_t s = slot_of_point[i];
        if(count[s]++ == rank[s])
        {
            kept[s] = (uint32_t)i;
        }
    }

    std::vector<PointTemplated<T> > output;
    output.reserve(kept.size());
    for(const uint32_t i : kept)
    {
        output.push_back(input[i]);
    }
    return output;
}





namespace detail
{
// Whether a point is kept by `splitChannels`. Written as the negation of the out-of-range test so
// that a NaN point, whose range compares false against both bounds, is kept as it always was.
inline bool inChannelRange(const Pointd& p, const double min_dist, const double max_dist)
{
    const double range = p.vec3().norm();
    return !(range < min_dist || range > max_dist);
}

// `splitChannels` for channel numbers too large or negative for a flat table: same passes, with
// a map. Slower, but it makes no assumption on the numbering.
inline std::vector<std::vector<Pointd> > splitChannelsMap(const std::vector<Pointd>& pc, double min_dist, double max_dist)
{
    std::map<int,int> channel_ids;
    int counter = 0;
    for(size_t i = 0; i < pc.size(); ++i)
    {
        if(channel_ids.find(pc[i].channel) != channel_ids.end() || !inChannelRange(pc[i], min_dist, max_dist))
        {
            continue;
        }
        channel_ids[pc[i].channel] = counter++;
    }

    std::vector<std::vector<Pointd> > output(counter);
    for(int i = 0; i < counter; ++i)
    {
        output[i].reserve(pc.size()/counter);
    }
    for(size_t i = 0; i < pc.size(); ++i)
    {
        if(inChannelRange(pc[i], min_dist, max_dist))
        {
            output[channel_ids.at(pc[i].channel)].push_back(pc[i]);
        }
    }
    return output;
}
} // namespace detail

// Split a point cloud into one vector per channel, keeping only the points whose range is within
// [min_dist, max_dist]. The channels come out in the order of their first kept point, and the
// points of a channel in their input order: the feature extraction concatenates its output in
// that order and downsamples it with random draws, so the order is part of the result.
//
// Both passes apply the same range test, so a point is kept iff it is in range, and its channel
// has necessarily been given an id by the time the second pass reaches it. (The second pass used
// to skip the test when min_dist <= 0: points beyond max_dist were then kept, and those of a
// channel with no point in range were appended to channel 0, mixing two scanlines.)
inline std::vector<std::vector<Pointd> > splitChannels(const std::vector<Pointd>& pc, double min_dist = 0.0, double max_dist = 1000.0)
{
    // Channel numbers are small non-negative integers on the lidars we handle, so they index a
    // flat table; this was a std::map, looked up twice per point, and 2.4 to 2.8 times slower.
    constexpr int kMaxTableChannel = 1 << 16;
    std::vector<int> channel_ids;
    int counter = 0;
    for(size_t i = 0; i < pc.size(); ++i)
    {
        const int channel = pc[i].channel;
        if(channel < 0 || channel >= kMaxTableChannel)
        {
            return detail::splitChannelsMap(pc, min_dist, max_dist);
        }
        if(channel >= (int)channel_ids.size())
        {
            channel_ids.resize(channel + 1, -1);
        }
        if(channel_ids[channel] >= 0 || !detail::inChannelRange(pc[i], min_dist, max_dist))
        {
            continue;
        }
        channel_ids[channel] = counter++;
    }

    std::vector<std::vector<Pointd> > output(counter);
    for(int i = 0; i < counter; ++i)
    {
        output[i].reserve(pc.size()/counter);
    }
    for(size_t i = 0; i < pc.size(); ++i)
    {
        if(detail::inChannelRange(pc[i], min_dist, max_dist))
        {
            output[channel_ids[pc[i].channel]].push_back(pc[i]);
        }
    }
    return output;
}

// Median of the point ranges, in the frame the points come in. Sampled rather than exhaustive: the
// median of a few thousand points of a scan is the same number for the purpose it serves, and this
// runs on every scan. Returns -1.0 when the cloud holds no usable point.
template <typename T>
inline double getMedianRange(const std::vector<PointTemplated<T> >& pc, const size_t max_samples = 4096)
{
    if(pc.empty())
    {
        return -1.0;
    }
    const size_t step = std::max<size_t>(1, pc.size()/max_samples);
    std::vector<double> ranges;
    ranges.reserve((pc.size()/step) + 1);
    for(size_t i = 0; i < pc.size(); i += step)
    {
        const auto& pt = pc[i];
        if(pt.type == kInvalidPoint)
        {
            continue;
        }
        if(!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z))
        {
            continue;
        }
        ranges.push_back(pt.vec3d().norm());
    }
    if(ranges.empty())
    {
        return -1.0;
    }
    const size_t mid = ranges.size()/2;
    std::nth_element(ranges.begin(), ranges.begin() + mid, ranges.end());
    return ranges[mid];
}

// Get median time between points in a point cloud channel
inline int64_t getMedianDt(const std::vector<Pointd>& pc)
{
    std::vector<int64_t> dt;
    for(size_t i = 1; i < pc.size(); ++i)
    {
        dt.push_back(pc[i].t - pc[i-1].t);
    }
    std::sort(dt.begin(), dt.end());
    return dt[dt.size()/2];
}

inline int64_t getMeanDt(const std::vector<Pointd>& pc)
{
    std::vector<int64_t> dt;
    for(size_t i = 1; i < pc.size(); ++i)
    {
        dt.push_back(pc[i].t - pc[i-1].t);
    }
    int64_t sum = std::accumulate(dt.begin(), dt.end(), 0);
    return sum/dt.size();
}


inline void savePointCloudToPly(const std::string& filename, const std::vector<Pointd>& pc, const std::array<uint8_t,3>& default_color = {255,255,255})
{
    happly::PLYData ply_data;
    std::vector<std::array<double, 3> > vertices;
    for(const auto& pt : pc)
    {
        vertices.push_back({pt.x, pt.y, pt.z});
    }
    ply_data.addVertexPositions(vertices);
    if(default_color != std::array<uint8_t,3>({255,255,255}))
    {
        ply_data.addVertexColors(std::vector<std::array<uint8_t, 3> >(pc.size(), default_color));
    }
    ply_data.write(filename, happly::DataFormat::Binary);
}


inline std::vector<Pointd> loadPointCloudFromPly(const std::string& filename)
{
    happly::PLYData ply_data(filename);
    std::vector<std::array<double, 3> > vertices = ply_data.getVertexPositions();
    std::vector<Pointd> pc;
    pc.reserve(vertices.size());
    for(const auto& v : vertices)
    {
        pc.emplace_back(v[0], v[1], v[2], 0);
    }
    return pc;
}
#pragma once
#include <lice/types.h>
#include "ankerl/unordered_dense.h"
#include <vector>
#include <cmath>
#include <limits>
#include <map>
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

template<typename T>
inline std::vector<PointTemplated<T> > filterPointsDensity(const std::vector<PointTemplated<T> >& input, T cell_size)
{
    ankerl::unordered_dense::map<GridIndex, std::vector<PointTemplated<T>>> occupied_cells;
    occupied_cells.reserve(input.size());
    for(const auto& pt : input)
    {
        GridIndex index = getGridIndex(pt, cell_size);
        if(occupied_cells.find(index) == occupied_cells.end())
        {
            occupied_cells[index] = std::vector<PointTemplated<T>>{pt};
        }
        else
        {
            occupied_cells[index].push_back(pt);
        }
    }
    std::vector<PointTemplated<T>> output;

    // If the 8 neighboring cells are all occupied, ignore the point in the cell
    for(const auto& [index, pts] : occupied_cells)
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
            output.insert(output.end(), pts.begin(), pts.end());
        }
    }
    return output;
}


template<typename T>
inline std::vector<PointTemplated<T>> downsamplePointCloud(const std::vector<PointTemplated<T> >& input, T cell_size, int max_points = -1, bool quadrant_balanced = false)
{
    std::vector<PointTemplated<T>> output;
    ankerl::unordered_dense::map<GridIndex, std::pair<Vec3, int>, ankerl::unordered_dense::hash<GridIndex>> grid_map;
    for(const auto& pt : input)
    {
        if(pt.type == kInvalidPoint)
        {
            continue;
        }
        GridIndex index = getGridIndex(pt, cell_size);
        if(grid_map.find(index) == grid_map.end())
        {
            grid_map[index] = std::make_pair(pt.vec3d(), 1);
        }
        else
        {
            auto& [sum, count] = grid_map[index];
            sum += pt.vec3d();
            count++;
        }
    }
    output.reserve(grid_map.size());
    for(const auto& [index, pair] : grid_map)
    {
        const auto& [centroid, count] = pair;
        output.push_back(Pointd(centroid / static_cast<T>(count),0));
    }

    if(max_points > 0 && output.size() > static_cast<size_t>(max_points))
    {
        
        if(quadrant_balanced)
        {
            std::vector<std::vector<Pointd>> quadrants(4);
            for(const auto& ptd : output)
            {
                Vec3 pt = ptd.vec3();
                if(pt[0] >= 0 && pt[1] >= 0)
                {
                    quadrants[0].push_back(ptd);
                }
                else if(pt[0] < 0 && pt[1] >= 0)
                {
                    quadrants[1].push_back(ptd);
                }
                else if(pt[0] < 0 && pt[1] < 0)
                {
                    quadrants[2].push_back(ptd);
                }
                else
                {
                    quadrants[3].push_back(ptd);
                }
            }
            // Sort the quadrants by size
            std::sort(quadrants.begin(), quadrants.end(), [](const std::vector<Pointd>& a, const std::vector<Pointd>& b) {
                return a.size() < b.size();
            });
            std::vector<Pointd> temp_pts;
            for(int i = 0; i < 4; ++i)
            {
                int num_pts_available = (max_points - temp_pts.size()) / (4 - i);
                if(quadrants[i].size() <= static_cast<size_t>(num_pts_available))
                {
                    temp_pts.insert(temp_pts.end(), quadrants[i].begin(), quadrants[i].end());
                }
                else
                {
                    std::vector<int> indexes = generateRandomIndexes(0, quadrants[i].size(), num_pts_available);
                    for(int idx: indexes)
                    {
                        temp_pts.push_back(quadrants[i][idx]);
                    }
                }
            }
            output = temp_pts;
        }
        else
        {
            std::vector<int> indexes = generateRandomIndexes(0, output.size(), max_points);
            std::vector<Pointd> temp_pts;
            for(int idx: indexes)
            {
                temp_pts.push_back(output[idx]);
            }
            output = temp_pts;
        }
    }



    return output;
}

template<typename T>
inline std::vector<PointTemplated<T>> downsamplePointCloudPerType(
    const std::vector<PointTemplated<T>>& input, double cell_size, int max_points = -1)
{
    std::map<int, std::vector<PointTemplated<T>>> type_to_points;
    for(const auto& pt : input)
    {
        if(type_to_points.find(pt.type) == type_to_points.end())
        {
            type_to_points[pt.type] = std::vector<PointTemplated<T>>();
            type_to_points[pt.type].reserve(input.size());
        }
        type_to_points[pt.type].push_back(pt);
    }
    std::vector<PointTemplated<T>> output;
    for(const auto& [type, pts] : type_to_points)
    {
        std::vector<PointTemplated<T>> downsampled_pts = downsamplePointCloud<T>(pts, cell_size, max_points*((double)pts.size())/input.size());
        for(auto& pt : downsampled_pts)
        {
            pt.type = type;
        }
        output.insert(output.end(), downsampled_pts.begin(), downsampled_pts.end());
    }
    return output;
}



template<typename T>
inline std::vector<PointTemplated<T> > downsamplePointCloudSubset(
    const std::vector<PointTemplated<T> >& input, T cell_size)
{
    std::vector<PointTemplated<T> > output;
    std::unordered_map<GridIndex, std::vector<PointTemplated<T> >, ankerl::unordered_dense::hash<GridIndex>> grid_map;
    for(const auto& pt : input)
    {
        GridIndex index = getGridIndex(pt, cell_size);
        if(grid_map.find(index) == grid_map.end())
        {
            grid_map[index] = std::vector<PointTemplated<T> >(1, pt);
        }
        else
        {
            grid_map[index].push_back(pt);
        }
    }
    output.reserve(grid_map.size());
    for(const auto& [index, pts] : grid_map)
    {
        if(pts.size() > 1)
        {
            // Get a random point from the cluster
            int random_index = rand() % pts.size();
            output.push_back(pts[random_index]);
        }
        else
        {
            output.push_back(pts[0]);
        }
    }
    return output;
}





inline std::vector<std::vector<Pointd> > splitChannels(const std::vector<Pointd>& pc, double min_dist = 0.0, double max_dist = 1000.0)
{
    std::vector<double> ranges;
    std::map<int,int> channel_ids;
    int counter = 0;
    for(size_t i = 0; i < pc.size(); ++i)
    {
        if(channel_ids.find(pc[i].channel) != channel_ids.end())
        {
            continue;
        }

        double range = pc[i].vec3().norm();
        if(range < min_dist || range > max_dist)
        {
            continue;
        }
        channel_ids[pc[i].channel] = counter;
        counter++;
    }

    std::vector<std::vector<Pointd> > output(counter);
    for(int i = 0; i < counter; ++i)
    {
        output[i].reserve(pc.size()/counter);
    }
    for(size_t i = 0; i < pc.size(); ++i)
    {
        if(min_dist > 0.0)
        {
            double range = pc[i].vec3().norm();
            if(range < min_dist || range > max_dist)
                continue;
        }
        output[channel_ids[pc[i].channel]].push_back(pc[i]);
    }
    return output;
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
#pragma once

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "geometry_msgs/msg/transform.hpp"

#include "lice/types.h"
#include <Eigen/Dense>
#include <cmath>
#include <cstring>
#include <set>
#include <type_traits>



//////// Beginning helper functions to read parameters from the node
template<class T>
inline void printOption(rclcpp::Node* n, bool user_defined, std::string field, T value)
{
    std::stringstream stream;
    if(user_defined)
    {
        stream << "[Param] User defined value for " << field << " is " << value;
    }
    else
    {
        stream << "[Param] Default value for " << field << " is " << value;
    }
    RCLCPP_INFO(n->get_logger(), "%s",stream.str().c_str());
}

inline void printOptionError(rclcpp::Node* n, std::string field)
{
    std::stringstream stream;
    stream << "[Param] It seems that the parameter " << field << " is not provided";
    RCLCPP_ERROR(n->get_logger(), "%s",stream.str().c_str());
    throw std::invalid_argument("Invalid parameter");
}

template<class T>
inline T lowLevelReadField(rclcpp::Node* n, std::string field, bool required, T default_value =T())
{
    T output;
    if (n->get_parameter(field, output))
    {
        printOption(n, true, field, output);
    }
    else
    {
        if (required)
        {
            printOptionError(n, field);
        }
        else
        {
            output = default_value;
            printOption(n, false, field, output);
        }

    }
    return output;
}


inline double readRequiredFieldDouble(rclcpp::Node* n, std::string field)
{
    n->declare_parameter(field, rclcpp::PARAMETER_DOUBLE);
    return lowLevelReadField<double>(n, field, true);
}
inline double readFieldDouble(rclcpp::Node* n, std::string field, double default_value)
{
    n->declare_parameter(field, rclcpp::PARAMETER_DOUBLE);
    return lowLevelReadField<double>(n, field, false, default_value);
}

inline int readRequiredFieldInt(rclcpp::Node* n, std::string field)
{
    n->declare_parameter(field, rclcpp::PARAMETER_INTEGER);
    return lowLevelReadField<int>(n, field, true);
}
inline int readFieldInt(rclcpp::Node* n, std::string field, int default_value)
{
    n->declare_parameter(field, rclcpp::PARAMETER_INTEGER);
    return lowLevelReadField<int>(n, field, false, default_value);
}

inline std::string readRequiredFieldString(rclcpp::Node* n, std::string field)
{
    n->declare_parameter(field, rclcpp::PARAMETER_STRING);
    return lowLevelReadField<std::string>(n, field, true);
}
inline std::string readFieldString(rclcpp::Node* n, std::string field, std::string default_value)
{
    n->declare_parameter(field, rclcpp::PARAMETER_STRING);
    return lowLevelReadField<std::string>(n, field, false, default_value);
}

inline bool readRequiredFieldBool(rclcpp::Node* n, std::string field)
{
    n->declare_parameter(field, rclcpp::PARAMETER_BOOL);
    return lowLevelReadField<bool>(n, field, true);
}
inline bool readFieldBool(rclcpp::Node* n, std::string field, bool default_value)
{
    n->declare_parameter(field, rclcpp::PARAMETER_BOOL);
    return lowLevelReadField<bool>(n, field, false, default_value);
}
/////// End helper functions to read parameters from the node



/////// Beginning helper functions to subscribe and publish PointCloud2 messages
enum PointFieldTypes
{
    TIME = 0,
    INTENSITY = 1,
    CHANNEL = 2,
    TYPE = 3,
    X = 4,
    Y = 5,
    Z = 6,
    RGB = 7,
    // The 6 unique entries of the symmetric 3x3 position covariance of the point. They are kept
    // contiguous and in that order: the covariance reader below addresses them as COV_XX + k
    COV_XX = 8,
    COV_XY = 9,
    COV_XZ = 10,
    COV_YY = 11,
    COV_YZ = 12,
    COV_ZZ = 13,
    NUM_COVARIANCE_TYPES = 6,
    NUM_TYPES = 14
};

// Variance given to a point whose covariance is missing or unusable. Large enough (1e3 m of standard
// deviation) for the point to weigh next to nothing in a covariance-weighted registration, while
// keeping its geometry in the map.
const double kInvalidPointVariance = 1e6;

inline std::vector<std::pair<int, int>> getPointFields(const std::vector<sensor_msgs::msg::PointField>& fields, bool need_time=false)
{
    std::vector<std::pair<int,int> > output(PointFieldTypes::NUM_TYPES, {-1, -1});
    for(size_t i = 0; i < fields.size(); ++i)
    {
        if((fields[i].name == "time")||(fields[i].name == "point_time_offset")||(fields[i].name == "ts")||(fields[i].name == "t")||(fields[i].name == "timestamp"))
        {
            output[PointFieldTypes::TIME] = {fields[i].offset , fields[i].datatype};
        }
        else if((fields[i].name == "channel")||(fields[i].name == "ring"))
        {
            output[PointFieldTypes::CHANNEL] = {fields[i].offset , fields[i].datatype};
        }
        else if(fields[i].name == "intensity")
        {
            output[PointFieldTypes::INTENSITY] = {fields[i].offset , fields[i].datatype};
        }
        else if(fields[i].name == "type")
        {
            output[PointFieldTypes::TYPE] = {fields[i].offset , fields[i].datatype};
        }
        else if(fields[i].name == "x")
        {
            output[PointFieldTypes::X] = {fields[i].offset , fields[i].datatype};
        }
        else if(fields[i].name == "y")
        {
            output[PointFieldTypes::Y] = {fields[i].offset , fields[i].datatype};
        }
        else if(fields[i].name == "z")
        {
            output[PointFieldTypes::Z] = {fields[i].offset , fields[i].datatype};
        }
        else if(fields[i].name == "rgb")
        {
            output[PointFieldTypes::RGB] = {fields[i].offset , fields[i].datatype};
        }
        else if(fields[i].name == "cov_xx")
        {
            output[PointFieldTypes::COV_XX] = {fields[i].offset , fields[i].datatype};
        }
        else if(fields[i].name == "cov_xy")
        {
            output[PointFieldTypes::COV_XY] = {fields[i].offset , fields[i].datatype};
        }
        else if(fields[i].name == "cov_xz")
        {
            output[PointFieldTypes::COV_XZ] = {fields[i].offset , fields[i].datatype};
        }
        else if(fields[i].name == "cov_yy")
        {
            output[PointFieldTypes::COV_YY] = {fields[i].offset , fields[i].datatype};
        }
        else if(fields[i].name == "cov_yz")
        {
            output[PointFieldTypes::COV_YZ] = {fields[i].offset , fields[i].datatype};
        }
        else if(fields[i].name == "cov_zz")
        {
            output[PointFieldTypes::COV_ZZ] = {fields[i].offset , fields[i].datatype};
        }
    }
    if(need_time&&(output[PointFieldTypes::TIME].first == -1))
    {
        std::cout << "The point cloud does not seem to contain timestamp information (field 'time', 'ts', or 'point_time_offset'" << std::endl;
    }
    if((output[PointFieldTypes::X].first == -1)||(output[PointFieldTypes::Y].first == -1)||(output[PointFieldTypes::Z].first == -1))
    {
        std::cout << "The point cloud seems to miss at least one component (x, y, or z)" << std::endl;
    }
    return output;
}

// Non-finite coordinates do happen in the wild, most lidar drivers use them for the points without
// return. They are filtered when adding points to the map, but nothing protects the registration,
// which would hand them over to the solver as nan residuals and jacobians. They are therefore
// dropped as soon as the message is read.
inline bool isFinitePoint(const float x, const float y, const float z)
{
    return std::isfinite(x) && std::isfinite(y) && std::isfinite(z);
}

// Report the points dropped by the barrier above. Only the first occurrence is printed: a cloud with
// no-return points has them in every single scan, and the message would drown everything else.
inline void reportNonFinitePoints(const size_t num_dropped, const size_t num_points)
{
    static bool already_reported = false;
    if((num_dropped == 0) || already_reported)
    {
        return;
    }
    already_reported = true;
    std::cout << "Dropped " << num_dropped << " of the " << num_points << " points of the point cloud"
              << " (non-finite coordinates). Only reported once." << std::endl;
}

// True when the cloud carries the 6 entries of the per-point position covariance
inline bool hasPointCovariance(const std::vector<std::pair<int, int> >& fields)
{
    for(int k = 0; k < PointFieldTypes::NUM_COVARIANCE_TYPES; ++k)
    {
        if(fields[PointFieldTypes::COV_XX + k].first == -1)
        {
            return false;
        }
    }
    return true;
}

// Read one float32 or float64 field of a point as a double
inline double readPointFieldAsDouble(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg, const size_t point_index, const std::pair<int, int>& field)
{
    if(field.second == sensor_msgs::msg::PointField::FLOAT64)
    {
        double value;
        memcpy(&value, &(msg->data[(msg->point_step*point_index) + field.first]), sizeof(double));
        return value;
    }
    float value;
    memcpy(&value, &(msg->data[(msg->point_step*point_index) + field.first]), sizeof(float));
    return (double)value;
}

// Rebuild the symmetric position covariance of one point from its 6 unique entries. A non-finite or
// obviously invalid covariance (a negative variance) is replaced by a very large isotropic one: the
// point keeps its geometry, and a covariance-weighted registration gives it no say.
inline Mat3 readPointCovariance(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg, const size_t point_index, const std::vector<std::pair<int, int> >& fields)
{
    double entries[PointFieldTypes::NUM_COVARIANCE_TYPES];
    for(int k = 0; k < PointFieldTypes::NUM_COVARIANCE_TYPES; ++k)
    {
        entries[k] = readPointFieldAsDouble(msg, point_index, fields[PointFieldTypes::COV_XX + k]);
    }
    Mat3 covariance;
    covariance << entries[0], entries[1], entries[2],
                  entries[1], entries[3], entries[4],
                  entries[2], entries[4], entries[5];
    if(!covariance.allFinite() || (covariance.diagonal().minCoeff() < 0.0))
    {
        return Mat3::Identity()*kInvalidPointVariance;
    }
    return covariance;
}

template <typename T>
inline void preparePointCloud2Msg(sensor_msgs::msg::PointCloud2& output, const std::vector<PointTemplated<T>>& pts, const std::string& frame_id, const rclcpp::Time& time)
{
    output.header.frame_id = frame_id;
    output.header.stamp = time;
    output.width  = pts.size();
    output.height = 1;
    output.is_bigendian = false;
    output.point_step = 28;
    if(pts.size() == 0)
    {
        return;
    }
    if(pts[0].has_color)
    {
        output.point_step += 4;
    }
    output.row_step = output.point_step * output.width;
    output.fields.resize(7);

    output.fields[0].name = "x";
    output.fields[0].count =1;
    output.fields[0].offset = 0;
    output.fields[0].datatype = sensor_msgs::msg::PointField::FLOAT32;
    output.fields[1].name = "y";
    output.fields[1].count =1;
    output.fields[1].offset = 4;
    output.fields[1].datatype = sensor_msgs::msg::PointField::FLOAT32;
    output.fields[2].name = "z";
    output.fields[2].count =1;
    output.fields[2].offset = 8;
    output.fields[2].datatype = sensor_msgs::msg::PointField::FLOAT32;
    output.fields[3].name = "intensity";
    output.fields[3].count =1;
    output.fields[3].offset = 12;
    output.fields[3].datatype = sensor_msgs::msg::PointField::FLOAT32;
    output.fields[4].name = "t";
    output.fields[4].count =1;
    output.fields[4].offset = 16;
    output.fields[4].datatype = sensor_msgs::msg::PointField::UINT32;
    output.fields[5].name = "channel";
    output.fields[5].count =1;
    output.fields[5].offset = 20;
    output.fields[5].datatype = sensor_msgs::msg::PointField::INT32;
    output.fields[6].name = "type";
    output.fields[6].count =1;
    output.fields[6].offset = 24;
    output.fields[6].datatype = sensor_msgs::msg::PointField::INT32;
    if(pts[0].has_color)
    {
        output.fields.resize(8);
        output.fields[7].name = "rgb";
        output.fields[7].count =1;
        output.fields[7].offset = 28;
        output.fields[7].datatype = sensor_msgs::msg::PointField::FLOAT32;
    }

    output.row_step = output.point_step * output.width;
    output.data.resize(output.point_step*pts.size());
}

inline sensor_msgs::msg::PointCloud2 ptsVecToPointCloud2MsgInternal(const std::vector<Pointd>& pts, const std::string& frame_id, const rclcpp::Time& time)
{
    sensor_msgs::msg::PointCloud2 output;
    int64_t t_offset = time.nanoseconds();
    preparePointCloud2Msg(output, pts, frame_id, time);
    for(size_t i = 0; i < pts.size(); ++i)
    {
        float x = (float)pts[i].x;
        float y = (float)pts[i].y;
        float z = (float)pts[i].z;
        uint32_t t = static_cast<uint32_t>(pts[i].nanos() - t_offset);
        memcpy(&(output.data[(output.point_step*i) ]), &(x), sizeof(float));
        memcpy(&(output.data[(output.point_step*i) + 4]), &(y), sizeof(float));
        memcpy(&(output.data[(output.point_step*i) + 8]), &(z), sizeof(float));
        memcpy(&(output.data[(output.point_step*i) + 12]), &(pts[i].i), sizeof(float));
        memcpy(&(output.data[(output.point_step*i) + 16]), &(t), sizeof(uint32_t));
        memcpy(&(output.data[(output.point_step*i) + 20]), &(pts[i].channel), sizeof(int));
        memcpy(&(output.data[(output.point_step*i) + 24]), &(pts[i].type), sizeof(int));
        if(pts[i].has_color)
        {
            memcpy(&(output.data[(output.point_step*i) + 30]), &(pts[i].r), sizeof(unsigned char));
            memcpy(&(output.data[(output.point_step*i) + 29]), &(pts[i].g), sizeof(unsigned char));
            memcpy(&(output.data[(output.point_step*i) + 28]), &(pts[i].b), sizeof(unsigned char));
        }
    }
    return output;
}

//inline sensor_msgs::msg::PointCloud2 ptsVecToPointCloud2MsgInternal(const std::vector<Pointd>& pts, const std_msgs::msg::Header& header)
//{
//    return ptsVecToPointCloud2MsgInternal(pts, header.frame_id, rclcpp::Time(header.stamp));
//}

inline sensor_msgs::msg::PointCloud2 ptsVecToPointCloud2MsgInternal(const std::vector<Pointf>& pts, const std::string& frame_id, const rclcpp::Time& time)
{
    sensor_msgs::msg::PointCloud2 output;
    int64_t t_offset = time.nanoseconds();
    preparePointCloud2Msg(output, pts, frame_id, time);
    for(size_t i = 0; i < pts.size(); ++i)
    {
        memcpy(&(output.data[(output.point_step*i) ]), &(pts[i].x), sizeof(float));
        memcpy(&(output.data[(output.point_step*i) + 4]), &(pts[i].y), sizeof(float));
        memcpy(&(output.data[(output.point_step*i) + 8]), &(pts[i].z), sizeof(float));
        memcpy(&(output.data[(output.point_step*i) + 12]), &(pts[i].i), sizeof(float));
        uint32_t t = static_cast<uint32_t>(pts[i].nanos() - t_offset);
        memcpy(&(output.data[(output.point_step*i) + 16]), &(t), sizeof(uint32_t));
        memcpy(&(output.data[(output.point_step*i) + 20]), &(pts[i].channel), sizeof(int));
        memcpy(&(output.data[(output.point_step*i) + 24]), &(pts[i].type), sizeof(int));
        if(pts[i].has_color)
        {
            memcpy(&(output.data[(output.point_step*i) + 30]), &(pts[i].r), sizeof(unsigned char));
            memcpy(&(output.data[(output.point_step*i) + 29]), &(pts[i].g), sizeof(unsigned char));
            memcpy(&(output.data[(output.point_step*i) + 28]), &(pts[i].b), sizeof(unsigned char));
        }
    }
    return output;
}




inline std::pair<std::vector<Pointd>, bool> pointCloud2MsgToPtsVecInternal(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg)
{
    size_t num_points = msg->width*msg->height;
    std::vector<Pointd> output;
    output.resize(num_points);
    // The colour is the 8th field of the internal layout, but it is not the only thing that can be
    // appended to it (the per-point covariance is 6 more fields), so the field count alone would read
    // the first of those as a colour. The name is checked as well, and the count is kept so that no
    // existing producer changes behaviour.
    bool has_color = (msg->fields.size() > 8) && (getPointFields(msg->fields)[PointFieldTypes::RGB].first != -1);
    int64_t time_offset = rclcpp::Time(msg->header.stamp).nanoseconds();
    bool is_2d = true;
    // The non-finite points are dropped, so the points are written at their own index and the output
    // is shrunk to the number of points that made it through
    size_t num_valid = 0;
    for(size_t i=0; i < num_points; ++i)
    {
        float temp_x, temp_y, temp_z;
        memcpy(&(temp_x), &(msg->data[(msg->point_step*i) + 0]), sizeof(float));
        memcpy(&(temp_y), &(msg->data[(msg->point_step*i) + 4]), sizeof(float));
        memcpy(&(temp_z), &(msg->data[(msg->point_step*i) + 8]), sizeof(float));
        if(!isFinitePoint(temp_x, temp_y, temp_z))
        {
            continue;
        }
        Pointd& pt = output[num_valid];
        num_valid++;
        pt.x = (double)temp_x;
        pt.y = (double)temp_y;
        pt.z = (double)temp_z;
        if(temp_z != 0.0f)
        {
            is_2d = false;
        }
        uint32_t t;
        memcpy(&(pt.i), &(msg->data[(msg->point_step*i) + 12]), sizeof(float));
        memcpy(&t, &(msg->data[(msg->point_step*i) + 16]), sizeof(uint32_t));
        pt.t = (int64_t)(t) + time_offset;
        memcpy(&(pt.channel), &(msg->data[(msg->point_step*i) + 20]), sizeof(int));
        memcpy(&(pt.type), &(msg->data[(msg->point_step*i) + 24]), sizeof(int));
        if(has_color)
        {
            pt.r = msg->data[(msg->point_step*i) + 30];
            pt.g = msg->data[(msg->point_step*i) + 29];
            pt.b = msg->data[(msg->point_step*i) + 28];
            pt.has_color = true;
        }
    }
    reportNonFinitePoints(num_points - num_valid, num_points);
    output.resize(num_valid);
    return {output, is_2d};
}

namespace detail
{

// Markers for the field types pointCloud2MsgToPtsVec dispatches on: a field the cloud does not
// have (or a channel of a type it cannot read, which leaves the channel at 0), and a time field of
// a type it cannot read
struct NoField {};
struct UnknownTimeField {};

// What pointCloud2MsgToPtsVec reads from the message once, rather than once per point
struct PointCloud2Layout
{
    const uint8_t* data = nullptr;
    size_t point_step = 0;
    size_t num_points = 0;
    int off_x = 0;
    int off_y = 0;
    int off_z = 0;
    int off_channel = -1;
    int off_time = -1;
    int off_intensity = -1;
    int off_type = -1;
    int off_rgb = -1;
    int64_t time_ns = 0;            // header stamp
    int64_t unknown_time_value = 0; // what a time field of unknown type gives: the stamp in seconds
    double time_multiplier = 1.0;
    bool absolute_time = false;
};

// Dead channel lookup: a table over the range of the dead channels, or the set itself when that
// range is too large for one
class DeadChannels
{
    public:
        explicit DeadChannels(const std::set<int>& dead)
            : dead_(dead)
        {
            if(!dead.empty() && (((int64_t)*dead.rbegin() - (int64_t)*dead.begin()) < kMaxTableSize))
            {
                min_ = *dead.begin();
                table_.assign((size_t)(*dead.rbegin() - min_ + 1), 0);
                for(const int channel : dead)
                {
                    table_[channel - min_] = 1;
                }
            }
        }

        bool contains(const int channel) const
        {
            if(!table_.empty())
            {
                const int64_t k = (int64_t)channel - min_;
                return (k >= 0) && (k < (int64_t)table_.size()) && table_[k];
            }
            return dead_.find(channel) != dead_.end();
        }

    private:
        static constexpr int64_t kMaxTableSize = 1 << 16;
        const std::set<int>& dead_;
        int min_ = 0;
        std::vector<char> table_;
};

template<typename S>
inline S readUnaligned(const uint8_t* p)
{
    S value;
    memcpy(&value, p, sizeof(S));
    return value;
}

// The per-point loop of pointCloud2MsgToPtsVec for one type of channel field and one type of time
// field. Chosen once per message, so each field is read with a plain load instead of testing its
// type for every point, and every offset is a local constant rather than a reload through the
// message and the field table, which the writes to the output could otherwise alias.
template<typename T, typename ChannelT, typename TimeT>
inline void convertPointCloud2(const PointCloud2Layout layout, const DeadChannels* dead_channels,
        const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg, const std::vector<std::pair<int,int> >& fields,
        std::vector<Mat3>* covariances, std::vector<PointTemplated<T> >& output, bool& is_2d, size_t& num_non_finite)
{
    for(size_t i = 0; i < layout.num_points; ++i)
    {
        const uint8_t* p = layout.data + layout.point_step*i;
        int channel = 0;
        if constexpr (!std::is_same_v<ChannelT, NoField>)
        {
            channel = (int)readUnaligned<ChannelT>(p + layout.off_channel);
        }
        if((dead_channels != nullptr) && dead_channels->contains(channel))
        {
            continue; // Skip points with dead channels
        }

        const float x = readUnaligned<float>(p + layout.off_x);
        const float y = readUnaligned<float>(p + layout.off_y);
        const float z = readUnaligned<float>(p + layout.off_z);
        if(!isFinitePoint(x, y, z))
        {
            num_non_finite++;
            continue;
        }
        if(z != 0.0f)
        {
            is_2d = false;
        }

        PointTemplated<T>& pt = output.emplace_back();
        pt.x = (T)x;
        pt.y = (T)y;
        pt.z = (T)z;
        pt.channel = channel;
        if constexpr (std::is_same_v<TimeT, UnknownTimeField>)
        {
            pt.t = layout.unknown_time_value;
        }
        else if constexpr (!std::is_same_v<TimeT, NoField>)
        {
            const TimeT time = readUnaligned<TimeT>(p + layout.off_time);
            pt.t = (int64_t)(time * layout.time_multiplier);
            if(!layout.absolute_time)
            {
                pt.t += layout.time_ns;
            }
        }
        if(layout.off_intensity >= 0)
        {
            pt.i = readUnaligned<float>(p + layout.off_intensity);
        }
        if(layout.off_type >= 0)
        {
            pt.type = readUnaligned<int>(p + layout.off_type);
        }
        if(layout.off_rgb >= 0)
        {
            pt.r = p[layout.off_rgb + 2];
            pt.g = p[layout.off_rgb + 1];
            pt.b = p[layout.off_rgb + 0];
            pt.has_color = true;
        }
        if(covariances != nullptr)
        {
            covariances->push_back(readPointCovariance(msg, i, fields));
        }
    }
}

} // namespace detail

// Function to read a PointCloud2 message and convert it to a vector of points.
// `covariances`, when given, is filled with the per-point position covariance if the cloud carries
// the 6 cov_* fields, and left empty otherwise. It is an out-parameter rather than part of the return
// value so that the callers that do not care about it are unaffected. It stays aligned with the
// returned points, the skipped ones (dead channel, non-finite coordinates) being skipped in both.
template <typename T>
inline std::tuple<std::vector<PointTemplated<T> >, bool, bool, bool> pointCloud2MsgToPtsVec(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg, const double time_scale = 1e-9, bool need_time = true, const std::set<int>& dead_channels = std::set<int>(), bool absolute_time = false, std::vector<Mat3>* covariances = nullptr)
{
    std::vector<PointTemplated<T>> output;
    std::vector<std::pair<int,int> > fields = getPointFields(msg->fields, need_time);
    bool has_intensity = (fields[PointFieldTypes::INTENSITY].first != -1);
    bool has_channel = (fields[PointFieldTypes::CHANNEL].first != -1);
    bool has_type = (fields[PointFieldTypes::TYPE].first != -1);
    bool has_time = (fields[PointFieldTypes::TIME].first != -1);
    bool has_color = (fields[PointFieldTypes::RGB].first != -1);

    detail::PointCloud2Layout layout;
    layout.data = msg->data.data();
    layout.point_step = msg->point_step;
    layout.num_points = msg->width * msg->height;
    layout.off_x = fields[PointFieldTypes::X].first;
    layout.off_y = fields[PointFieldTypes::Y].first;
    layout.off_z = fields[PointFieldTypes::Z].first;
    layout.off_channel = fields[PointFieldTypes::CHANNEL].first;
    layout.off_time = fields[PointFieldTypes::TIME].first;
    layout.off_intensity = has_intensity ? fields[PointFieldTypes::INTENSITY].first : -1;
    layout.off_type = has_type ? fields[PointFieldTypes::TYPE].first : -1;
    layout.off_rgb = has_color ? fields[PointFieldTypes::RGB].first : -1;
    layout.time_ns = rclcpp::Time(msg->header.stamp).nanoseconds();
    layout.unknown_time_value = (int64_t)rclcpp::Time(msg->header.stamp).seconds();
    layout.time_multiplier = time_scale * 1e9;
    layout.absolute_time = absolute_time;
    output.reserve(layout.num_points);

    const detail::DeadChannels dead(dead_channels);
    const detail::DeadChannels* dead_ptr = (has_channel && !dead_channels.empty()) ? &dead : nullptr;

    bool fill_covariance = false;
    if(covariances != nullptr)
    {
        covariances->clear();
        fill_covariance = hasPointCovariance(fields);
        if(fill_covariance)
        {
            covariances->reserve(layout.num_points);
        }
    }
    std::vector<Mat3>* covariances_out = fill_covariance ? covariances : nullptr;

    bool is_2d = true;
    size_t num_non_finite = 0;
    auto convert = [&](auto channel_tag, auto time_tag)
    {
        detail::convertPointCloud2<T, decltype(channel_tag), decltype(time_tag)>(
                layout, dead_ptr, msg, fields, covariances_out, output, is_2d, num_non_finite);
    };
    auto dispatchTime = [&](auto channel_tag)
    {
        if(!has_time)
        {
            convert(channel_tag, detail::NoField{});
            return;
        }
        switch(fields[PointFieldTypes::TIME].second)
        {
            case sensor_msgs::msg::PointField::FLOAT64: convert(channel_tag, double{}); break;
            case sensor_msgs::msg::PointField::FLOAT32: convert(channel_tag, float{}); break;
            case sensor_msgs::msg::PointField::UINT32: convert(channel_tag, uint32_t{}); break;
            default:
                std::cout << "The time field is not of type float32 or float64 or unit32" << std::endl;
                convert(channel_tag, detail::UnknownTimeField{});
        }
    };
    if(!has_channel)
    {
        dispatchTime(detail::NoField{});
    }
    else
    {
        switch(fields[PointFieldTypes::CHANNEL].second)
        {
            case sensor_msgs::msg::PointField::UINT16: dispatchTime(uint16_t{}); break;
            case sensor_msgs::msg::PointField::INT32: dispatchTime(int32_t{}); break;
            case sensor_msgs::msg::PointField::UINT32: dispatchTime(uint32_t{}); break;
            case sensor_msgs::msg::PointField::INT16: dispatchTime(int16_t{}); break;
            case sensor_msgs::msg::PointField::INT8: dispatchTime(int8_t{}); break;
            case sensor_msgs::msg::PointField::UINT8: dispatchTime(uint8_t{}); break;
            default:
                std::cout << "The channel field is of unknown type" << std::endl;
                dispatchTime(detail::NoField{});
        }
    }
    reportNonFinitePoints(num_non_finite, layout.num_points);
    return {output, has_intensity, has_channel, is_2d};
}
/////// End helper functions to subscribe and publish PointCloud2 messages



/////// Beginning helper functions to convert between geometry_msgs::msg::Transform and Mat4
inline Mat4 transformToMat4(const geometry_msgs::msg::Transform& msg)
{
    Mat4 output = Mat4::Identity();
    output(0,3) = msg.translation.x;
    output(1,3) = msg.translation.y;
    output(2,3) = msg.translation.z;
    Eigen::Quaterniond q(msg.rotation.w, msg.rotation.x, msg.rotation.y, msg.rotation.z);
    output.block<3,3>(0,0) = q.toRotationMatrix();
    return output;
}

inline geometry_msgs::msg::Transform mat4ToTransform(const Mat4& mat)
{
    geometry_msgs::msg::Transform output;
    output.translation.x = mat(0,3);
    output.translation.y = mat(1,3);
    output.translation.z = mat(2,3);
    Eigen::Quaterniond q(mat.block<3,3>(0,0));
    output.rotation.x = q.x();
    output.rotation.y = q.y();
    output.rotation.z = q.z();
    output.rotation.w = q.w();
    return output;
}
/////// End helper functions to convert between geometry_msgs::msg::Transform and Mat4


inline int64_t getTimeNs(const rclcpp::Time& time)
{
    return time.nanoseconds();
}

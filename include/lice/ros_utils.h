#pragma once

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "geometry_msgs/msg/transform.hpp"

#include "types.h"
#include <Eigen/Dense>



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
    NUM_TYPES = 8
};

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
    std::vector<Pointd> output;
    output.resize(msg->width*msg->height);
    bool has_color = (msg->fields.size() > 8);
    int64_t time_offset = rclcpp::Time(msg->header.stamp).nanoseconds();
    bool is_2d = true;
    for(size_t i=0; i < output.size(); ++i)
    {
        float temp_x, temp_y, temp_z;
        memcpy(&(temp_x), &(msg->data[(msg->point_step*i) + 0]), sizeof(float));
        memcpy(&(temp_y), &(msg->data[(msg->point_step*i) + 4]), sizeof(float));
        memcpy(&(temp_z), &(msg->data[(msg->point_step*i) + 8]), sizeof(float));
        output[i].x = (double)temp_x;
        output[i].y = (double)temp_y;
        output[i].z = (double)temp_z;
        if(temp_z != 0.0f)
        {
            is_2d = false;
        }
        uint32_t t;
        memcpy(&(output[i].i), &(msg->data[(msg->point_step*i) + 12]), sizeof(float));
        memcpy(&t, &(msg->data[(msg->point_step*i) + 16]), sizeof(uint32_t));
        output[i].t = (int64_t)(t) + time_offset;
        memcpy(&(output[i].channel), &(msg->data[(msg->point_step*i) + 20]), sizeof(int));
        memcpy(&(output[i].type), &(msg->data[(msg->point_step*i) + 24]), sizeof(int));
        if(has_color)
        {
            output[i].r = msg->data[(msg->point_step*i) + 30];
            output[i].g = msg->data[(msg->point_step*i) + 29];
            output[i].b = msg->data[(msg->point_step*i) + 28];
            output[i].has_color = true;
        }
    }
    return {output, is_2d};
}

// Function to read a PointCloud2 message and convert it to a vector of points
template <typename T>
inline std::tuple<std::vector<PointTemplated<T> >, bool, bool, bool> pointCloud2MsgToPtsVec(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg, const double time_scale = 1e-9, bool need_time = true, const std::set<int>& dead_channels = std::set<int>(), bool absolute_time = false)
{
    std::vector<PointTemplated<T>> output;
    rclcpp::Time time = rclcpp::Time(msg->header.stamp);
    int64_t time_ns = time.nanoseconds();
    double time_multiplier = time_scale * 1e9;
    size_t num_points = msg->width * msg->height;
    output.reserve(num_points);
    std::vector<std::pair<int,int> > fields = getPointFields(msg->fields, need_time);
    bool has_intensity = (fields[PointFieldTypes::INTENSITY].first != -1);
    bool has_channel = (fields[PointFieldTypes::CHANNEL].first != -1);
    bool has_type = (fields[PointFieldTypes::TYPE].first != -1);
    bool has_time = (fields[PointFieldTypes::TIME].first != -1);
    bool has_color = (fields[PointFieldTypes::RGB].first != -1);

    bool has_dead_channel = false;
    bool is_2d = true;
    if(has_channel && !dead_channels.empty())
    {
        has_dead_channel = true;
    }

    for(size_t i = 0; i < num_points; ++i)
    {
        PointTemplated<T> pt;
        if(has_channel)
        {
            if(fields[PointFieldTypes::CHANNEL].second == sensor_msgs::msg::PointField::UINT16)
            {
                uint16_t temp_channel;
                memcpy(&(temp_channel),&(msg->data[(msg->point_step*i) + fields[PointFieldTypes::CHANNEL].first]), sizeof(uint16_t));
                pt.channel = (int)temp_channel;
            }
            else if(fields[PointFieldTypes::CHANNEL].second == sensor_msgs::msg::PointField::INT32)
            {
                int32_t temp_channel;
                memcpy(&(temp_channel),&(msg->data[(msg->point_step*i) + fields[PointFieldTypes::CHANNEL].first]), sizeof(int32_t));
                pt.channel = (int)temp_channel;
            }
            else if(fields[PointFieldTypes::CHANNEL].second == sensor_msgs::msg::PointField::UINT32)
            {
                uint32_t temp_channel;
                memcpy(&(temp_channel),&(msg->data[(msg->point_step*i) + fields[PointFieldTypes::CHANNEL].first]), sizeof(uint32_t));
                pt.channel = (int)temp_channel;
            }
            else if(fields[PointFieldTypes::CHANNEL].second == sensor_msgs::msg::PointField::INT16)
            {
                int16_t temp_channel;
                memcpy(&(temp_channel),&(msg->data[(msg->point_step*i) + fields[PointFieldTypes::CHANNEL].first]), sizeof(int16_t));
                pt.channel = (int)temp_channel;
            }
            else if(fields[PointFieldTypes::CHANNEL].second == sensor_msgs::msg::PointField::INT8)
            {
                int8_t temp_channel;
                memcpy(&(temp_channel),&(msg->data[(msg->point_step*i) + fields[PointFieldTypes::CHANNEL].first]), sizeof(int8_t));
                pt.channel = (int)temp_channel;
            }
            else if(fields[PointFieldTypes::CHANNEL].second == sensor_msgs::msg::PointField::UINT8)
            {
                uint8_t temp_channel;
                memcpy(&(temp_channel),&(msg->data[(msg->point_step*i) + fields[PointFieldTypes::CHANNEL].first]), sizeof(uint8_t));
                pt.channel = (int)temp_channel;
            }
            else
            {
                std::cout << "The channel field is of unknown type" << std::endl;
            }
        }
        // WARNING, CAN BE MAD FASTER BUY DOING A LOOKUP TABLE INSTEAD OF A SET
        if(has_dead_channel && (dead_channels.find(pt.channel) != dead_channels.end()))
        {
            continue; // Skip points with dead channels
        }

        float temp_x, temp_y, temp_z;
        memcpy(&(temp_x), &(msg->data[(msg->point_step*i) + fields[PointFieldTypes::X].first]), sizeof(float));
        memcpy(&(temp_y), &(msg->data[(msg->point_step*i) + fields[PointFieldTypes::Y].first]), sizeof(float));
        memcpy(&(temp_z), &(msg->data[(msg->point_step*i) + fields[PointFieldTypes::Z].first]), sizeof(float));
        pt.x = (T)temp_x;
        pt.y = (T)temp_y;
        pt.z = (T)temp_z;
        if(temp_z != 0.0f)
        {
            is_2d = false;
        }
        if(has_time)
        {
            if(fields[PointFieldTypes::TIME].second == sensor_msgs::msg::PointField::FLOAT64)
            {
                double temp_t;
                memcpy(&(temp_t), &(msg->data[(msg->point_step*i) + fields[PointFieldTypes::TIME].first]), sizeof(double));
                if(absolute_time)
                {
                    pt.t = (int64_t)(temp_t * time_multiplier);
                }
                else
                {
                    pt.t = time_ns + (int64_t)(temp_t * time_multiplier);
                }
            }
            else if(fields[PointFieldTypes::TIME].second == sensor_msgs::msg::PointField::FLOAT32)
            {
                float temp_t;
                memcpy(&(temp_t), &(msg->data[(msg->point_step*i) + fields[PointFieldTypes::TIME].first]), sizeof(float));
                if(absolute_time)
                {
                    pt.t = (int64_t)(temp_t * time_multiplier);
                }
                else
                {
                    pt.t = time_ns + (int64_t)(temp_t * time_multiplier);
                }
            }
            else if(fields[PointFieldTypes::TIME].second == sensor_msgs::msg::PointField::UINT32)
            {
                uint32_t temp_t;
                memcpy(&(temp_t), &(msg->data[(msg->point_step*i) + fields[PointFieldTypes::TIME].first]), sizeof(uint32_t));
                if(absolute_time)
                {
                    pt.t = (int64_t)(temp_t * time_multiplier);
                }
                else
                {
                    if(time_multiplier == 1.0)
                    {
                        pt.t = time_ns + (int64_t)temp_t;
                    }
                    else
                    {
                        pt.t = time_ns + (int64_t)(temp_t * time_multiplier);
                    }
                }
            }
            else
            {
                std::cout << "The time field is not of type float32 or float64 or unit32" << std::endl;
                pt.t = rclcpp::Time(msg->header.stamp).seconds();
            }
        }
        if(has_intensity)
        {
            memcpy(&(pt.i),&(msg->data[(msg->point_step*i) + fields[PointFieldTypes::INTENSITY].first]), sizeof(float));
        }
        if(has_type)
        {
            memcpy(&(pt.type),&(msg->data[(msg->point_step*i) + fields[PointFieldTypes::TYPE].first]), sizeof(int));
        }
        if(has_color)
        {
            memcpy(&(pt.r),&(msg->data[(msg->point_step*i) + fields[PointFieldTypes::RGB].first + 2]), sizeof(uint8_t));
            memcpy(&(pt.g),&(msg->data[(msg->point_step*i) + fields[PointFieldTypes::RGB].first + 1]), sizeof(uint8_t));
            memcpy(&(pt.b),&(msg->data[(msg->point_step*i) + fields[PointFieldTypes::RGB].first + 0]), sizeof(uint8_t));
            pt.has_color = true;
        }
        output.push_back(pt);
    }
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

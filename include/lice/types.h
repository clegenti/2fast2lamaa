#pragma once

#include <Eigen/Dense>
#include <vector>
#include <set>
#include <memory>
#include <iostream>
#include <vector>
#include <map>


// Types for the lidar features: 1: planar, 2: edge, 3: rough
const std::set<int> kTypes = {1, 2, 3};
const int kSkyPoint = 101;
const int kNonSkyPoint = 100;
const int kInvalidPoint = 200;



typedef Eigen::Matrix<double, 2, 2> Mat2;
typedef Eigen::Matrix<double, 3, 3> Mat3;
typedef Eigen::Matrix<float, 3, 3> Mat3f;
typedef Eigen::Matrix<double, 3, 1> Vec3;
typedef Eigen::Matrix<float, 3, 1> Vec3f;
typedef Eigen::Matrix<double, 4, 1> Vec4;
typedef Eigen::Matrix<float, 4, 1> Vec4f;
typedef Eigen::Matrix<double, 1, 3> Row3;
typedef Eigen::Matrix<double, 1, 4> Row4;
typedef Eigen::Matrix<double, 1, 6> Row6;
typedef Eigen::Matrix<double, 1, 2> Row2;
typedef Eigen::Matrix<double, 6, 6> Mat6;
typedef Eigen::Matrix<double, 4, 4> Mat4;
typedef Eigen::Matrix<float, 4, 4> Mat4f;
typedef Eigen::Matrix<double, 9, 9> Mat9;
typedef Eigen::Matrix<double, 12, 12> Mat12;
typedef Eigen::Matrix<double, 2, 1> Vec2;
typedef Eigen::Matrix<double, 6, 1> Vec6;
typedef Eigen::Matrix<double, 7, 1> Vec7;
typedef Eigen::Matrix<double, 8, 1> Vec8;
typedef Eigen::Matrix<double, 1, 8> Row8;
typedef Eigen::Matrix<double, 9, 1> Vec9;
typedef Eigen::Matrix<double, 1, 9> Row9;
typedef Eigen::Matrix<double, 1, 12> Row12;
typedef Eigen::Matrix<double, 12, 1> Vec12;
typedef Eigen::Matrix<double, 3, 6> Mat3_6;
typedef Eigen::Matrix<double, 3, 2> Mat3_2;
typedef Eigen::Matrix<double, 3, 4> Mat3_4;
typedef Eigen::Matrix<double, 3, 12> Mat3_12;
typedef Eigen::Matrix<double, 2, 6> Mat2_6;
typedef Eigen::Matrix<double, 9, 6> Mat9_6;
typedef Eigen::Matrix<double, 3, 9> Mat3_9;
typedef Eigen::Matrix<double, 6, 9> Mat6_9;
typedef Eigen::Matrix<double, 2, 8> Mat2_8;
typedef Eigen::Matrix<double, 2, 9> Mat2_9;
typedef Eigen::Matrix<double, 2, 3> Mat2_3;
typedef Eigen::Matrix<double, 9, 3> Mat9_3;
typedef Eigen::Matrix<double, 9, 8> Mat9_8;
typedef Eigen::Matrix<double, 12, 3> Mat12_3;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1> VecX;
typedef Eigen::Matrix<double, 1, Eigen::Dynamic> RowX;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatX;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatXRow;
typedef Eigen::Matrix<double, 3, Eigen::Dynamic> Mat3X;
typedef Eigen::Matrix<double, 2, Eigen::Dynamic> Mat2X;

typedef std::tuple<int, int, int> GridIndex;

const int kNoChannel = -10000;

template <typename T>
struct PointTemplated
{
    T x = 0.0;
    T y = 0.0;
    T z = 0.0;
    int64_t t = 0.0;
    float i = 0.0;
    int channel = 0;
    int type = 0;
    uint8_t r = 0;
    uint8_t g = 0;
    uint8_t b = 0;
    bool has_color = false;


    PointTemplated(const T _x, const T _y, const T _z, const int64_t _t, const float _intensity=0.0, const int _channel=0, const int _type=0)
        : x(_x)
        , y(_y)
        , z(_z)
        , t(_t)
        , i(_intensity)
        , channel(_channel)
        , type(_type)
    {
    }
    PointTemplated(const Vec3& pt, const int64_t _t, const double _intensity=0.0, const int _channel=0, const int _type=0)
        : x(pt[0])
        , y(pt[1])
        , z(pt[2])
        , t(_t)
        , i(_intensity)
        , channel(_channel)
        , type(_type)
    {
    }

    PointTemplated()
        : x(0.0)
        , y(0.0)
        , z(0.0)
        , t(0.0)
        , i(0.0)
    {
    }
    Eigen::Matrix<T,3,1> vec3() const
    {
        return Eigen::Matrix<T,3,1>(x, y, z);
    }
    Vec3f vec3f() const
    {
        return Vec3f((float)x, (float)y, (float)z);
    }
    Vec3 vec3d() const
    {
        return Vec3((double)x, (double)y, (double)z);
    }

    void setVec3(const Vec3& pt)
    {
        x = pt[0];
        y = pt[1];
        z = pt[2];
    }

    int64_t nanos() const
    {
        return t;
    }
};

typedef PointTemplated<double> Pointd;
typedef PointTemplated<float> Pointf;



struct DataAssociation
{
    std::vector<std::pair<int, int> > target_ids;
    int pc_id;
    int feature_id;
    int type; // 0: planar, 1: edge, 2: rough

    std::pair<Vec3, std::vector<Vec3>> getPointVectors(const std::vector<std::shared_ptr<std::vector<Pointd> > >& features) const
    {
        std::vector<Vec3> points;
        for(size_t i = 0; i < target_ids.size(); ++i)
        {
            points.push_back(features[target_ids[i].first]->at(target_ids[i].second).vec3());
        }
        return std::make_pair(features[pc_id]->at(feature_id).vec3(), points);
    }

    int residualSize() const
    {
        if((type == 2) || (type == 3)) return 1;
        
        return 1;
    }
    int nbTargets() const
    {
        if((type == 2) || (type == 3)) return 2;
        return 3;
    }
    template<typename T>
    T computeResidual(const Eigen::Matrix<T, 3,1> & feature, const std::vector<Eigen::Matrix<T,3,1> >& targets) const
    {
        T output = T(0.0);
        // Point to plane residual
        if(type == 1)
        {
            Eigen::Matrix<T,3,1> v1 = targets[0] - targets[1];
            Eigen::Matrix<T,3,1> v2 = targets[0] - targets[2];
            Eigen::Matrix<T,3,1> n = v1.cross(v2);
            n.normalize();
            output = n.dot(feature - targets[0]);
        }
        // Point to line residual
        else if((type == 2) || (type == 3))
        {
            Eigen::Matrix<T,3,1> v1 = feature - targets[0];
            Eigen::Matrix<T,3,1> v2 = feature - targets[1];
            Eigen::Matrix<T,3,1> v3 = targets[1] - targets[0];
            Eigen::Matrix<T,3,1> n = v1.cross(v2);
            output = (n / (v3.norm())).norm();
        }
        return output;
    }

    RowX computeJacobian(const Vec3& feature, const std::vector<Vec3>& targets) const
    {
        RowX output(3+(3*targets.size()));
        if(type == 1)
        {
            Vec3 v1 = targets[0] - targets[1];
            Vec3 v2 = targets[0] - targets[2];
            Vec3 n = v1.cross(v2);
            double n_norm = n.norm();
            if (n_norm < 1e-6)
            {
                output.setZero();
                return output;
            }
            Vec3 v3 = feature - targets[0];
            Vec3 nu = n / n_norm;
            Vec3 v21 = targets[2] - targets[1];

            Row12 temp_1;
            temp_1.block<1,3>(0,0) = n.transpose();
            temp_1.block<1,3>(0,3) = (v3.cross(v21) - n).transpose();
            temp_1.block<1,3>(0,6) = (v3.cross(v2)).transpose();
            temp_1.block<1,3>(0,9) = (v1.cross(v3)).transpose();

            Row12 temp_2;
            temp_2.block<1,3>(0,0) = Row3::Zero();
            temp_2[3] = nu[1]*v21[2] - nu[2]*v21[1];
            temp_2[4] = nu[2]*v21[0] - nu[0]*v21[2];
            temp_2[5] = nu[0]*v21[1] - nu[1]*v21[0];
            temp_2[6] = nu[1]*v2[2] - nu[2]*v2[1];
            temp_2[7] = nu[2]*v2[0] - nu[0]*v2[2];
            temp_2[8] = nu[0]*v2[1] - nu[1]*v2[0];
            temp_2[9] = nu[2]*v1[1] - nu[1]*v1[2];
            temp_2[10] = nu[0]*v1[2] - nu[2]*v1[0];
            temp_2[11] = nu[1]*v1[0] - nu[0]*v1[1];

            output = (temp_1 * n_norm - temp_2*(v3.dot(n)) ) / (n_norm*n_norm);
        }
        else if(type == 2 || type == 3)
        {
            Vec3 v1 = feature - targets[0];
            Vec3 v2 = feature - targets[1];
            Vec3 v3 = targets[1] - targets[0];
            double v3_norm = v3.norm();
            if (v3_norm < 1e-6)
            {
                output.setZero();
                return output;
            }
            double v3_norm_inv = 1.0/v3_norm;
            Vec3 n = v1.cross(v2);
            
            double temp_norm = (n/(v1.norm()*v2.norm())).norm();
            if( std::isnan(temp_norm) || temp_norm < 1e-4)
            {
                output.setZero();
                return output;
            }   

            Mat3_6 temp_d_vector_prod;
            temp_d_vector_prod(0,0) = 0.0;
            temp_d_vector_prod(0,1) = -v2[2];
            temp_d_vector_prod(0,2) = v2[1];
            temp_d_vector_prod(0,3) = 0.0;
            temp_d_vector_prod(0,4) = v1[2];
            temp_d_vector_prod(0,5) = -v1[1];
            temp_d_vector_prod(1,0) = v2[2];
            temp_d_vector_prod(1,1) = 0.0;
            temp_d_vector_prod(1,2) = -v2[0];
            temp_d_vector_prod(1,3) = -v1[2];
            temp_d_vector_prod(1,4) = 0.0;
            temp_d_vector_prod(1,5) = v1[0];
            temp_d_vector_prod(2,0) = -v2[1];
            temp_d_vector_prod(2,1) = v2[0];
            temp_d_vector_prod(2,2) = 0.0;
            temp_d_vector_prod(2,3) = v1[1];
            temp_d_vector_prod(2,4) = -v1[0];
            temp_d_vector_prod(2,5) = 0.0;

            Row6 temp_d_v3_norm;
            temp_d_v3_norm[0] = -v3[0]*v3_norm_inv;
            temp_d_v3_norm[1] = -v3[1]*v3_norm_inv;
            temp_d_v3_norm[2] = -v3[2]*v3_norm_inv;
            temp_d_v3_norm[3] = v3[0]*v3_norm_inv;
            temp_d_v3_norm[4] = v3[1]*v3_norm_inv;
            temp_d_v3_norm[5] = v3[2]*v3_norm_inv;

            MatX temp(3, 9);
            temp(0,0) = 0.0;
            temp(0,1) = v3_norm_inv*(-v3[2]);
            temp(0,2) = v3_norm_inv*(v3[1]);
            temp(1,0) = v3_norm_inv*(v3[2]);
            temp(1,1) = 0.0;
            temp(1,2) = v3_norm_inv*(-v3[0]);
            temp(2,0) = v3_norm_inv*(-v3[1]);
            temp(2,1) = v3_norm_inv*(v3[0]);
            temp(2,2) = 0.0;

            temp.block<3,6>(0,3) = temp_d_vector_prod * v3_norm_inv - (v3_norm_inv*v3_norm_inv*n)*temp_d_v3_norm;

            output =  ((n.transpose() * v3_norm_inv) / ((n*v3_norm_inv).norm()))*temp;
        }

        return output;
    }

};

inline void testDataAssociationJacobian(int type = 1)
{
    DataAssociation da;
    da.type = type;

    Vec3 feature = Vec3::Random();
    std::vector<Vec3> targets;
    for (int i = 0; i < da.nbTargets(); ++i)
    {
        targets.push_back(Vec3::Random());
    }

    RowX jacobian = da.computeJacobian(feature, targets);

    RowX jacobian_num = MatX::Zero(da.residualSize(), 3+(3*targets.size()));

    double eps = 1e-6;

    double res = da.computeResidual(feature, targets);

    for (int i = 0; i < jacobian.cols(); ++i)
    {
        Vec3 f_shift = feature;
        std::vector<Vec3> t_shift = targets;
        if(i < 3)
        {
            f_shift[i] += eps;
        }
        else
        {
            t_shift[(i-3)/3][(i-3)%3] += eps;
        }
        double res_shift = da.computeResidual(f_shift, t_shift);
        jacobian_num[i] = (res_shift - res) / eps;
    }

    std::cout << "Test data association residuals type: " << std::endl << type << std::endl;
    std::cout << "Jacobian: " << std::endl << jacobian << std::endl;
    std::cout << "Jacobian num: " << std::endl << jacobian_num << std::endl;
}





enum class LidarOdometryMode
{
    IMU,
    GYR,
    NO_IMU
};
const std::map<std::string, LidarOdometryMode> kLidarOdometryModeMap = {
    {"imu", LidarOdometryMode::IMU},
    {"gyr", LidarOdometryMode::GYR},
    {"no_imu", LidarOdometryMode::NO_IMU}
};

#pragma once


#include "lice/types.h"
#include <random>
#include <ceres/rotation.h>

const double kExpNormTolerance = 1e-14;


inline Mat3 toSkewSymMat(const Vec3& rot_vec)
{
    Mat3 skew_mat;
    skew_mat << 0.0, -rot_vec(2), rot_vec(1),
                rot_vec(2), 0.0, -rot_vec(0),
                -rot_vec(1), rot_vec(0), 0.0;
    return skew_mat;

}

inline Mat3_4 jacobianQuatRotation(const Vec4& q, const Vec3& x)
{
    Mat3_4 output;
    const double& q1 = q[0];
    const double& q2 = q[1];
    const double& q3 = q[2];
    const double& q4 = q[3];
    const double& x1 = x[0];
    const double& x2 = x[1];
    const double& x3 = x[2];
    output << 
        2*q3*x3 - 2*q4*x2,           2*q3*x2 + 2*q4*x3, 2*q1*x3 + 2*q2*x2 - 4*q3*x1, 2*q2*x3 - 2*q1*x2 - 4*q4*x1,
        2*q4*x1 - 2*q2*x3, 2*q3*x1 - 4*q2*x2 - 2*q1*x3,           2*q2*x1 + 2*q4*x3, 2*q1*x1 + 2*q3*x3 - 4*q4*x2,
        2*q2*x2 - 2*q3*x1, 2*q1*x2 - 4*q2*x3 + 2*q4*x1, 2*q4*x2 - 4*q3*x3 - 2*q1*x1,           2*q2*x1 + 2*q3*x2;

    return output;
}


// Righthand Jacobian of SO3 Exp mapping
template<typename T>
inline Eigen::Matrix<T, 3, 3> jacobianRighthandSO3(const Eigen::Matrix<T, 3, 1> rot_vec)
{    
    Eigen::Matrix<T, 3, 3> output = Eigen::Matrix<T, 3, 3>::Identity();
    T vec_norm = rot_vec.norm();

            
    Eigen::Matrix<T, 3, 1> vec = rot_vec;

    if(vec_norm > kExpNormTolerance)
    {    

        Eigen::Matrix<T, 3, 3> skew_mat;
        skew_mat << T(0.0), T(-vec(2)), T(vec(1)),
                    T(vec(2)), T(0.0), T(-vec(0)),
                    T(-vec(1)), T(vec(0)), T(0.0);
                
        output += ( (vec_norm - sin(vec_norm)) / (vec_norm*vec_norm*vec_norm) )*skew_mat*skew_mat  - ( (1.0 - cos(vec_norm))/(vec_norm*vec_norm) )*skew_mat;
    }    
    return output;
}

// Lefthand Jacobian of SO3 Exp mapping
inline Mat3 jacobianLefthandSO3( Vec3 rot_vec)
{
    return jacobianRighthandSO3<double>(-rot_vec);
}



// SO3 Log mapping
inline Vec3 logMap(const Mat3& rot_mat){
    Eigen::AngleAxisd rot_axis(rot_mat);
    return rot_axis.angle() * rot_axis.axis();
}   


// SO3 Exp mapping
inline Mat3 expMap(const Vec3& vec){
    Eigen::AngleAxisd rot_axis(vec.norm(), vec.normalized());
    return rot_axis.toRotationMatrix();
}

inline std::pair<Vec3, Vec3> invertTransform(const Vec3& p, const Vec3& rot)
{
    // Convert to rotation matrix
    Mat3 R = expMap(rot);
    // Invert rotation matrix
    Mat3 R_inv = R.transpose();
    // Invert translation
    Vec3 p_inv = -R_inv*p;
    // Convert back to rotation vector
    Vec3 rot_inv = logMap(R_inv);
    return {p_inv, rot_inv};
}


inline std::pair<Vec3, Vec3> combineTransforms(const Vec3& p0, const Vec3& rot0, const Vec3& p1, const Vec3& rot1)
{
    // Convert to rotation matrix
    Mat3 R0 = expMap(rot0);
    Mat3 R1 = expMap(rot1);
    // Combine rotation matrix
    Mat3 R = R0*R1;
    // Combine translation
    Vec3 p = R0*p1 + p0;
    // Convert back to rotation vector
    Vec3 rot = logMap(R);
    return {p, rot};
}

inline std::pair<double, double> distanceBetweenTransforms(const Mat4& T0, const Mat4& T1)
{
    Mat4 T = T0.inverse()*T1;
    Vec3 p = T.block<3, 1>(0, 3);
    Vec3 rot = logMap(T.block<3, 3>(0, 0));
    return {p.norm(), rot.norm()};
}


inline Mat4 posRotToTransform(const Vec3& pos, const Vec3& rot)
{
    Mat4 T = Mat4::Identity();
    T.block<3, 3>(0, 0) = expMap(rot);
    T.block<3, 1>(0, 3) = pos;
    return T;
}

inline Mat4 posRotToTransform(const Vec6& pose)
{
    Mat4 T = Mat4::Identity();
    T.block<3, 3>(0, 0) = expMap(pose.tail<3>());
    T.block<3, 1>(0, 3) = pose.head<3>();
    return T;
}


// Cartesian to polar coordinates (r, elevation, azimuth)
inline Vec3 toPolar(const Vec3& xyz)
{
    double xy2 = (xyz[0]*xyz[0]) + (xyz[1]*xyz[1]);
    return Vec3(
        std::sqrt(xy2 + (xyz[2]*xyz[2])),
        std::atan2(std::sqrt(xy2), xyz[2]),
        std::atan2(xyz[1], xyz[0]));
}

inline Vec3f toPolarFloat(const Vec3f& xyz)
{
    float xy2 = (xyz[0]*xyz[0]) + (xyz[1]*xyz[1]);
    return Vec3f(
        std::sqrt(xy2 + (xyz[2]*xyz[2])),
        std::atan2(std::sqrt(xy2), xyz[2]),
        std::atan2(xyz[1], xyz[0]));
}
        

inline Vec3 toXYZ(const Vec3& pol)
{
    double s = std::sin(pol[1]);
    return Vec3(
        pol[0]*s*std::cos(pol[2]),
        pol[0]*s*std::sin(pol[2]),
        pol[0]*std::cos(pol[1]));
}

inline Vec3f toXYZFloat(const Vec3f& pol)
{
    float s = std::sin(pol[1]);
    return Vec3f(
        pol[0]*s*std::cos(pol[2]),
        pol[0]*s*std::sin(pol[2]),
        pol[0]*std::cos(pol[1]));
}


inline std::pair<Vec3, Vec3> getPcTransform(const std::vector<Vec3>& target, const std::vector<Vec3>& source)
{
    if(target.size() != source.size() || target.size() <= 2)
    {
        throw std::runtime_error("getPcTransform: target and source must have the same size and at least 3 points");
    }

    // Compute centroids
    Vec3 target_centroid = Vec3::Zero();
    Vec3 source_centroid = Vec3::Zero();
    for(size_t i = 0; i < target.size(); ++i)
    {
        target_centroid += target[i];
        source_centroid += source[i];
    }
    target_centroid /= target.size();
    source_centroid /= source.size();

    // Compute cross-covariance matrix
    Mat3 H = Mat3::Zero();
    for(size_t i = 0; i < target.size(); ++i)
    {
        Vec3 d_target = target[i] - target_centroid;
        Vec3 d_source = source[i] - source_centroid;
        H += d_source * d_target.transpose();
    }

    // Compute SVD using eigen's JacobiSVD
    Eigen::JacobiSVD<Mat3> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Mat3 U = svd.matrixU();
    Mat3 V = svd.matrixV();
    Mat3 R = V * U.transpose();
    if(R.determinant() < 0)
    {
        V.col(2) *= -1;
        R = V * U.transpose();
    }
    Vec3 t = target_centroid - R * source_centroid;
    return {t, logMap(R)};
}


template <typename T>
inline Eigen::Matrix<T, 3, 1> rotMatToAngleAxis(const Eigen::Matrix<T, 3, 3>& R)
{
    Eigen::Matrix<T, 3, 1> angle_axis;
    ceres::RotationMatrixToAngleAxis(R.data(), angle_axis.data());
    return angle_axis;
}


template <typename T>
inline Eigen::Matrix<T, 4, 4> posQuatToTransform(const T* const pose)
{
    Eigen::Matrix<T, 4, 4> trans = Eigen::Matrix<T, 4, 4>::Identity();
    trans.template block<3, 1>(0, 3) = Eigen::Matrix<T, 3, 1>(pose[0], pose[1], pose[2]);
    Eigen::Matrix<T, 3, 3, Eigen::RowMajor> R;
    ceres::QuaternionToRotation(pose + 3, R.data());
    trans.template block<3, 3>(0, 0) = R;
    return trans;
}

template <typename T>
inline Eigen::Matrix<T, 4, 4> posQuatToTransform(const Eigen::Matrix<T, 7, 1>& pose)
{
    Eigen::Matrix<T, 4, 4> trans = Eigen::Matrix<T, 4, 4>::Identity();
    trans.template block<3, 1>(0, 3) = pose.template block<3, 1>(0, 0);
    Eigen::Matrix<T, 3, 3, Eigen::RowMajor> R;
    ceres::QuaternionToRotation(pose.template block<4, 1>(3, 0).data(), R.data());
    trans.template block<3, 3>(0, 0) = R;
    return trans;
}

template <typename T>
inline Eigen::Matrix<T, 7, 1> transformToPosQuat(const Eigen::Matrix<T, 4, 4>& trans)
{
    Eigen::Matrix<T, 7, 1> pose;
    // Translation part
    pose.template block<3, 1>(0, 0) = trans.template block<3, 1>(0, 3);
    // Rotation part (as quaternion)
    Eigen::Matrix<T, 3, 3> R = trans.template block<3, 3>(0, 0);
    Eigen::Matrix<T, 4, 1> quat;
    ceres::RotationMatrixToQuaternion(R.data(), quat.data());
    pose.template block<4, 1>(3, 0) = quat;
    return pose;
}

template <typename T>
inline Eigen::Matrix<T, 6, 1> posQuatToPosRot(const Eigen::Matrix<T, 7, 1>& pose)
{
    Eigen::Matrix<T, 6, 1> pose_vec;
    // Translation part
    pose_vec.template block<3, 1>(0, 0) = pose.template block<3, 1>(0, 0);
    // Rotation part (using logarithm map)
    Eigen::Matrix<T, 3, 3, Eigen::RowMajor> R;
    ceres::QuaternionToRotation(pose.template block<4, 1>(3, 0).data(), R.data());
    Eigen::Matrix<T, 3, 3> R_col_major = R;
    Eigen::Matrix<T, 3, 1> rot_vec;
    ceres::RotationMatrixToAngleAxis(R_col_major.data(), rot_vec.data());
    pose_vec.template block<3, 1>(3, 0) = rot_vec;
    return pose_vec;
}


template <typename T>
inline Eigen::Matrix<T, 6, 1> transformToPoseVector(const Eigen::Matrix<T, 4, 4>& trans)
{
    Eigen::Matrix<T, 6, 1> pose_vec;
    // Translation part
    pose_vec.template block<3, 1>(0, 0) = trans.template block<3, 1>(0, 3);
    // Rotation part (using logarithm map)
    Eigen::Matrix<T, 3, 3> R = trans.template block<3, 3>(0, 0);
    Eigen::Matrix<T, 3, 1> rot_vec;
    ceres::RotationMatrixToAngleAxis(R.data(), rot_vec.data());
    pose_vec.template block<3, 1>(3, 0) = rot_vec;
    return pose_vec;
}
#ifndef PREINT_TYPES_H
#define PREINT_TYPES_H

#include <iostream>
#include <algorithm>
#include <memory>
#include <numeric>
#include "Eigen/Dense"


namespace ugpm
{

    // Preintegration methods
    enum PreintType {LPM, UGPM};

    inline PreintType strToPreintType(std::string type)
    {
        std::transform(type.begin(), type.end(), type.begin(), [](unsigned char c){ return std::tolower(c); });
        PreintType output;
        if(type == "lpm") output = LPM;
        else if(type == "ugpm") output = UGPM;
        else
        {
            throw std::range_error("The type of preintegration method is unknown, program stopping now");
        }
        return output;
    }



    typedef Eigen::Matrix<double, 2, 1> Vec2;
    typedef Eigen::Matrix<double, 3, 1> Vec3;
    typedef Eigen::Matrix<double, 9, 1> Vec9;
    typedef Eigen::Matrix<double, 18, 1> Vec18;
    typedef Eigen::Matrix<double, Eigen::Dynamic, 1> VecX;
    typedef Eigen::Matrix<double, 1, Eigen::Dynamic> RowX;

    typedef Eigen::Matrix<double, 1, 9> Row9;

    typedef Eigen::Matrix<double, 3, 3> Mat3;
    typedef Eigen::Matrix<double, 6, 6> Mat6;
    typedef Eigen::Matrix<double, 9, 9> Mat9;
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatX;

    typedef Eigen::Matrix<double, 3, 2> Mat3_2;
    typedef Eigen::Matrix<double, 3, 6> Mat3_6;
    typedef Eigen::Matrix<double, 3, 9> Mat3_9;

    typedef Eigen::Matrix<double, 9, 3> Mat9_3;
    typedef Eigen::Matrix<double, 9, 6> Mat9_6;

    inline Vec3 stdToVec3(const std::vector<double>& v)
    {
        Vec3 output;
        output << v[0], v[1], v[2];
        return output;
    }


    // Data structure to store individual measurements (can be acc or gyr data)
    struct ImuSample
    {
        double t;
        double data[3];
    };

    // Data structure to store a collection of IMU measurements
    struct ImuData
    {
        // Offset in second for every sample timestamps
        double t_offset = 0.0;

        // Vectors of samples
        std::vector<ImuSample> acc;
        std::vector<ImuSample> gyr;

        // Noise specification (assuming same on 3 axis)
        double acc_var;
        double gyr_var;

        ImuData(){};

        ImuData(const ImuData& imu_data):
            t_offset(imu_data.t_offset),
            acc(imu_data.acc),
            gyr(imu_data.gyr),
            acc_var(imu_data.acc_var),
            gyr_var(imu_data.gyr_var)
        {
        };

        ImuData& operator=(const ImuData& imu_data)
        {
            t_offset = imu_data.t_offset;
            acc = imu_data.acc;
            gyr = imu_data.gyr;
            acc_var = imu_data.acc_var;
            gyr_var = imu_data.gyr_var;
            return *this;
        }

        bool checkFrequency()
        {
            bool output = true;
            // Get min and max delta t for both the acc and gyr data
            double min_acc_dt = 1e10;
            double max_acc_dt = -1e10;
            for(size_t i = 0; i < acc.size()-1; ++i)
            {
                double dt = acc[i+1].t - acc[i].t;
                if(dt < min_acc_dt) min_acc_dt = dt;
                if(dt > max_acc_dt) max_acc_dt = dt;
            }
            double min_gyr_dt = 1e10;
            double max_gyr_dt = -1e10;
            for(size_t i = 0; i < gyr.size()-1; ++i)
            {
                double dt = gyr[i+1].t - gyr[i].t;
                if(dt < min_gyr_dt) min_gyr_dt = dt;
                if(dt > max_gyr_dt) max_gyr_dt = dt;
            }

            if(min_acc_dt < 0)
            {
                std::cout << "WARNING: Accelerometer data is not sorted in time" << std::endl;
                output = false;
            }
            if(min_gyr_dt < 0)
            {
                std::cout << "WARNING: Gyroscope data is not sorted in time" << std::endl;
                output = false;
            }
            double range_acc_dt = max_acc_dt - min_acc_dt;
            double range_gyr_dt = max_gyr_dt - min_gyr_dt;
            if ( (range_acc_dt/max_acc_dt > 0.1) || (range_gyr_dt/max_gyr_dt > 0.1) )
            {
                std::cout << "WARNING: Accelerometer or gyroscope data is not sampled at a constant frequency" << std::endl;
                std::cout << "\tMin \\ max delta time for acc " << min_acc_dt << " \\ " << max_acc_dt << std::endl;
                std::cout << "\tMin \\ max delta time for gyr " << min_gyr_dt << " \\ " << max_gyr_dt << std::endl;
                output = false;
            }
            return output;
        }
        // Return the collection of samples in between the two timestamps
        ImuData get(double from, double to)
        {
            // Check if the query interval makes sense
            if(from <= to)
            {
                ImuData output;

                output.t_offset = t_offset;
                output.acc_var = acc_var;
                output.gyr_var = gyr_var;
                output.acc = get(acc, from, to);
                output.gyr = get(gyr, from, to);
                return output;
            }
            // If the query inteval does not make sense throw an exception
            else
            {
                throw std::invalid_argument("The argument of ImuData::Get are not consistent");
            }
        }

        void print()
        {
            std::cout << "Imu data with offset of " << t_offset << std::endl;
            std::cout << "  Accelerometer data (" << acc.size() << " samples with std = " << sqrt(acc_var) << ")" << std::endl;
            for(const auto& a : acc)
            {
                std::cout << "      t = " << a.t << ":    "
                        << a.data[0] << "    "
                        << a.data[1] << "    "
                        << a.data[2] << std::endl;
            }
            std::cout << "  Gyroscope data (" << gyr.size() << " samples with std = " << sqrt(gyr_var) << ")" << std::endl;
            for(const auto& g : gyr)
            {
                std::cout << "      t = " << g.t << ":    " 
                        << g.data[0] << "    "
                        << g.data[1] << "    "
                        << g.data[2] << std::endl;
            }
        }

        private:
            // Return the collection of samples in between the two timestamps
            std::vector<ImuSample> get(const std::vector<ImuSample>& samples, double from, double to)
            {
                std::vector<ImuSample> output;
                if(from >= to)
                {
                    std::cout << "WARNING: Trying to get IMU data from a time range null or negative" << std::endl;
                    return output;
                }

                // TODO: could be optimised (keeping an internal pointer for repetitive calls instead for looping through all the data)
                size_t i = 0;
                bool loop = true;
                while(loop)
                {
                    if((samples[i].t > from) || (samples[i].t <= from && i == samples.size() - 1) || (samples[i].t <= from && samples[i+1].t > from) )
                    {
                        if( (samples[i].t < to) || (samples[i].t >= to && i == 0) || (samples[i].t >= to && samples[i-1].t < to) )
                        {
                            output.push_back(samples[i]);
                        }
                        else
                        {
                            loop = false;
                        }
                    }
                    if(i < (samples.size() - 1) )
                    {
                        i++;
                    }
                    else
                    {
                        loop = false;
                    }
                }

                return output;
            }
    };
    typedef std::shared_ptr<ImuData> ImuDataPtr;









    // Structure to store the different elements of a preintegrated measurement
    struct PreintMeasBasic{
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        Mat3 delta_R;
        Vec3 delta_v;
        Vec3 delta_p;
        double dt; 
        double dt_sq_half;

        void print() const
        {
            std::cout << "Preintegrated measurement" << std::endl;
            std::cout << "  dt = " << dt << std::endl;
            std::cout << "  dt_sq_half = " << dt_sq_half << std::endl;
            std::cout << "  Delta R = " << std::endl
                    << "    " << delta_R.row(0) << std::endl
                    << "    " << delta_R.row(1) << std::endl
                    << "    " << delta_R.row(2) << std::endl;
            std::cout << "  Delta v = " << std::endl 
                    << "    " << delta_v.row(0) << std::endl
                    << "    " << delta_v.row(1) << std::endl
                    << "    " << delta_v.row(2) << std::endl;
            std::cout << "  Delta p = " << std::endl
                    << "    " << delta_p.row(0) << std::endl
                    << "    " << delta_p.row(1) << std::endl
                    << "    " << delta_p.row(2) << std::endl;
        }
    };

    struct PreintMeas : PreintMeasBasic
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        Mat9 cov;

        Mat3 d_delta_R_d_bw;
        Vec3 d_delta_R_d_t;
     
        Mat3 d_delta_v_d_bw;
        Mat3 d_delta_v_d_bf;
        Vec3 d_delta_v_d_t;
     
        Mat3 d_delta_p_d_bw;
        Mat3 d_delta_p_d_bf;
        Vec3 d_delta_p_d_t;

        PreintMeas(Mat3 d_R,
                Vec3 d_v,
                Vec3 d_p,
                double dt, 
                double dt_sq_half,
                Mat9 cov_mat) :
                PreintMeasBasic{d_R, d_v, d_p, dt, dt_sq_half}, cov(cov_mat) {}; 

        PreintMeas(){};



        void printAll() const
        {
            print();
            std::cout << "  Covariance = " << std::endl
                    << "    " << cov.row(0) << std::endl
                    << "    " << cov.row(1) << std::endl
                    << "    " << cov.row(2) << std::endl
                    << "    " << cov.row(3) << std::endl
                    << "    " << cov.row(4) << std::endl
                    << "    " << cov.row(5) << std::endl
                    << "    " << cov.row(6) << std::endl
                    << "    " << cov.row(7) << std::endl
                    << "    " << cov.row(8) << std::endl;
            std::cout << "  Jacobian Delta R / gyr bias = " << std::endl
                    << "    " << d_delta_R_d_bw.row(0) << std::endl
                    << "    " << d_delta_R_d_bw.row(1) << std::endl
                    << "    " << d_delta_R_d_bw.row(2) << std::endl;
            std::cout << "  Jacobian Delta R / time-shift = " << std::endl
                    << "    " << d_delta_R_d_t.transpose() << std::endl;
            std::cout << "  Jacobian Delta v / gyr bias = " << std::endl
                    << "    " << d_delta_v_d_bw.row(0) << std::endl
                    << "    " << d_delta_v_d_bw.row(1) << std::endl
                    << "    " << d_delta_v_d_bw.row(2) << std::endl;
            std::cout << "  Jacobian Delta v / acc bias = " << std::endl
                    << "    " << d_delta_v_d_bf.row(0) << std::endl
                    << "    " << d_delta_v_d_bf.row(1) << std::endl
                    << "    " << d_delta_v_d_bf.row(2) << std::endl;
            std::cout << "  Jacobian Delta v / time-shift = " << std::endl
                    << "    " << d_delta_v_d_t.transpose() << std::endl;
            std::cout << "  Jacobian Delta p / gyr bias = " << std::endl
                    << "    " << d_delta_p_d_bw.row(0) << std::endl
                    << "    " << d_delta_p_d_bw.row(1) << std::endl
                    << "    " << d_delta_p_d_bw.row(2) << std::endl;
            std::cout << "  Jacobian Delta p / acc bias = " << std::endl
                    << "    " << d_delta_p_d_bf.row(0) << std::endl
                    << "    " << d_delta_p_d_bf.row(1) << std::endl
                    << "    " << d_delta_p_d_bf.row(2) << std::endl;
            std::cout << "  Jacobian Delta p / time-shift = " << std::endl
                    << "    " << d_delta_p_d_t.transpose() << std::endl;
        }


    };
    typedef std::shared_ptr<PreintMeas> PreintMeasPtr;


    struct PreintOption
    {
        double min_freq = 500;
        PreintType type = UGPM;
        double quantum = -1;
        double state_freq = 50.0;
        bool correlate = true;
    };

    struct PreintPrior
    {
        std::vector<double> acc_bias = {0,0,0};
        std::vector<double> gyr_bias = {0,0,0};
    };

    
    struct GPSeHyper
    {
        // Lengthscale squared
        double l2;
        // Sigma squared of the signal (in front of exponential)
        double sf2;
        // Sigma squared of measurement noise
        double sz2;
        // Mean of signal
        double mean;
    };


     // Function to sort indexes
    template <typename T>
    std::vector<int> sortIndexes(const std::vector<T> &v)
    {

        // initialize original index locations
        std::vector<int> idx(v.size());
        std::iota(idx.begin(), idx.end(), 0); 

        // sort indexes based on comparing values in v
        std::sort(idx.begin(), idx.end(),
            [&v](int i1, int i2) {return v[i1] < v[i2];});

        return idx;
    }


    // Tool to sort vector of vector and keep track of original indexes
    template <class T>
    class SortIndexTracker2
    {

        public:
            SortIndexTracker2( std::vector<std::vector<T> >& data):
                data_(&data)
            {
                std::vector<T> temp_data;
                std::vector<std::pair<int, int> > temp_map;
                for(size_t i = 0; i < data_->size(); ++i)
                {
                    temp_data.insert(temp_data.end(), data_->at(i).begin(), data_->at(i).end());
                    for(size_t j = 0; j < data_->at(i).size(); ++j)
                    {
                        temp_map.push_back(std::make_pair(i,j));
                    }
                }
                
                std::vector<int> sorted_indexes = sortIndexes(temp_data);

                for(size_t i = 0; i < sorted_indexes.size(); ++i)
                {
                    index_map_.push_back(temp_map[sorted_indexes[i]]);
                }
            }

            T get(const int i) const
            {
                return (*data_)[index_map_[i].first][index_map_[i].second];
            }

            std::pair<int, int> getIndexPair(const int i) const
            {
                return index_map_[i];
            }

            std::vector<int> getIndexVector(const int vec_i) const
            {
                std::vector<int> output;
                for(size_t i = 0; i < index_map_.size(); ++i)
                {
                    if(index_map_[i].first == (size_t)vec_i) output.push_back(i);
                }
                return output;
            }

            template <class U>
            std::vector<U> getVector(const std::vector<U>& data, const int vec_i) const
            {
                std::vector<U> output;
                for(size_t i = 0; i < index_map_.size(); ++i)
                {
                    if(index_map_[i].first == (size_t)(vec_i)) output.push_back(data[i]);
                }
                return output;
            }


            // Get the 1D index corresponding to the 2D indices
            size_t getIndex(const size_t i, const size_t j) const
            {
                for(size_t k = 0; k < index_map_.size(); ++k)
                {
                    if((index_map_[k].first == i) && (index_map_[k].second == j))
                    {
                        return k;
                    }
                }
                return -1;
            }

            // Get from the 2D indices in a 1D structure
            template <class U>
            U get(const int i, const int j, const std::vector<U>& data) const
            {
                for(size_t k = 0; k < index_map_.size(); ++k)
                {
                    if(((int)(index_map_[k].first) == i) && ((int)(index_map_[k].second) == j))
                    {
                        return data[k];
                    }
                }
                throw std::range_error("SortIndexTracker2: trying to 'get' with indices not present in the index_map");
            }


            // Get from the 2D indices in a 1D structure
            template <class U>
            std::vector<U> applySort(const std::vector<std::vector<U> >& data) const
            {
                std::vector<U> output;
                for(size_t k = 0; k < index_map_.size(); ++k)
                {
                    output.push_back(data[index_map_[k].first][index_map_[k].second]);
                }
                return output;
            }


            T back() const
            {
                return (*data_)[index_map_.back().first][index_map_.back().second];
            }

            size_t size() const
            {
                return index_map_.size();
            }
            
            T getSmallestGap()
            {
                T diff = get(1) - get(0);
                for(size_t i = 1; i < (size()-1); ++i)
                {
                    diff = get(i+1) - get(i);
                }
                return diff;
            }

        private:
            std::vector<std::vector<T> >* data_;

            std::vector<std::pair<size_t, size_t> > index_map_;
            

    };

}

#endif

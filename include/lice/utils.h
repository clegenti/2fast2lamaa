/**
 *  Author: Cedric LE GENTIL 
 *
 *  Copyright 2021 Cedric LE GENTIL
 *
 *  For any further question, recommendation or contribution
 *  le.gentil.cedric@gmail.com
 **/
#pragma once

#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <ctime>
#include <chrono>

#include <numeric>

class StopWatch
{
    public:
        StopWatch()
        {
            duration_ = std::chrono::high_resolution_clock::duration::zero();
        }

        //Start the counting (also used to get time while running);
        double start()
        {
            double output = 0.0;
            auto temp = std::chrono::high_resolution_clock::now();
            if(!stopped_)
            {
                duration_ += (temp - last_);
                output = std::chrono::duration_cast<std::chrono::nanoseconds>(duration_).count();
            }
            last_ = temp;
            stopped_ = false;
            return output/1000000.0;
        }

        // Since the first "start after the last reset"
        double getTotal()
        {
            auto temp = std::chrono::high_resolution_clock::now();
            return std::chrono::duration_cast<std::chrono::nanoseconds>(duration_ + (temp - last_)).count()/1000000.0;
        }

        // Since the 
        double getLast()
        {
            auto temp = std::chrono::high_resolution_clock::now();
            return std::chrono::duration_cast<std::chrono::nanoseconds>(temp - last_).count()/1000000.0;

        }

        double stop()
        {
            if(!stopped_)
            {
                auto temp = std::chrono::high_resolution_clock::now();
                stopped_ = true;
                duration_ += (temp - last_);
                return std::chrono::duration_cast<std::chrono::nanoseconds>(duration_).count()/1000000.0;
            }
            else
            {
                std::cout << "WARNING: Stopping a StopWatch that is not running, returning negative time" << std::endl;
                return -1;
            }
        }

        void reset()
        {
            stopped_ = true;
            duration_ = std::chrono::high_resolution_clock::duration::zero();
        }


        void print()
        {
            print("Time elapsed:");
        }

        void print(std::string str)
        {
            std::cout << str << " " << std::chrono::duration_cast<std::chrono::nanoseconds>(duration_).count()/1000000.0 << " ms" << std::endl;
        }

    private:
        double counter_ = 0;
        bool stopped_ = true;
        std::chrono::high_resolution_clock::duration duration_;
        std::chrono::high_resolution_clock::time_point last_;
};



// Gerenate random indexes between start and end (excluded) without repetition
inline std::vector<int> generateRandomIndexes(int start, int end, int nb)
{
    std::vector<int> output;
    if(nb > end - start)
    {
        std::cout << "ERROR: generateRandomIndexes: nb > end - start" << std::endl;
        return output;
    }
    std::vector<int> indexes;
    for(int i = start; i < end; i++)
    {
        indexes.push_back(i);
    }

    auto rng = std::default_random_engine {};
    std::shuffle(indexes.begin(), indexes.end(), rng);
    for(int i = 0; i < nb; i++)
    {
        output.push_back(indexes[i]);
    }
    return output;
}





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



// Function to split a string and return a vector of doubles
inline std::vector<float> splitString(std::string str, std::string delimiter)
{
    std::vector<float> output;
    size_t pos = 0;
    std::string token;
    while ((pos = str.find(delimiter)) != std::string::npos)
    {
        try
        {
            token = str.substr(0, pos);
            output.push_back(std::stof(token));
            str.erase(0, pos + delimiter.length());
        }
        catch(const std::exception& e)
        {
            std::cout << "ERROR: splitString: " << e.what() << std::endl;
            return output;
        }
    }
    try
    {
        output.push_back(std::stof(str));
    }
    catch(const std::exception& e)
    {
        std::cout << "ERROR: splitString: " << e.what() << std::endl;
        return output;
    }
    return output;
}


// Function to print a vector
template <typename T>
void printVector(std::vector<T> vec)
{
    std::cout << "[";
    for(int i = 0; i < vec.size(); i++)
    {
        std::cout << vec[i];
        if(i < vec.size() - 1)
        {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
}


inline double nanosToSeconds(const int64_t nanos, const int64_t time_offset)
{
    return (nanos - time_offset) * 1e-9;
}
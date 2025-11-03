#pragma once

#include <opencv2/opencv.hpp>

namespace byte_track
{

class KalmanFilter
{
public:
    using DetectBox = cv::Matx<float, 1, 4>;
    using StateMean = cv::Matx<float, 1, 8>;
    using StateCov = cv::Matx<float, 8, 8>;
    using StateHMean = cv::Matx<float, 1, 4>;
    using StateHCov = cv::Matx<float, 4, 4>;

    KalmanFilter(const float& std_weight_position = 1.0f / 20,
                 const float& std_weight_velocity = 1.0f / 160);

    void initiate(StateMean& mean, StateCov& covariance, const DetectBox& measurement);
    void predict(StateMean& mean, StateCov& covariance);
    void update(StateMean& mean, StateCov& covariance, const DetectBox& measurement);

private:
    float std_weight_position_;
    float std_weight_velocity_;

    cv::Matx<float, 8, 8> motion_mat_;
    cv::Matx<float, 4, 8> update_mat_;

    void project(StateHMean& projected_mean, StateHCov& projected_covariance,
                 const StateMean& mean, const StateCov& covariance);
};
}

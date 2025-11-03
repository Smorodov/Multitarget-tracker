#include "KalmanFilter.h"

namespace byte_track
{

KalmanFilter::KalmanFilter(const float& std_weight_position,
                           const float& std_weight_velocity) :
    std_weight_position_(std_weight_position),
    std_weight_velocity_(std_weight_velocity)
{
    constexpr size_t ndim = 4;
    constexpr float dt = 1.0f;

    motion_mat_ = cv::Matx<float, 8, 8>::eye();
    update_mat_ = cv::Matx<float, 4, 8>::eye();

    for (size_t i = 0; i < ndim; i++)
    {
        motion_mat_(i, ndim + i) = dt;
    }
}

void KalmanFilter::initiate(StateMean& mean, StateCov& covariance, const DetectBox& measurement)
{
    for (int i = 0; i < 4; i++)
    {
        mean(0, i) = measurement(0, i);
        mean(0, i + 4) = 0.0f;
    }

	StateMean std(
		2 * std_weight_position_ * measurement(3),
		2 * std_weight_position_ * measurement(3),
		1e-2f,
		2 * std_weight_position_ * measurement(3),
		10 * std_weight_velocity_ * measurement(3),
		10 * std_weight_velocity_ * measurement(3),
		1e-5f,
		10 * std_weight_velocity_ * measurement(3));

    covariance = StateCov::zeros();
    for (int i = 0; i < 8; i++)
    {
        covariance(i, i) = std(i) * std(i);
    }
}

void KalmanFilter::predict(StateMean& mean, StateCov& covariance)
{
	StateMean std(
		std_weight_position_ * mean(3),
		std_weight_position_ * mean(3),
		1e-2f,
		std_weight_position_ * mean(3),
		std_weight_velocity_ * mean(3),
		std_weight_velocity_ * mean(3),
		1e-5f,
		std_weight_velocity_ * mean(3));

    StateCov motion_cov = StateCov::zeros();
    for (int i = 0; i < 8; i++)
    {
        motion_cov(i, i) = std(i) * std(i);
    }

    StateMean new_mean = mean * motion_mat_.t();
    mean = new_mean;
    
    covariance = motion_mat_ * covariance * motion_mat_.t() + motion_cov;
}

void KalmanFilter::update(StateMean& mean, StateCov& covariance, const DetectBox& measurement)
{
    StateHMean projected_mean;
    StateHCov projected_cov;
    project(projected_mean, projected_cov, mean, covariance);

    cv::Matx<float, 4, 8> B = (covariance * update_mat_.t()).t();

    cv::Matx<float, 4, 8> kalman_gain;
    cv::solve(projected_cov, B, kalman_gain, cv::DECOMP_CHOLESKY);

    StateHMean innovation = measurement - projected_mean;

    StateMean tmp = innovation * kalman_gain;
    mean = mean + tmp;
    covariance = covariance - kalman_gain.t() * projected_cov * kalman_gain;
}

void KalmanFilter::project(StateHMean& projected_mean, StateHCov& projected_covariance,
                           const StateMean& mean, const StateCov& covariance)
{
	DetectBox std(
		std_weight_position_ * mean(3),
		std_weight_position_ * mean(3),
		1e-1f,
		std_weight_position_ * mean(3));

    projected_mean = (update_mat_ * mean.t()).t();
    projected_covariance = update_mat_ * covariance * update_mat_.t();

    for (int i = 0; i < 4; i++)
    {
        projected_covariance(i, i) += std(i) * std(i);
    }
}

}

#include "Kalman.h"
#include "opencv2/opencv.hpp"
#include <iostream>
#include <vector>

//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
TKalmanFilter::TKalmanFilter(
        Point_t pt,
        track_t deltaTime, // time increment (lower values makes target more "massive")
        track_t accelNoiseMag
        )
{
	// We don't know acceleration, so, assume it to process noise.
	// But we can guess, the range of acceleration values thich can be achieved by tracked object. 
    // Process noise. (standard deviation of acceleration: m/s^2)
	// shows, woh much target can accelerate.
    //track_t accelNoiseMag = 0.5;

	//4 state variables, 2 measurements
    kalman = new cv::KalmanFilter(4, 2, 0);
	// Transition cv::Matrix
    kalman->transitionMatrix = (cv::Mat_<track_t>(4, 4) << 1, 0, deltaTime, 0, 0, 1, 0, deltaTime, 0, 0, 1, 0, 0, 0, 0, 1);

	// init... 
    lastPointResult = pt;
	kalman->statePre.at<track_t>(0) = pt.x; // x
	kalman->statePre.at<track_t>(1) = pt.y; // y

	kalman->statePre.at<track_t>(2) = 0;
	kalman->statePre.at<track_t>(3) = 0;

	kalman->statePost.at<track_t>(0) = pt.x;
	kalman->statePost.at<track_t>(1) = pt.y;

	cv::setIdentity(kalman->measurementMatrix);

	kalman->processNoiseCov = (cv::Mat_<track_t>(4, 4) <<
        pow(deltaTime,4.0)/4.0	,0						,pow(deltaTime,3.0)/2.0		,0,
        0						,pow(deltaTime,4.0)/4.0	,0							,pow(deltaTime,3.0)/2.0,
        pow(deltaTime,3.0)/2.0	,0						,pow(deltaTime,2.0)			,0,
        0						,pow(deltaTime,3.0)/2.0	,0							,pow(deltaTime,2.0));


    kalman->processNoiseCov *= accelNoiseMag;

	setIdentity(kalman->measurementNoiseCov, cv::Scalar::all(0.1));

	setIdentity(kalman->errorCovPost, cv::Scalar::all(.1));
}

//---------------------------------------------------------------------------
TKalmanFilter::TKalmanFilter(
        cv::Rect rect,
        track_t deltaTime, // time increment (lower values makes target more "massive")
        track_t accelNoiseMag
        )
{
    // We don't know acceleration, so, assume it to process noise.
    // But we can guess, the range of acceleration values thich can be achieved by tracked object.
    // Process noise. (standard deviation of acceleration: m/s^2)
    // shows, woh much target can accelerate.
    //track_t accelNoiseMag = 0.5;

    //4 state variables (x, y, dx, dy, width, height), 4 measurements (x, y, width, height)
    kalman = new cv::KalmanFilter(6, 4, 0);
    // Transition cv::Matrix
    kalman->transitionMatrix = (cv::Mat_<track_t>(6, 6) <<
                                1, 0, 0, 0, deltaTime, 0,
                                0, 1, 0, 0, 0,         deltaTime,
                                0, 0, 1, 0, 0,         0,
                                0, 0, 0, 1, 0,         0,
                                0, 0, 0, 0, 1,         0,
                                0, 0, 0, 0, 0,         1);

    // init...
    lastRectResult = rect;
	kalman->statePre.at<track_t>(0) = static_cast<track_t>(rect.x);      // x
	kalman->statePre.at<track_t>(1) = static_cast<track_t>(rect.y);      // y
	kalman->statePre.at<track_t>(2) = static_cast<track_t>(rect.width);  // width
	kalman->statePre.at<track_t>(3) = static_cast<track_t>(rect.height); // height
    kalman->statePre.at<track_t>(4) = 0;                                 // dx
    kalman->statePre.at<track_t>(5) = 0;                                 // dy

	kalman->statePost.at<track_t>(0) = static_cast<track_t>(rect.x);
	kalman->statePost.at<track_t>(1) = static_cast<track_t>(rect.y);
	kalman->statePost.at<track_t>(2) = static_cast<track_t>(rect.width);
	kalman->statePost.at<track_t>(3) = static_cast<track_t>(rect.height);

    cv::setIdentity(kalman->measurementMatrix);

    kalman->processNoiseCov = (cv::Mat_<track_t>(6, 6) <<
        pow(deltaTime,4.)/4., 0,                    0,                    0,                    pow(deltaTime,3.)/2., 0,
        0,                    pow(deltaTime,4.)/4., 0,                    0,                    pow(deltaTime,3.)/2., 0,
        0,                    0,                    pow(deltaTime,4.)/4., 0,                    0,                    0,
        0,                    0,                    0,                    pow(deltaTime,4.)/4., 0,                    0,
        pow(deltaTime,3.)/2., 0,                    0,                    0,                    pow(deltaTime,2.),    0,
        0,                    pow(deltaTime,3.)/2., 0,                    0,                    0,                    pow(deltaTime,2.));


    kalman->processNoiseCov *= accelNoiseMag;

    setIdentity(kalman->measurementNoiseCov, cv::Scalar::all(0.1));

    setIdentity(kalman->errorCovPost, cv::Scalar::all(.1));
}

//---------------------------------------------------------------------------
TKalmanFilter::~TKalmanFilter()
{
	delete kalman;
}

//---------------------------------------------------------------------------
Point_t TKalmanFilter::GetPointPrediction()
{
	cv::Mat prediction = kalman->predict();
    lastPointResult = Point_t(prediction.at<track_t>(0), prediction.at<track_t>(1));
    return lastPointResult;
}

//---------------------------------------------------------------------------
Point_t TKalmanFilter::Update(Point_t p, bool dataCorrect)
{
	cv::Mat measurement(2, 1, Mat_t(1));
    if (!dataCorrect)
	{
        measurement.at<track_t>(0) = lastPointResult.x;  //update using prediction
        measurement.at<track_t>(1) = lastPointResult.y;
	}
	else
	{
		measurement.at<track_t>(0) = p.x;  //update using measurements
		measurement.at<track_t>(1) = p.y;
	}
	// Correction
	cv::Mat estiMated = kalman->correct(measurement);
    lastPointResult.x = estiMated.at<track_t>(0);   //update using measurements
    lastPointResult.y = estiMated.at<track_t>(1);

    return lastPointResult;
}
//---------------------------------------------------------------------------

cv::Rect TKalmanFilter::GetRectPrediction()
{
    cv::Mat prediction = kalman->predict();
    lastRectResult = cv::Rect_<track_t>(prediction.at<track_t>(0), prediction.at<track_t>(1), prediction.at<track_t>(2), prediction.at<track_t>(3));
	return cv::Rect(static_cast<int>(lastRectResult.x), static_cast<int>(lastRectResult.y), static_cast<int>(lastRectResult.width), static_cast<int>(lastRectResult.height));
}

//---------------------------------------------------------------------------
cv::Rect TKalmanFilter::Update(cv::Rect rect, bool dataCorrect)
{
    cv::Mat measurement(4, 1, Mat_t(1));
    if (!dataCorrect)
    {
        measurement.at<track_t>(0) = lastRectResult.x;  // update using prediction
        measurement.at<track_t>(1) = lastRectResult.y;
        measurement.at<track_t>(2) = lastRectResult.width;
        measurement.at<track_t>(3) = lastRectResult.height;
    }
    else
    {
		measurement.at<track_t>(0) = static_cast<track_t>(rect.x);  // update using measurements
		measurement.at<track_t>(1) = static_cast<track_t>(rect.y);
		measurement.at<track_t>(2) = static_cast<track_t>(rect.width);
		measurement.at<track_t>(3) = static_cast<track_t>(rect.height);
    }
    // Correction
    cv::Mat estiMated = kalman->correct(measurement);
    lastRectResult.x = estiMated.at<track_t>(0);   //update using measurements
    lastRectResult.y = estiMated.at<track_t>(1);
    lastRectResult.width = estiMated.at<track_t>(2);
    lastRectResult.height = estiMated.at<track_t>(3);

	return cv::Rect(static_cast<int>(lastRectResult.x), static_cast<int>(lastRectResult.y), static_cast<int>(lastRectResult.width), static_cast<int>(lastRectResult.height));
}
//---------------------------------------------------------------------------


///
/// \brief The BallisticModel class
/// In this two tests Unscented Kalman Filter are applied to the dynamic system from example "The reentry problem" from
/// "A New Extension of the Kalman Filter to Nonlinear Systems" by Simon J. Julier and Jeffrey K. Uhlmann.
///
class BallisticModel: public cv::tracking::UkfSystemModel
{
    static const double step;

    cv::Mat diff_eq(const cv::Mat& x)
    {
        double x1 = x.at<double>(0, 0);
        double x2 = x.at<double>(1, 0);
        double x3 = x.at<double>(2, 0);
        double x4 = x.at<double>(3, 0);
        double x5 = x.at<double>(4, 0);

        const double h0 = 9.3;
        const double beta0 = 0.59783;
        const double Gm = 3.9860044 * 1e5;
        const double r_e = 6374;

        const double r = sqrt( x1*x1 + x2*x2 );
        const double v = sqrt( x3*x3 + x4*x4 );
        const double d = - beta0 * exp( ( r_e - r )/h0 ) * exp( x5 ) * v;
        const double g = - Gm / (r*r*r);

        cv::Mat fx = x.clone();

        fx.at<double>(0, 0) = x3;
        fx.at<double>(1, 0) = x4;
        fx.at<double>(2, 0) = d * x3 + g * x1;
        fx.at<double>(3, 0) = d * x4 + g * x2;
        fx.at<double>(4, 0) = 0.0;

        return fx;
    }
public:
    void stateConversionFunction(const cv::Mat& x_k, const cv::Mat& u_k, const cv::Mat& v_k, cv::Mat& x_kplus1)
    {
        cv::Mat v = sqrt(step) * v_k.clone();
        v.at<double>(0, 0) = 0.0;
        v.at<double>(1, 0) = 0.0;

        cv::Mat k1 = diff_eq( x_k ) + v;
        cv::Mat tmp = x_k + step*0.5*k1;
        cv::Mat k2 = diff_eq( tmp ) + v;
        tmp = x_k + step*0.5*k2;
        cv::Mat k3 = diff_eq( tmp ) + v;
        tmp = x_k + step*k3;
        cv::Mat k4 = diff_eq( tmp ) + v;

        x_kplus1 = x_k + (1.0/6.0)*step*( k1 + 2.0*k2 + 2.0*k3 + k4 ) + u_k;
    }

    void measurementFunction(const cv::Mat& x_k, const cv::Mat& n_k, cv::Mat& z_k)
    {
        double x1 = x_k.at<double>(0, 0);
        double x2 = x_k.at<double>(1, 0);
        double x1_r = 6374.0;
        double x2_r = 0.0;

        double R = sqrt( pow( x1 - x1_r, 2 ) + pow( x2 - x2_r, 2 ) );
        double Phi = atan( (x2 - x2_r)/(x1 - x1_r) );

        R += n_k.at<double>(0, 0);
        Phi += n_k.at<double>(1, 0);

        z_k.at<double>(0, 0) = R;
        z_k.at<double>(1, 0) = Phi;
    }
};
const double BallisticModel::step = 0.05;

///
/// \brief The UnivariateNonstationaryGrowthModel class
/// In this test Unscented Kalman Filter are applied to the univariate nonstationary growth model (UNGM).
/// This model was used in example from "Unscented Kalman filtering for additive noise case: Augmented vs. non-augmented"
/// by Yuanxin Wu and Dewen Hu.
///
class UnivariateNonstationaryGrowthModel: public cv::tracking::UkfSystemModel
{

public:
    void stateConversionFunction(const cv::Mat& x_k, const cv::Mat& u_k, const cv::Mat& v_k, cv::Mat& x_kplus1)
    {
        double x = x_k.at<double>(0, 0);
        double n = u_k.at<double>(0, 0);
        double q = v_k.at<double>(0, 0);
        double u = u_k.at<double>(0, 0);

        double x1 = 0.5*x + 25*( x/(x*x + 1) ) + 8*cos( 1.2*(n-1) ) + q + u;
        x_kplus1.at<double>(0, 0) = x1;
    }
    void measurementFunction(const cv::Mat& x_k, const cv::Mat& n_k, cv::Mat& z_k)
    {
        double x = x_k.at<double>(0, 0);
        double r = n_k.at<double>(0, 0);

        double y = x*x/20.0 + r;
        z_k.at<double>(0, 0) = y;
    }
};

///
/// \brief TKalmanFilter::SomeFunction
///
void TKalmanFilter::SomeFunction()
{
    const double alpha = 1;
    const double beta = 2.0;
    const double kappa = -2.0;

    int MP = 2;
    int DP = 5;
    int CP = 0;

    cv::Mat processNoiseCov = cv::Mat::zeros( DP, DP, Mat_t(1) );
    processNoiseCov.at<double>(0, 0) = 1e-14;
    processNoiseCov.at<double>(1, 1) = 1e-14;
    processNoiseCov.at<double>(2, 2) = 2.4065 * 1e-5;
    processNoiseCov.at<double>(3, 3) = 2.4065 * 1e-5;
    processNoiseCov.at<double>(4, 4) = 1e-6;
    cv::Mat processNoiseCovSqrt = cv::Mat::zeros( DP, DP, Mat_t(1) );
    sqrt( processNoiseCov, processNoiseCovSqrt );

    cv::Mat measurementNoiseCov = cv::Mat::zeros( MP, MP, Mat_t(1) );
    measurementNoiseCov.at<double>(0, 0) = 1e-3*1e-3;
    measurementNoiseCov.at<double>(1, 1) = 0.13*0.13;
    cv::Mat measurementNoiseCovSqrt = cv::Mat::zeros( MP, MP, Mat_t(1) );
    sqrt( measurementNoiseCov, measurementNoiseCovSqrt );

    cv::RNG rng( 117 );

    cv::Mat state( DP, 1, Mat_t(1) );
    state.at<double>(0, 0) = 6500.4;
    state.at<double>(1, 0) = 349.14;
    state.at<double>(2, 0) = -1.8093;
    state.at<double>(3, 0) = -6.7967;
    state.at<double>(4, 0) = 0.6932;

    cv::Mat initState = state.clone();
    initState.at<double>(4, 0) = 0.0;

    cv::Mat P = 1e-6 * cv::Mat::eye( DP, DP, Mat_t(1) );
    P.at<double>(4, 4) = 1.0;

    cv::Mat measurement( MP, 1, Mat_t(1) );

    cv::Mat q( DP, 1, Mat_t(1) );
    cv::Mat r( MP, 1, Mat_t(1) );

    cv::Ptr<BallisticModel> model( new BallisticModel() );
    cv::tracking::AugmentedUnscentedKalmanFilterParams params( DP, MP, CP, 0, 0, model );

    params.stateInit = initState.clone();
    params.errorCovInit = P.clone();
    params.measurementNoiseCov = measurementNoiseCov.clone();
    params.processNoiseCov = processNoiseCov.clone();

    params.alpha = alpha;
    params.beta = beta;
    params.k = kappa;

    cv::Ptr<cv::tracking::UnscentedKalmanFilter> augmentedUncsentedKalmanFilter = cv::tracking::createAugmentedUnscentedKalmanFilter(params);

    cv::Mat correctStateUKF( DP, 1, Mat_t(1) );
    cv::Mat u = cv::Mat::zeros( DP, 1, Mat_t(1) );

#if 0
    for (int i = 0; i<nIterations; i++)
    {
        rng.fill( q, cv::RNG::NORMAL, cv::Scalar::all(0),  cv::Scalar::all(1) );
        q = processNoiseCovSqrt*q;

        rng.fill( r, cv::RNG::NORMAL, cv::Scalar::all(0), cv::Scalar::all(1) );
        r = measurementNoiseCovSqrt*r;

        model->stateConversionFunction(state, u, q, state);
        model->measurementFunction(state, r, measurement);

        augmentedUncsentedKalmanFilter->predict();
        correctStateUKF = augmentedUncsentedKalmanFilter->correct( measurement );
    }

    double landing_y = correctStateUKF.at<double>(1, 0);
    const double landing_coordinate = 2.5; // the expected landing coordinate
    ASSERT_NEAR(landing_coordinate, landing_y, abs_error);
#endif
}

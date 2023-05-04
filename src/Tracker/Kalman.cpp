#include "Kalman.h"
#include <iostream>
#include <vector>

#ifdef USE_OCV_UKF
#if (((CV_VERSION_MAJOR == 4) && (CV_VERSION_MINOR < 5)) || ((CV_VERSION_MAJOR == 4) && (CV_VERSION_MINOR == 5) && (CV_VERSION_REVISION < 1)) || (CV_VERSION_MAJOR == 3))
namespace kalman = cv::tracking;
#else
namespace kalman = cv::detail::tracking;
#endif
#endif

///
/// \brief TKalmanFilter::TKalmanFilter
/// \param type
/// \param useAcceleration
/// \param deltaTime
/// \param accelNoiseMag
///
TKalmanFilter::TKalmanFilter(tracking::KalmanType type,
                             bool useAcceleration,
                             track_t deltaTime, // time increment (lower values makes target more "massive")
                             track_t accelNoiseMag)
    :
      m_accelNoiseMag(accelNoiseMag),
      m_deltaTime(deltaTime),
      m_deltaTimeMin(deltaTime),
      m_deltaTimeMax(2 * deltaTime),
      m_type(type),
      m_useAcceleration(useAcceleration)
{
    m_deltaStep = (m_deltaTimeMax - m_deltaTimeMin) / m_deltaStepsCount;
}

///
/// \brief TKalmanFilter::CreateLinear
/// \param xy0
/// \param xyv0
///
void TKalmanFilter::CreateLinear(Point_t xy0, Point_t xyv0)
{
    // We don't know acceleration, so, assume it to process noise.
    // But we can guess, the range of acceleration values thich can be achieved by tracked object.
    // Process noise. (standard deviation of acceleration: m/s^2)
    // shows, woh much target can accelerate.

    // 4 state variables, 2 measurements
    m_linearKalman.init(4, 2, 0, El_t);
    // Transition cv::Matrix
    m_linearKalman.transitionMatrix = (cv::Mat_<track_t>(4, 4) <<
                                        1, 0, m_deltaTime, 0,
                                        0, 1, 0, m_deltaTime,
                                        0, 0, 1, 0,
                                        0, 0, 0, 1);

    // init...
    m_lastPointResult = xy0;
    m_linearKalman.statePre.at<track_t>(0) = xy0.x;  // x
    m_linearKalman.statePre.at<track_t>(1) = xy0.y;  // y
    m_linearKalman.statePre.at<track_t>(2) = xyv0.x; // vx
    m_linearKalman.statePre.at<track_t>(3) = xyv0.y; // vy

    m_linearKalman.statePost.at<track_t>(0) = xy0.x;
    m_linearKalman.statePost.at<track_t>(1) = xy0.y;
    m_linearKalman.statePost.at<track_t>(2) = xyv0.x;
    m_linearKalman.statePost.at<track_t>(3) = xyv0.y;

    cv::setIdentity(m_linearKalman.measurementMatrix);

    m_linearKalman.processNoiseCov = (cv::Mat_<track_t>(4, 4) <<
                                       pow(m_deltaTime,4.0)/4.0	,0						,pow(m_deltaTime,3.0)/2.0		,0,
                                       0						,pow(m_deltaTime,4.0)/4.0	,0							,pow(m_deltaTime,3.0)/2.0,
                                       pow(m_deltaTime,3.0)/2.0	,0						,pow(m_deltaTime,2.0)			,0,
                                       0						,pow(m_deltaTime,3.0)/2.0	,0							,pow(m_deltaTime,2.0));


    m_linearKalman.processNoiseCov *= m_accelNoiseMag;

    cv::setIdentity(m_linearKalman.measurementNoiseCov, cv::Scalar::all(0.1));

    cv::setIdentity(m_linearKalman.errorCovPost, cv::Scalar::all(.1));

    m_initialPoints.reserve(MIN_INIT_VALS);

    m_initialized = true;
}

///
/// \brief TKalmanFilter::CreateLinear
/// \param rect0
/// \param rectv0
///
void TKalmanFilter::CreateLinear(cv::Rect_<track_t> rect0, Point_t rectv0)
{
    // We don't know acceleration, so, assume it to process noise.
    // But we can guess, the range of acceleration values thich can be achieved by tracked object.
    // Process noise. (standard deviation of acceleration: m/s^2)
    // shows, woh much target can accelerate.

    // 8 state variables (x, y, vx, vy, width, height, vw, vh), 4 measurements (x, y, width, height)
    m_linearKalman.init(8, 4, 0, El_t);
    // Transition cv::Matrix
    m_linearKalman.transitionMatrix = (cv::Mat_<track_t>(8, 8) <<
                                        1, 0, 0, 0, m_deltaTime, 0,           0,           0,
                                        0, 1, 0, 0, 0,           m_deltaTime, 0,           0,
                                        0, 0, 1, 0, 0,           0,           m_deltaTime, 0,
                                        0, 0, 0, 1, 0,           0,           0,           m_deltaTime / 10.f,
                                        0, 0, 0, 0, 1,           0,           0,           0,
                                        0, 0, 0, 0, 0,           1,           0,           0,
                                        0, 0, 0, 0, 0,           0,           1,           0,
                                        0, 0, 0, 0, 0,           0,           0,           1);

    // init...
    m_linearKalman.statePre.at<track_t>(0) = rect0.x;      // x
    m_linearKalman.statePre.at<track_t>(1) = rect0.y;      // y
    m_linearKalman.statePre.at<track_t>(2) = rect0.width;  // width
    m_linearKalman.statePre.at<track_t>(3) = rect0.height; // height
    m_linearKalman.statePre.at<track_t>(4) = rectv0.x;     // dx
    m_linearKalman.statePre.at<track_t>(5) = rectv0.y;     // dy
    m_linearKalman.statePre.at<track_t>(6) = 0;            // dw
    m_linearKalman.statePre.at<track_t>(7) = 0;            // dh

    m_linearKalman.statePost.at<track_t>(0) = rect0.x;
    m_linearKalman.statePost.at<track_t>(1) = rect0.y;
    m_linearKalman.statePost.at<track_t>(2) = rect0.width;
    m_linearKalman.statePost.at<track_t>(3) = rect0.height;
    m_linearKalman.statePost.at<track_t>(4) = rectv0.x;
    m_linearKalman.statePost.at<track_t>(5) = rectv0.y;
    m_linearKalman.statePost.at<track_t>(6) = 0;
    m_linearKalman.statePost.at<track_t>(7) = 0;

    cv::setIdentity(m_linearKalman.measurementMatrix);

    track_t n1 = pow(m_deltaTime, 4.f) / 4.f;
    track_t n2 = pow(m_deltaTime, 3.f) / 2.f;
    track_t n3 = pow(m_deltaTime, 2.f);
    m_linearKalman.processNoiseCov = (cv::Mat_<track_t>(8, 8) <<
                                       n1, 0,  0,  0,  n2, 0,  0,  0,
                                       0,  n1, 0,  0,  0,  n2, 0,  0,
                                       0,  0,  n1, 0,  0,  0,  n2, 0,
                                       0,  0,  0,  n1, 0,  0,  0,  n2,
                                       n2, 0,  0,  0,  n3, 0,  0,  0,
                                       0,  n2, 0,  0,  0,  n3, 0,  0,
                                       0,  0,  n2, 0,  0,  0,  n3, 0,
                                       0,  0,  0,  n2, 0,  0,  0,  n3);

    m_linearKalman.processNoiseCov *= m_accelNoiseMag;

    cv::setIdentity(m_linearKalman.measurementNoiseCov, cv::Scalar::all(0.1));

    cv::setIdentity(m_linearKalman.errorCovPost, cv::Scalar::all(.1));

    m_initialRects.reserve(MIN_INIT_VALS);

    m_initialized = true;
}

///
/// \brief TKalmanFilter::CreateLinear
/// \param rrect0
/// \param rrectv0
///
void TKalmanFilter::CreateLinear(cv::RotatedRect rrect0, Point_t rrectv0)
{
	// We don't know acceleration, so, assume it to process noise.
	// But we can guess, the range of acceleration values thich can be achieved by tracked object.
	// Process noise. (standard deviation of acceleration: m/s^2)
	// shows, woh much target can accelerate.

	// 10 state variables (x, y, vx, vy, width, height, vw, vh, angle, vangle), 5 measurements (x, y, width, height, angle)
	m_linearKalman.init(10, 5, 0, El_t);
	// Transition cv::Matrix
	m_linearKalman.transitionMatrix = (cv::Mat_<track_t>(10, 10) <<
                                       1, 0, 0, 0, 0, m_deltaTime, 0,           0,           0,           0,
                                       0, 1, 0, 0, 0, 0,           m_deltaTime, 0,           0,           0,
                                       0, 0, 1, 0, 0, 0,           0,           m_deltaTime, 0,           0,
                                       0, 0, 0, 1, 0, 0,           0,           0,           m_deltaTime, 0,
                                       0, 0, 0, 0, 1, 0,           0,           0,           0,           m_deltaTime,
                                       0, 0, 0, 0, 0, 1,           0,           0,           0,           0,
                                       0, 0, 0, 0, 0, 0,           1,           0,           0,           0,
                                       0, 0, 0, 0, 0, 0,           0,           1,           0,           0,
                                       0, 0, 0, 0, 0, 0,           0,           0,           1,           0,
                                       0, 0, 0, 0, 0, 0,           0,           0,           0,           1);
	// init...
	m_linearKalman.statePre.at<track_t>(0) = rrect0.center.x;      // x
	m_linearKalman.statePre.at<track_t>(1) = rrect0.center.y;      // y
	m_linearKalman.statePre.at<track_t>(2) = rrect0.size.width;    // width
	m_linearKalman.statePre.at<track_t>(3) = rrect0.size.height;   // height
	m_linearKalman.statePre.at<track_t>(4) = rrect0.angle;         // angle
	m_linearKalman.statePre.at<track_t>(5) = rrectv0.x;            // dx
	m_linearKalman.statePre.at<track_t>(6) = rrectv0.y;            // dy
	m_linearKalman.statePre.at<track_t>(7) = 0;                    // dw
	m_linearKalman.statePre.at<track_t>(8) = 0;                    // dh
	m_linearKalman.statePre.at<track_t>(9) = 0;                    // da

	m_linearKalman.statePost.at<track_t>(0) = rrect0.center.x;
	m_linearKalman.statePost.at<track_t>(1) = rrect0.center.y;
	m_linearKalman.statePost.at<track_t>(2) = rrect0.size.width;
	m_linearKalman.statePost.at<track_t>(3) = rrect0.size.height;
	m_linearKalman.statePost.at<track_t>(4) = rrect0.angle;
	m_linearKalman.statePost.at<track_t>(5) = rrectv0.x;
	m_linearKalman.statePost.at<track_t>(6) = rrectv0.y;
	m_linearKalman.statePost.at<track_t>(7) = 0;
	m_linearKalman.statePost.at<track_t>(8) = 0;
	m_linearKalman.statePost.at<track_t>(9) = 0;

	cv::setIdentity(m_linearKalman.measurementMatrix);

	track_t n1 = pow(m_deltaTime, 4.f) / 4.f;
	track_t n2 = pow(m_deltaTime, 3.f) / 2.f;
	track_t n3 = pow(m_deltaTime, 2.f);
	m_linearKalman.processNoiseCov = (cv::Mat_<track_t>(10, 10) <<
		n1, 0,  0,  0,  0,  n2, 0,  0,  0,  0,
		0,  n1, 0,  0,  0,  0,  n2, 0,  0,  0,
		0,  0,  n1, 0,  0,  0,  0,  n2, 0,  0,
		0,  0,  0,  n1, 0,  0,  0,  0,  n2, 0,
		0,  0,  0,  0,  n1, 0,  0,  0,  0,  n2,
		n2, 0,  0,  0,  0,  n3, 0,  0,  0,	0,
		0,  n2, 0,  0,  0,  0,  n3, 0,  0,	0,
		0,  0,  n2, 0,  0,  0,  0,  n3, 0,	0,
		0,  0,  0,  n2, 0,  0,  0,  0,  n3, 0,
		0,  0,  0,  0,  n2, 0,  0,  0,  0,  n3);

	m_linearKalman.processNoiseCov *= m_accelNoiseMag;

	cv::setIdentity(m_linearKalman.measurementNoiseCov, cv::Scalar::all(0.1));

	cv::setIdentity(m_linearKalman.errorCovPost, cv::Scalar::all(.1));

	m_initialRects.reserve(MIN_INIT_VALS);

	m_initialized = true;
}

///
/// \brief TKalmanFilter::CreateLinearAcceleration
/// \param xy0
/// \param xyv0
///
void TKalmanFilter::CreateLinearAcceleration(Point_t xy0, Point_t xyv0)
{
    // 6 state variables, 2 measurements
    m_linearKalman.init(6, 2, 0, El_t);
    // Transition cv::Matrix
    const track_t dt = m_deltaTime;
    const track_t dt2 = 0.5f * m_deltaTime * m_deltaTime;
    m_linearKalman.transitionMatrix = (cv::Mat_<track_t>(6, 6) <<
                                       1, 0, dt, 0,  dt2, 0,
                                       0, 1, 0,  dt, 0,   dt2,
                                       0, 0, 1,  0,  dt,  0,
                                       0, 0, 0,  1,  0,   dt,
                                       0, 0, 0,  0,  1,   0,
                                       0, 0, 0,  0,  0,   1);

    // init...
    m_lastPointResult = xy0;
    m_linearKalman.statePre.at<track_t>(0) = xy0.x;  // x
    m_linearKalman.statePre.at<track_t>(1) = xy0.y;  // y
    m_linearKalman.statePre.at<track_t>(2) = xyv0.x; // vx
    m_linearKalman.statePre.at<track_t>(3) = xyv0.y; // vy
    m_linearKalman.statePre.at<track_t>(4) = 0;      // ax
    m_linearKalman.statePre.at<track_t>(5) = 0;      // ay

    m_linearKalman.statePost.at<track_t>(0) = xy0.x;
    m_linearKalman.statePost.at<track_t>(1) = xy0.y;
    m_linearKalman.statePost.at<track_t>(2) = xyv0.x;
    m_linearKalman.statePost.at<track_t>(3) = xyv0.y;
    m_linearKalman.statePost.at<track_t>(4) = 0;
    m_linearKalman.statePost.at<track_t>(5) = 0;

    cv::setIdentity(m_linearKalman.measurementMatrix);

    track_t n1 = pow(m_deltaTime, 4.f) / 4.f;
    track_t n2 = pow(m_deltaTime, 3.f) / 2.f;
    track_t n3 = pow(m_deltaTime, 2.f);
    m_linearKalman.processNoiseCov = (cv::Mat_<track_t>(6, 6) <<
                                      n1, 0,  n2, 0,  n2, 0,
                                      0,  n1, 0,  n2, 0,  n2,
                                      n2, 0,  n3, 0,  n3, 0,
                                      0,  n2, 0,  n3, 0,  n3,
                                      0,  0,  n2, 0,  n3, 0,
                                      0,  0,  0,  n2, 0,  n3);

    m_linearKalman.processNoiseCov *= m_accelNoiseMag;

    cv::setIdentity(m_linearKalman.measurementNoiseCov, cv::Scalar::all(0.1));

    cv::setIdentity(m_linearKalman.errorCovPost, cv::Scalar::all(.1));

    m_initialPoints.reserve(MIN_INIT_VALS);

    m_initialized = true;
}

///
/// \brief TKalmanFilter::CreateLinearAcceleration
/// \param rect0
/// \param rectv0
///
void TKalmanFilter::CreateLinearAcceleration(cv::Rect_<track_t> rect0, Point_t rectv0)
{
    // 12 state variables (x, y, vx, vy, ax, ay, width, height, vw, vh, aw, ah), 4 measurements (x, y, width, height)
    m_linearKalman.init(12, 4, 0, El_t);
    // Transition cv::Matrix
    const track_t dt = m_deltaTime;
    const track_t dt2 = 0.5f * m_deltaTime * m_deltaTime;
    m_linearKalman.transitionMatrix = (cv::Mat_<track_t>(12, 12) <<
                                       1, 0, 0, 0, dt, 0,  0,  0,  dt2, 0,   dt2, 0,
                                       0, 1, 0, 0, 0,  dt, 0,  0,  0,   dt2, 0,   dt2,
                                       0, 0, 1, 0, 0,  0,  dt, 0,  0,   0,   dt2, 0,
                                       0, 0, 0, 1, 0,  0,  0,  dt, 0,   0,   0,   dt2,
                                       0, 0, 0, 0, 1,  0,  0,  0,  dt,  0,   0,   0,
                                       0, 0, 0, 0, 0,  1,  0,  0,  0,   dt,  0,   0,
                                       0, 0, 0, 0, 0,  0,  1,  0,  0,   0,   dt,  0,
                                       0, 0, 0, 0, 0,  0,  0,  1,  0,   0,   0,   dt,
                                       0, 0, 0, 0, 0,  0,  0,  0,  1,   0,   0,   0,
                                       0, 0, 0, 0, 0,  0,  0,  0,  0,   1,   0,   0,
                                       0, 0, 0, 0, 0,  0,  0,  0,  0,   0,   1,   0,
                                       0, 0, 0, 0, 0,  0,  0,  0,  0,   0,   0,   1);

    // init...
    m_linearKalman.statePre.at<track_t>(0) = rect0.x;      // x
    m_linearKalman.statePre.at<track_t>(1) = rect0.y;      // y
    m_linearKalman.statePre.at<track_t>(2) = rect0.width;  // width
    m_linearKalman.statePre.at<track_t>(3) = rect0.height; // height
    m_linearKalman.statePre.at<track_t>(4) = rectv0.x;     // dx
    m_linearKalman.statePre.at<track_t>(5) = rectv0.y;     // dy
    m_linearKalman.statePre.at<track_t>(6) = 0;            // dw
    m_linearKalman.statePre.at<track_t>(7) = 0;            // dh
    m_linearKalman.statePre.at<track_t>(8) = 0;            // ax
    m_linearKalman.statePre.at<track_t>(9) = 0;            // ay
    m_linearKalman.statePre.at<track_t>(10) = 0;           // aw
    m_linearKalman.statePre.at<track_t>(11) = 0;           // ah

    m_linearKalman.statePost.at<track_t>(0) = rect0.x;
    m_linearKalman.statePost.at<track_t>(1) = rect0.y;
    m_linearKalman.statePost.at<track_t>(2) = rect0.width;
    m_linearKalman.statePost.at<track_t>(3) = rect0.height;
    m_linearKalman.statePost.at<track_t>(4) = rectv0.x;
    m_linearKalman.statePost.at<track_t>(5) = rectv0.y;
    m_linearKalman.statePost.at<track_t>(6) = 0;
    m_linearKalman.statePost.at<track_t>(7) = 0;
    m_linearKalman.statePost.at<track_t>(8) = 0;
    m_linearKalman.statePost.at<track_t>(9) = 0;
    m_linearKalman.statePost.at<track_t>(10) = 0;
    m_linearKalman.statePost.at<track_t>(11) = 0;

    cv::setIdentity(m_linearKalman.measurementMatrix);

    track_t n1 = pow(m_deltaTime, 4.f) / 4.f;
    track_t n2 = pow(m_deltaTime, 3.f) / 2.f;
    track_t n3 = pow(m_deltaTime, 2.f);
    m_linearKalman.processNoiseCov = (cv::Mat_<track_t>(12, 12) <<
                                      n1, 0,  0,  0,  n2, 0,  0,  0,  n2, 0,  n2, 0,
                                      0,  n1, 0,  0,  0,  n2, 0,  0,  0,  n2, 0,  n2,
                                      0,  0,  n1, 0,  0,  0,  n2, 0,  0,  0,  n2, 0,
                                      0,  0,  0,  n1, 0,  0,  0,  n2, 0,  0,  0,  n2,
                                      n2, 0,  0,  0,  n3, 0,  0,  0,  n3, 0,  n3, 0,
                                      0,  n2, 0,  0,  0,  n3, 0,  0,  0,  n3, 0,  n3,
                                      0,  0,  n2, 0,  0,  0,  n3, 0,  0,  0,  n3, 0,
                                      0,  0,  0,  n2, 0,  0,  0,  n3, 0,  0,  0,  n3,
                                      n2, 0,  0,  0,  n3, 0,  0,  0,  n3, 0,  0,  0,
                                      0,  n2, 0,  0,  0,  n3, 0,  0,  0,  n3, 0,  0,
                                      0,  0,  n2, 0,  0,  0,  n3, 0,  0,  0,  n3, 0,
                                      0,  0,  0,  n2, 0,  0,  0,  n3, 0,  0,  0,  n3);

    m_linearKalman.processNoiseCov *= m_accelNoiseMag;

    cv::setIdentity(m_linearKalman.measurementNoiseCov, cv::Scalar::all(0.1));

    cv::setIdentity(m_linearKalman.errorCovPost, cv::Scalar::all(.1));

    m_initialRects.reserve(MIN_INIT_VALS);

    m_initialized = true;
}

///
/// \brief TKalmanFilter::CreateLinearAcceleration
/// \param rect0
/// \param rectv0
///
void TKalmanFilter::CreateLinearAcceleration(cv::RotatedRect /*rrect0*/, Point_t /*rrectv0*/)
{
	// TODO
	assert(0);
}

#ifdef USE_OCV_UKF
//---------------------------------------------------------------------------
class AcceleratedModel final : public kalman::UkfSystemModel
{
public:
    AcceleratedModel(track_t deltaTime, bool rectModel)
        :
          kalman::UkfSystemModel(),
          m_deltaTime(deltaTime),
          m_rectModel(rectModel)
    {
    }

    void stateConversionFunction(const cv::Mat& x_k, const cv::Mat& u_k, const cv::Mat& v_k, cv::Mat& x_kplus1)
    {
        track_t x0 = x_k.at<track_t>(0, 0);
        track_t y0 = x_k.at<track_t>(1, 0);
        track_t vx0 = x_k.at<track_t>(2, 0);
        track_t vy0 = x_k.at<track_t>(3, 0);
        track_t ax0 = x_k.at<track_t>(4, 0);
        track_t ay0 = x_k.at<track_t>(5, 0);

        x_kplus1.at<track_t>(0, 0) = x0 + vx0 * m_deltaTime + ax0 * sqr(m_deltaTime) / 2;
        x_kplus1.at<track_t>(1, 0) = y0 + vy0 * m_deltaTime + ay0 * sqr(m_deltaTime) / 2;
        x_kplus1.at<track_t>(2, 0) = vx0 + ax0 * m_deltaTime;
        x_kplus1.at<track_t>(3, 0) = vy0 + ay0 * m_deltaTime;
        x_kplus1.at<track_t>(4, 0) = ax0;
        x_kplus1.at<track_t>(5, 0) = ay0;

        if (m_rectModel)
        {
            x_kplus1.at<track_t>(6, 0) = x_k.at<track_t>(6, 0);
            x_kplus1.at<track_t>(7, 0) = x_k.at<track_t>(7, 0);
        }

        if (v_k.size() == u_k.size())
            x_kplus1 += v_k + u_k;
        else
            x_kplus1 += v_k;
    }

    void measurementFunction(const cv::Mat& x_k, const cv::Mat& n_k, cv::Mat& z_k)
    {
        track_t x0 = x_k.at<track_t>(0, 0);
        track_t y0 = x_k.at<track_t>(1, 0);
        track_t vx0 = x_k.at<track_t>(2, 0);
        track_t vy0 = x_k.at<track_t>(3, 0);
        track_t ax0 = x_k.at<track_t>(4, 0);
        track_t ay0 = x_k.at<track_t>(5, 0);

        z_k.at<track_t>(0, 0) = x0 + vx0 * m_deltaTime + ax0 * sqr(m_deltaTime) / 2 + n_k.at<track_t>(0, 0);
        z_k.at<track_t>(1, 0) = y0 + vy0 * m_deltaTime + ay0 * sqr(m_deltaTime) / 2 + n_k.at<track_t>(1, 0);

        if (m_rectModel)
        {
            z_k.at<track_t>(2, 0) = x_k.at<track_t>(6, 0);
            z_k.at<track_t>(3, 0) = x_k.at<track_t>(7, 0);
        }
    }

private:
    track_t m_deltaTime;
    bool m_rectModel;
};

//---------------------------------------------------------------------------
void TKalmanFilter::CreateUnscented(Point_t xy0, Point_t xyv0)
{
    int MP = 2;
    int DP = 6;
    int CP = 0;

    cv::Mat processNoiseCov = cv::Mat::zeros(DP, DP, Mat_t(1));
    processNoiseCov.at<track_t>(0, 0) = 1e-14f;
    processNoiseCov.at<track_t>(1, 1) = 1e-14f;
    processNoiseCov.at<track_t>(2, 2) = 1e-6f;
    processNoiseCov.at<track_t>(3, 3) = 1e-6f;
    processNoiseCov.at<track_t>(4, 4) = 1e-6f;
    processNoiseCov.at<track_t>(5, 5) = 1e-6f;

    cv::Mat measurementNoiseCov = cv::Mat::zeros(MP, MP, Mat_t(1));
    measurementNoiseCov.at<track_t>(0, 0) = 1e-6f;
    measurementNoiseCov.at<track_t>(1, 1) = 1e-6f;

    cv::Mat initState(DP, 1, Mat_t(1));
    initState.at<track_t>(0, 0) = xy0.x;
    initState.at<track_t>(1, 0) = xy0.y;
    initState.at<track_t>(2, 0) = xyv0.x;
    initState.at<track_t>(3, 0) = xyv0.y;
    initState.at<track_t>(4, 0) = 0;
    initState.at<track_t>(5, 0) = 0;

    cv::Mat P = 1e-6f * cv::Mat::eye(DP, DP, Mat_t(1));

    cv::Ptr<AcceleratedModel> model(new AcceleratedModel(m_deltaTime, false));
    kalman::UnscentedKalmanFilterParams params(DP, MP, CP, 0, 0, model);
    params.dataType = Mat_t(1);
    params.stateInit = initState.clone();
    params.errorCovInit = P.clone();
    params.measurementNoiseCov = measurementNoiseCov.clone();
    params.processNoiseCov = processNoiseCov.clone();

    params.alpha = 1.0;
    params.beta = 2.0;
    params.k = -2.0;

    m_uncsentedKalman = kalman::createUnscentedKalmanFilter(params);

    m_initialized = true;
}

//---------------------------------------------------------------------------
void TKalmanFilter::CreateUnscented(cv::Rect_<track_t> rect0, Point_t rectv0)
{
    int MP = 4;
    int DP = 8;
    int CP = 0;

    cv::Mat processNoiseCov = cv::Mat::zeros(DP, DP, Mat_t(1));
    processNoiseCov.at<track_t>(0, 0) = 1e-3f;
    processNoiseCov.at<track_t>(1, 1) = 1e-3f;
    processNoiseCov.at<track_t>(2, 2) = 1e-3f;
    processNoiseCov.at<track_t>(3, 3) = 1e-3f;
    processNoiseCov.at<track_t>(4, 4) = 1e-3f;
    processNoiseCov.at<track_t>(5, 5) = 1e-3f;
    processNoiseCov.at<track_t>(6, 6) = 1e-3f;
    processNoiseCov.at<track_t>(7, 7) = 1e-3f;

    cv::Mat measurementNoiseCov = cv::Mat::zeros(MP, MP, Mat_t(1));
    measurementNoiseCov.at<track_t>(0, 0) = 1e-3f;
    measurementNoiseCov.at<track_t>(1, 1) = 1e-3f;
    measurementNoiseCov.at<track_t>(2, 2) = 1e-3f;
    measurementNoiseCov.at<track_t>(3, 3) = 1e-3f;

    cv::Mat initState(DP, 1, Mat_t(1));
    initState.at<track_t>(0, 0) = rect0.x;
    initState.at<track_t>(1, 0) = rect0.y;
    initState.at<track_t>(2, 0) = rectv0.x;
    initState.at<track_t>(3, 0) = rectv0.y;
    initState.at<track_t>(4, 0) = 0;
    initState.at<track_t>(5, 0) = 0;
    initState.at<track_t>(6, 0) = rect0.width;
    initState.at<track_t>(7, 0) = rect0.height;

    cv::Mat P = 1e-3f * cv::Mat::eye(DP, DP, Mat_t(1));

    cv::Ptr<AcceleratedModel> model(new AcceleratedModel(m_deltaTime, true));
    kalman::UnscentedKalmanFilterParams params(DP, MP, CP, 0, 0, model);
    params.dataType = Mat_t(1);
    params.stateInit = initState.clone();
    params.errorCovInit = P.clone();
    params.measurementNoiseCov = measurementNoiseCov.clone();
    params.processNoiseCov = processNoiseCov.clone();

    params.alpha = 1;
    params.beta = 2.0;
    params.k = -2.0;

    m_uncsentedKalman = kalman::createUnscentedKalmanFilter(params);

    m_initialized = true;
}

//---------------------------------------------------------------------------
void TKalmanFilter::CreateAugmentedUnscented(Point_t xy0, Point_t xyv0)
{
    int MP = 2;
    int DP = 6;
    int CP = 0;

    cv::Mat processNoiseCov = cv::Mat::zeros(DP, DP, Mat_t(1));
    processNoiseCov.at<track_t>(0, 0) = 1e-14f;
    processNoiseCov.at<track_t>(1, 1) = 1e-14f;
    processNoiseCov.at<track_t>(2, 2) = 1e-6f;
    processNoiseCov.at<track_t>(3, 3) = 1e-6f;
    processNoiseCov.at<track_t>(4, 4) = 1e-6f;
    processNoiseCov.at<track_t>(5, 5) = 1e-6f;

    cv::Mat measurementNoiseCov = cv::Mat::zeros(MP, MP, Mat_t(1));
    measurementNoiseCov.at<track_t>(0, 0) = 1e-6f;
    measurementNoiseCov.at<track_t>(1, 1) = 1e-6f;

    cv::Mat initState(DP, 1, Mat_t(1));
    initState.at<track_t>(0, 0) = xy0.x;
    initState.at<track_t>(1, 0) = xy0.y;
    initState.at<track_t>(2, 0) = xyv0.x;
    initState.at<track_t>(3, 0) = xyv0.y;
    initState.at<track_t>(4, 0) = 0;
    initState.at<track_t>(5, 0) = 0;

    cv::Mat P = 1e-6f * cv::Mat::eye(DP, DP, Mat_t(1));

    cv::Ptr<AcceleratedModel> model(new AcceleratedModel(m_deltaTime, false));
    kalman::AugmentedUnscentedKalmanFilterParams params(DP, MP, CP, 0, 0, model);
    params.dataType = Mat_t(1);
    params.stateInit = initState.clone();
    params.errorCovInit = P.clone();
    params.measurementNoiseCov = measurementNoiseCov.clone();
    params.processNoiseCov = processNoiseCov.clone();

    params.alpha = 1;
    params.beta = 2.0;
    params.k = -2.0;

    m_uncsentedKalman = kalman::createAugmentedUnscentedKalmanFilter(params);

    m_initialized = true;
}

//---------------------------------------------------------------------------
void TKalmanFilter::CreateAugmentedUnscented(cv::Rect_<track_t> rect0, Point_t rectv0)
{
    int MP = 4;
    int DP = 8;
    int CP = 0;

    cv::Mat processNoiseCov = cv::Mat::zeros(DP, DP, Mat_t(1));
    processNoiseCov.at<track_t>(0, 0) = 1e-3f;
    processNoiseCov.at<track_t>(1, 1) = 1e-3f;
    processNoiseCov.at<track_t>(2, 2) = 1e-3f;
    processNoiseCov.at<track_t>(3, 3) = 1e-3f;
    processNoiseCov.at<track_t>(4, 4) = 1e-3f;
    processNoiseCov.at<track_t>(5, 5) = 1e-3f;
    processNoiseCov.at<track_t>(6, 6) = 1e-3f;
    processNoiseCov.at<track_t>(7, 7) = 1e-3f;

    cv::Mat measurementNoiseCov = cv::Mat::zeros(MP, MP, Mat_t(1));
    measurementNoiseCov.at<track_t>(0, 0) = 1e-3f;
    measurementNoiseCov.at<track_t>(1, 1) = 1e-3f;
    measurementNoiseCov.at<track_t>(2, 2) = 1e-3f;
    measurementNoiseCov.at<track_t>(3, 3) = 1e-3f;

    cv::Mat initState(DP, 1, Mat_t(1));
    initState.at<track_t>(0, 0) = rect0.x;
    initState.at<track_t>(1, 0) = rect0.y;
    initState.at<track_t>(2, 0) = rectv0.x;
    initState.at<track_t>(3, 0) = rectv0.y;
    initState.at<track_t>(4, 0) = 0;
    initState.at<track_t>(5, 0) = 0;
    initState.at<track_t>(6, 0) = rect0.width;
    initState.at<track_t>(7, 0) = rect0.height;

    cv::Mat P = 1e-3f * cv::Mat::eye(DP, DP, Mat_t(1));

    cv::Ptr<AcceleratedModel> model(new AcceleratedModel(m_deltaTime, true));
    kalman::AugmentedUnscentedKalmanFilterParams params(DP, MP, CP, 0, 0, model);
    params.dataType = Mat_t(1);
    params.stateInit = initState.clone();
    params.errorCovInit = P.clone();
    params.measurementNoiseCov = measurementNoiseCov.clone();
    params.processNoiseCov = processNoiseCov.clone();

    params.alpha = 1;
    params.beta = 2.0;
    params.k = -2.0;

    m_uncsentedKalman = kalman::createAugmentedUnscentedKalmanFilter(params);

    m_initialized = true;
}
#endif

///
/// \brief TKalmanFilter::GetPointPrediction
/// \return
///
Point_t TKalmanFilter::GetPointPrediction()
{
    if (m_initialized)
    {
        cv::Mat prediction;

        switch (m_type)
        {
        case tracking::KalmanLinear:
            prediction = m_linearKalman.predict();
            break;

        case tracking::KalmanUnscented:
        case tracking::KalmanAugmentedUnscented:
#ifdef USE_OCV_UKF
            prediction = m_uncsentedKalman->predict();
#else
            prediction = m_linearKalman.predict();
            std::cerr << "UnscentedKalmanFilter was disabled in CMAKE! Set KalmanLinear in constructor." << std::endl;
#endif
            break;
        }

        m_lastPointResult = Point_t(prediction.at<track_t>(0), prediction.at<track_t>(1));
    }
    return m_lastPointResult;
}

///
/// \brief TKalmanFilter::Update
/// \param pt
/// \param dataCorrect
/// \return
///
Point_t TKalmanFilter::Update(Point_t pt, bool dataCorrect)
{
    if (!m_initialized)
    {
        if (m_initialPoints.size() < MIN_INIT_VALS)
        {
            if (dataCorrect)
            {
                m_initialPoints.push_back(pt);
                m_lastPointResult = pt;
            }
        }
        if (m_initialPoints.size() >= MIN_INIT_VALS)
        {
            track_t kx = 0;
            track_t bx = 0;
            track_t ky = 0;
            track_t by = 0;
            get_lin_regress_params(m_initialPoints, 0, MIN_INIT_VALS, kx, bx, ky, by);
            Point_t xy0(kx * (MIN_INIT_VALS - 1) + bx, ky * (MIN_INIT_VALS - 1) + by);
            Point_t xyv0(kx, ky);

            switch (m_type)
            {
            case tracking::KalmanLinear:
                if (m_useAcceleration)
					CreateLinearAcceleration(xy0, xyv0);
				else
					CreateLinear(xy0, xyv0);
                break;

            case tracking::KalmanUnscented:
#ifdef USE_OCV_UKF
                CreateUnscented(xy0, xyv0);
#else
				if (m_useAcceleration)
					CreateLinearAcceleration(xy0, xyv0);
				else
					CreateLinear(xy0, xyv0);
                std::cerr << "UnscentedKalmanFilter was disabled in CMAKE! Set KalmanLinear in constructor." << std::endl;
#endif
                break;

            case tracking::KalmanAugmentedUnscented:
#ifdef USE_OCV_UKF
                CreateAugmentedUnscented(xy0, xyv0);
#else
				if (m_useAcceleration)
					CreateLinearAcceleration(xy0, xyv0);
				else
					CreateLinear(xy0, xyv0);
                std::cerr << "AugmentedUnscentedKalmanFilter was disabled in CMAKE! Set KalmanLinear in constructor." << std::endl;
#endif
                break;
            }
            m_lastDist = 0;
        }
    }

    if (m_initialized)
    {
        cv::Mat measurement(2, 1, Mat_t(1));
        if (!dataCorrect)
        {
            measurement.at<track_t>(0) = m_lastPointResult.x;  //update using prediction
            measurement.at<track_t>(1) = m_lastPointResult.y;
        }
        else
        {
            measurement.at<track_t>(0) = pt.x;  //update using measurements
            measurement.at<track_t>(1) = pt.y;
        }
        // Correction
        cv::Mat estimated;
        switch (m_type)
        {
        case tracking::KalmanLinear:
        {
            estimated = m_linearKalman.correct(measurement);

            // Inertia correction
			if (!m_useAcceleration)
			{
				track_t currDist = sqrtf(sqr(estimated.at<track_t>(0) - pt.x) + sqr(estimated.at<track_t>(1) - pt.y));
				if (currDist > m_lastDist)
					m_deltaTime = std::min(m_deltaTime + m_deltaStep, m_deltaTimeMax);
				else
					m_deltaTime = std::max(m_deltaTime - m_deltaStep, m_deltaTimeMin);

				m_lastDist = currDist;

				m_linearKalman.transitionMatrix.at<track_t>(0, 2) = m_deltaTime;
				m_linearKalman.transitionMatrix.at<track_t>(1, 3) = m_deltaTime;
			}
            break;
        }

        case tracking::KalmanUnscented:
        case tracking::KalmanAugmentedUnscented:
#ifdef USE_OCV_UKF
            estimated = m_uncsentedKalman->correct(measurement);
#else
            estimated = m_linearKalman.correct(measurement);
            std::cerr << "UnscentedKalmanFilter was disabled in CMAKE! Set KalmanLinear in constructor." << std::endl;
#endif
            break;
        }

        m_lastPointResult.x = estimated.at<track_t>(0);   //update using measurements
        m_lastPointResult.y = estimated.at<track_t>(1);
    }
    else
    {
        if (dataCorrect)
            m_lastPointResult = pt;
    }
    return m_lastPointResult;
}

///
/// \brief TKalmanFilter::GetRectPrediction
/// \return
///
cv::Rect TKalmanFilter::GetRectPrediction()
{
    if (m_initialized)
    {
        cv::Mat prediction;

        switch (m_type)
        {
        case tracking::KalmanLinear:
            prediction = m_linearKalman.predict();
            break;

        case tracking::KalmanUnscented:
        case tracking::KalmanAugmentedUnscented:
#ifdef USE_OCV_UKF
            prediction = m_uncsentedKalman->predict();
#else
            prediction = m_linearKalman.predict();
            std::cerr << "UnscentedKalmanFilter was disabled in CMAKE! Set KalmanLinear in constructor." << std::endl;
#endif
            break;
        }

        m_lastRectResult = cv::Rect_<track_t>(prediction.at<track_t>(0), prediction.at<track_t>(1), prediction.at<track_t>(2), prediction.at<track_t>(3));
    }
    return cv::Rect(static_cast<int>(m_lastRectResult.x), static_cast<int>(m_lastRectResult.y), static_cast<int>(m_lastRectResult.width), static_cast<int>(m_lastRectResult.height));
}

///
/// \brief TKalmanFilter::Update
/// \param rect
/// \param dataCorrect
/// \return
///
cv::Rect TKalmanFilter::Update(cv::Rect rect, bool dataCorrect)
{
    if (!m_initialized)
    {
        if (m_initialRects.size() < MIN_INIT_VALS)
        {
            if (dataCorrect)
            {
                m_initialRects.push_back(rect);
                m_lastRectResult.x = static_cast<track_t>(rect.x);
                m_lastRectResult.y = static_cast<track_t>(rect.y);
                m_lastRectResult.width = static_cast<track_t>(rect.width);
                m_lastRectResult.height = static_cast<track_t>(rect.height);
            }
        }
        if (m_initialRects.size() == MIN_INIT_VALS)
        {
            std::vector<Point_t> initialPoints;
            Point_t averageSize(0, 0);
            for (const auto& r : m_initialRects)
            {
                initialPoints.emplace_back(static_cast<track_t>(r.x), static_cast<track_t>(r.y));
                averageSize.x += r.width;
                averageSize.y += r.height;
            }
            averageSize.x /= MIN_INIT_VALS;
            averageSize.y /= MIN_INIT_VALS;

            track_t kx = 0;
            track_t bx = 0;
            track_t ky = 0;
            track_t by = 0;
            get_lin_regress_params(initialPoints, 0, MIN_INIT_VALS, kx, bx, ky, by);
            cv::Rect_<track_t> rect0(kx * (MIN_INIT_VALS - 1) + bx, ky * (MIN_INIT_VALS - 1) + by, averageSize.x, averageSize.y);
            Point_t rectv0(kx, ky);

            switch (m_type)
            {
            case tracking::KalmanLinear:
				if (m_useAcceleration)
					CreateLinearAcceleration(rect0, rectv0);
				else
					CreateLinear(rect0, rectv0);
                break;

            case tracking::KalmanUnscented:
#ifdef USE_OCV_UKF
                CreateUnscented(rect0, rectv0);
#else
				if (m_useAcceleration)
					CreateLinearAcceleration(rect0, rectv0);
				else
					CreateLinear(rect0, rectv0);
                std::cerr << "UnscentedKalmanFilter was disabled in CMAKE! Set KalmanLinear in constructor." << std::endl;
#endif
                break;

            case tracking::KalmanAugmentedUnscented:
#ifdef USE_OCV_UKF
                CreateAugmentedUnscented(rect0, rectv0);
#else
				if (m_useAcceleration)
					CreateLinearAcceleration(rect0, rectv0);
				else
					CreateLinear(rect0, rectv0);
                std::cerr << "AugmentedUnscentedKalmanFilter was disabled in CMAKE! Set KalmanLinear in constructor." << std::endl;
#endif
                break;
            }
        }
    }

    if (m_initialized)
    {
        cv::Mat measurement(4, 1, Mat_t(1));
        if (!dataCorrect)
        {
            measurement.at<track_t>(0) = m_lastRectResult.x;  // update using prediction
            measurement.at<track_t>(1) = m_lastRectResult.y;
            measurement.at<track_t>(2) = m_lastRectResult.width;
            measurement.at<track_t>(3) = m_lastRectResult.height;
        }
        else
        {
            measurement.at<track_t>(0) = static_cast<track_t>(rect.x);  // update using measurements
            measurement.at<track_t>(1) = static_cast<track_t>(rect.y);
            measurement.at<track_t>(2) = static_cast<track_t>(rect.width);
            measurement.at<track_t>(3) = static_cast<track_t>(rect.height);
        }
        // Correction
        cv::Mat estimated;
        switch (m_type)
        {
        case tracking::KalmanLinear:
        {
            estimated = m_linearKalman.correct(measurement);

            m_lastRectResult.x = estimated.at<track_t>(0);   //update using measurements
            m_lastRectResult.y = estimated.at<track_t>(1);
            m_lastRectResult.width = estimated.at<track_t>(2);
            m_lastRectResult.height = estimated.at<track_t>(3);

            // Inertia correction
			if (!m_useAcceleration)
			{
				track_t currDist = sqrtf(sqr(estimated.at<track_t>(0) - rect.x) + sqr(estimated.at<track_t>(1) - rect.y) + sqr(estimated.at<track_t>(2) - rect.width) + sqr(estimated.at<track_t>(3) - rect.height));
				if (currDist > m_lastDist)
					m_deltaTime = std::min(m_deltaTime + m_deltaStep, m_deltaTimeMax);
				else
					m_deltaTime = std::max(m_deltaTime - m_deltaStep, m_deltaTimeMin);

				m_lastDist = currDist;

				m_linearKalman.transitionMatrix.at<track_t>(0, 4) = m_deltaTime;
				m_linearKalman.transitionMatrix.at<track_t>(1, 5) = m_deltaTime;
				m_linearKalman.transitionMatrix.at<track_t>(2, 6) = m_deltaTime;
				m_linearKalman.transitionMatrix.at<track_t>(3, 7) = m_deltaTime;
			}
            break;
        }

        case tracking::KalmanUnscented:
        case tracking::KalmanAugmentedUnscented:
#ifdef USE_OCV_UKF
            estimated = m_uncsentedKalman->correct(measurement);

            m_lastRectResult.x = estimated.at<track_t>(0);   //update using measurements
            m_lastRectResult.y = estimated.at<track_t>(1);
            m_lastRectResult.width = estimated.at<track_t>(6);
            m_lastRectResult.height = estimated.at<track_t>(7);
#else
            estimated = m_linearKalman.correct(measurement);

            m_lastRectResult.x = estimated.at<track_t>(0);   //update using measurements
            m_lastRectResult.y = estimated.at<track_t>(1);
            m_lastRectResult.width = estimated.at<track_t>(2);
            m_lastRectResult.height = estimated.at<track_t>(3);
            std::cerr << "UnscentedKalmanFilter was disabled in CMAKE! Set KalmanLinear in constructor." << std::endl;
#endif
            break;
        }
    }
    else
    {
        if (dataCorrect)
        {
            m_lastRectResult.x = static_cast<track_t>(rect.x);
            m_lastRectResult.y = static_cast<track_t>(rect.y);
            m_lastRectResult.width = static_cast<track_t>(rect.width);
            m_lastRectResult.height = static_cast<track_t>(rect.height);
        }
    }
    return cv::Rect(static_cast<int>(m_lastRectResult.x), static_cast<int>(m_lastRectResult.y), static_cast<int>(m_lastRectResult.width), static_cast<int>(m_lastRectResult.height));
}

///
/// \brief TKalmanFilter::GetRRectPrediction
/// \return
///
cv::RotatedRect TKalmanFilter::GetRRectPrediction()
{
	if (m_initialized)
	{
		cv::Mat prediction;

		switch (m_type)
		{
		case tracking::KalmanLinear:
			prediction = m_linearKalman.predict();
			break;

		case tracking::KalmanUnscented:
		case tracking::KalmanAugmentedUnscented:
#ifdef USE_OCV_UKF
			prediction = m_uncsentedKalman->predict();
#else
			prediction = m_linearKalman.predict();
			std::cerr << "UnscentedKalmanFilter was disabled in CMAKE! Set KalmanLinear in constructor." << std::endl;
#endif
			break;
		}

		m_lastRRectResult.center.x = prediction.at<track_t>(0);   //update using measurements
		m_lastRRectResult.center.y = prediction.at<track_t>(1);
		m_lastRRectResult.size.width = prediction.at<track_t>(2);
		m_lastRRectResult.size.height = prediction.at<track_t>(3);
		m_lastRRectResult.angle = prediction.at<track_t>(4);
	}
	return m_lastRRectResult;
}

///
/// \brief TKalmanFilter::Update
/// \param rrect
/// \param dataCorrect
/// \return
///
cv::RotatedRect TKalmanFilter::Update(cv::RotatedRect rrect, bool dataCorrect)
{
	if (!m_initialized)
	{
		if (m_initialRRects.size() < MIN_INIT_VALS)
		{
			if (dataCorrect)
			{
				m_initialRRects.push_back(rrect);
				m_lastRRectResult = rrect;
			}
		}
		if (m_initialRRects.size() == MIN_INIT_VALS)
		{
			std::vector<Point_t> initialPoints;
			Point_t averageSize(0, 0);
			track_t averageAngle = 0;
			for (const auto& rr : m_initialRRects)
			{
				initialPoints.emplace_back(static_cast<track_t>(rr.center.x), static_cast<track_t>(rr.center.y));
				averageSize.x += rr.size.width;
				averageSize.y += rr.size.height;
				averageAngle += rr.angle;
			}
			averageSize.x /= MIN_INIT_VALS;
			averageSize.y /= MIN_INIT_VALS;
			averageAngle /= MIN_INIT_VALS;

			track_t kx = 0;
			track_t bx = 0;
			track_t ky = 0;
			track_t by = 0;
			get_lin_regress_params(initialPoints, 0, MIN_INIT_VALS, kx, bx, ky, by);
			cv::RotatedRect rrect0(cv::Point2f(kx * (MIN_INIT_VALS - 1) + bx, ky * (MIN_INIT_VALS - 1) + by), averageSize, averageAngle);
			Point_t rrectv0(kx, ky);

			switch (m_type)
			{
			case tracking::KalmanLinear:
				if (m_useAcceleration)
					CreateLinearAcceleration(rrect0, rrectv0);
				else
					CreateLinear(rrect0, rrectv0);
				break;

			case tracking::KalmanUnscented:
#ifdef USE_OCV_UKF
				assert(0);
				//TODO: CreateUnscented(rrect0, rrectv0);
#else
				if (m_useAcceleration)
					CreateLinearAcceleration(rrect0, rrectv0);
				else
					CreateLinear(rrect0, rrectv0);
				std::cerr << "UnscentedKalmanFilter was disabled in CMAKE! Set KalmanLinear in constructor." << std::endl;
#endif
				break;

			case tracking::KalmanAugmentedUnscented:
#ifdef USE_OCV_UKF
				assert(0);
				// TODO: CreateAugmentedUnscented(rrect0, rrectv0);
#else
				if (m_useAcceleration)
					CreateLinearAcceleration(rrect0, rrectv0);
				else
					CreateLinear(rrect0, rrectv0);
				std::cerr << "AugmentedUnscentedKalmanFilter was disabled in CMAKE! Set KalmanLinear in constructor." << std::endl;
#endif
				break;
			}
		}
	}

	if (m_initialized)
	{
		cv::Mat measurement(5, 1, Mat_t(1));
		if (!dataCorrect)
		{
			measurement.at<track_t>(0) = m_lastRRectResult.center.x;  // update using prediction
			measurement.at<track_t>(1) = m_lastRRectResult.center.y;
			measurement.at<track_t>(2) = m_lastRRectResult.size.width;
			measurement.at<track_t>(3) = m_lastRRectResult.size.height;
			measurement.at<track_t>(4) = m_lastRRectResult.angle;
		}
		else
		{
			measurement.at<track_t>(0) = static_cast<track_t>(rrect.center.x);  // update using measurements
			measurement.at<track_t>(1) = static_cast<track_t>(rrect.center.y);
			measurement.at<track_t>(2) = static_cast<track_t>(rrect.size.width);
			measurement.at<track_t>(3) = static_cast<track_t>(rrect.size.height);
			measurement.at<track_t>(4) = static_cast<track_t>(rrect.angle);
		}
		// Correction
		cv::Mat estimated;
		switch (m_type)
		{
		case tracking::KalmanLinear:
		{
			estimated = m_linearKalman.correct(measurement);

			m_lastRRectResult.center.x = estimated.at<track_t>(0);   //update using measurements
			m_lastRRectResult.center.y = estimated.at<track_t>(1);
			m_lastRRectResult.size.width = estimated.at<track_t>(2);
			m_lastRRectResult.size.height = estimated.at<track_t>(3);
			m_lastRRectResult.angle = estimated.at<track_t>(4);

			// Inertia correction
			if (!m_useAcceleration)
			{
				track_t currDist = sqrtf(sqr(estimated.at<track_t>(0) - rrect.center.x) + sqr(estimated.at<track_t>(1) - rrect.center.y) +
					               sqr(estimated.at<track_t>(2) - rrect.size.width) + sqr(estimated.at<track_t>(3) - rrect.size.height));
				if (currDist > m_lastDist)
					m_deltaTime = std::min(m_deltaTime + m_deltaStep, m_deltaTimeMax);
				else
					m_deltaTime = std::max(m_deltaTime - m_deltaStep, m_deltaTimeMin);

				m_lastDist = currDist;

				m_linearKalman.transitionMatrix.at<track_t>(0, 5) = m_deltaTime;
				m_linearKalman.transitionMatrix.at<track_t>(1, 6) = m_deltaTime;
				m_linearKalman.transitionMatrix.at<track_t>(2, 7) = m_deltaTime;
				m_linearKalman.transitionMatrix.at<track_t>(3, 8) = m_deltaTime;
				m_linearKalman.transitionMatrix.at<track_t>(4, 9) = m_deltaTime;
			}
			break;
		}

		case tracking::KalmanUnscented:
		case tracking::KalmanAugmentedUnscented:
#ifdef USE_OCV_UKF
			estimated = m_uncsentedKalman->correct(measurement);

			m_lastRRectResult.center.x = estimated.at<track_t>(0);   //update using measurements
			m_lastRRectResult.center.y = estimated.at<track_t>(1);
			m_lastRRectResult.size.width = estimated.at<track_t>(6);
			m_lastRRectResult.size.height = estimated.at<track_t>(7);
			m_lastRRectResult.angle = estimated.at<track_t>(9);
#else
			estimated = m_linearKalman.correct(measurement);

			m_lastRRectResult.center.x = estimated.at<track_t>(0);   //update using measurements
			m_lastRRectResult.center.y = estimated.at<track_t>(1);
			m_lastRRectResult.size.width = estimated.at<track_t>(2);
			m_lastRRectResult.size.height = estimated.at<track_t>(3);
			m_lastRRectResult.angle = estimated.at<track_t>(4);
			std::cerr << "UnscentedKalmanFilter was disabled in CMAKE! Set KalmanLinear in constructor." << std::endl;
#endif
			break;
		}
	}
	else
	{
		if (dataCorrect)
			m_lastRRectResult = rrect;
	}
	return m_lastRRectResult;
}

///
/// \brief TKalmanFilter::GetVelocity
/// \return
///
cv::Vec<track_t, 2> TKalmanFilter::GetVelocity() const
{
    cv::Vec<track_t, 2> res(0, 0);
    if (m_initialized)
    {
        switch (m_type)
        {
        case tracking::KalmanLinear:
        {
            if (m_linearKalman.statePre.rows > 3)
            {
                int indX = 2;
                int indY = 3;
                if (m_linearKalman.statePre.rows > 4)
                {
                    indX = 4;
                    indY = 5;
                }
				//std::cout << "indX = " << indX << ", indY = " << indY << std::endl;
                res[0] = m_linearKalman.statePre.at<track_t>(indX);
                res[1] = m_linearKalman.statePre.at<track_t>(indY);
            }
            break;
        }

        case tracking::KalmanUnscented:
        case tracking::KalmanAugmentedUnscented:
#ifdef USE_OCV_UKF
            cv::Mat state = m_uncsentedKalman->getState();
            res[0] = state.at<track_t>(2);
            res[1] = state.at<track_t>(3);
#else
            std::cerr << "UnscentedKalmanFilter was disabled in CMAKE! Set KalmanLinear in constructor." << std::endl;
#endif
            break;
        }
    }
    return res;
}

//---------------------------------------------------------------------------
void TKalmanFilter::GetPtStateAndResCov(cv::Mat& covar, cv::Mat& state) const
{
    if (m_initialized)
    {
        switch (m_type)
        {
        case tracking::KalmanLinear:
        {
            state = m_linearKalman.statePost.clone();
            covar = m_linearKalman.processNoiseCov.clone();
            break;
        }

        case tracking::KalmanUnscented:
        case tracking::KalmanAugmentedUnscented:
#ifdef USE_OCV_UKF
            state = m_uncsentedKalman->getState();
#else
            std::cerr << "UnscentedKalmanFilter was disabled in CMAKE! Set KalmanLinear in constructor." << std::endl;
#endif
            break;
        }
    }
}


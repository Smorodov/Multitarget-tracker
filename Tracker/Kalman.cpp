#include "Kalman.h"
#include <iostream>
#include <vector>

//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
TKalmanFilter::TKalmanFilter(
        tracking::KalmanType type,
        Point_t pt,
        track_t deltaTime, // time increment (lower values makes target more "massive")
        track_t accelNoiseMag
        )
    :
      m_type(type),
      m_initialized(false),
      m_deltaTime(deltaTime),
      m_accelNoiseMag(accelNoiseMag)
{
    m_initialPoints.push_back(pt);
    m_lastPointResult = pt;
}

//---------------------------------------------------------------------------
TKalmanFilter::TKalmanFilter(
        tracking::KalmanType type,
        cv::Rect rect,
        track_t deltaTime, // time increment (lower values makes target more "massive")
        track_t accelNoiseMag
        )
    :
      m_type(type),
      m_initialized(false),
      m_deltaTime(deltaTime),
      m_accelNoiseMag(accelNoiseMag)
{
    m_initialRects.push_back(rect);
    m_lastRectResult = rect;
}

//---------------------------------------------------------------------------
TKalmanFilter::~TKalmanFilter()
{
}

//---------------------------------------------------------------------------
void TKalmanFilter::CreateLinear(Point_t xy0, Point_t xyv0)
{
    // We don't know acceleration, so, assume it to process noise.
    // But we can guess, the range of acceleration values thich can be achieved by tracked object.
    // Process noise. (standard deviation of acceleration: m/s^2)
    // shows, woh much target can accelerate.

    //4 state variables, 2 measurements
    m_linearKalman = std::make_unique<cv::KalmanFilter>(4, 2, 0);
    // Transition cv::Matrix
    m_linearKalman->transitionMatrix = (cv::Mat_<track_t>(4, 4) <<
                                        1, 0, m_deltaTime, 0,
                                        0, 1, 0, m_deltaTime,
                                        0, 0, 1, 0,
                                        0, 0, 0, 1);

    // init...
    m_lastPointResult = xy0;
    m_linearKalman->statePre.at<track_t>(0) = xy0.x; // x
    m_linearKalman->statePre.at<track_t>(1) = xy0.y; // y

    m_linearKalman->statePre.at<track_t>(2) = xyv0.x;
    m_linearKalman->statePre.at<track_t>(3) = xyv0.y;

    m_linearKalman->statePost.at<track_t>(0) = xy0.x;
    m_linearKalman->statePost.at<track_t>(1) = xy0.y;

    cv::setIdentity(m_linearKalman->measurementMatrix);

    m_linearKalman->processNoiseCov = (cv::Mat_<track_t>(4, 4) <<
                                       pow(m_deltaTime,4.0)/4.0	,0						,pow(m_deltaTime,3.0)/2.0		,0,
                                       0						,pow(m_deltaTime,4.0)/4.0	,0							,pow(m_deltaTime,3.0)/2.0,
                                       pow(m_deltaTime,3.0)/2.0	,0						,pow(m_deltaTime,2.0)			,0,
                                       0						,pow(m_deltaTime,3.0)/2.0	,0							,pow(m_deltaTime,2.0));


    m_linearKalman->processNoiseCov *= m_accelNoiseMag;

    setIdentity(m_linearKalman->measurementNoiseCov, cv::Scalar::all(0.1));

    setIdentity(m_linearKalman->errorCovPost, cv::Scalar::all(.1));

    m_initialized = true;
}

//---------------------------------------------------------------------------
void TKalmanFilter::CreateLinear(cv::Rect_<track_t> rect0, Point_t rectv0)
{
    // We don't know acceleration, so, assume it to process noise.
    // But we can guess, the range of acceleration values thich can be achieved by tracked object.
    // Process noise. (standard deviation of acceleration: m/s^2)
    // shows, woh much target can accelerate.

    //4 state variables (x, y, dx, dy, width, height), 4 measurements (x, y, width, height)
    m_linearKalman = std::make_unique<cv::KalmanFilter>(6, 4, 0);
    // Transition cv::Matrix
    m_linearKalman->transitionMatrix = (cv::Mat_<track_t>(6, 6) <<
                                        1, 0, 0, 0, m_deltaTime, 0,
                                        0, 1, 0, 0, 0,           m_deltaTime,
                                        0, 0, 1, 0, 0,           0,
                                        0, 0, 0, 1, 0,           0,
                                        0, 0, 0, 0, 1,           0,
                                        0, 0, 0, 0, 0,           1);

    // init...
    m_linearKalman->statePre.at<track_t>(0) = rect0.x;      // x
    m_linearKalman->statePre.at<track_t>(1) = rect0.y;      // y
    m_linearKalman->statePre.at<track_t>(2) = rect0.width;  // width
    m_linearKalman->statePre.at<track_t>(3) = rect0.height; // height
    m_linearKalman->statePre.at<track_t>(4) = rectv0.x;     // dx
    m_linearKalman->statePre.at<track_t>(5) = rectv0.y;     // dy

    m_linearKalman->statePost.at<track_t>(0) = rect0.x;
    m_linearKalman->statePost.at<track_t>(1) = rect0.y;
    m_linearKalman->statePost.at<track_t>(2) = rect0.width;
    m_linearKalman->statePost.at<track_t>(3) = rect0.height;

    cv::setIdentity(m_linearKalman->measurementMatrix);

    track_t n1 = pow(m_deltaTime, 4.) / 4.;
    track_t n2 = pow(m_deltaTime, 3.) / 2.;
    track_t n3 = pow(m_deltaTime, 2.);
    m_linearKalman->processNoiseCov = (cv::Mat_<track_t>(6, 6) <<
                                       n1, 0,  0,  0,  n2, 0,
                                       0,  n1, 0,  0,  0,  n2,
                                       0,  0,  n1, 0,  0,  0,
                                       0,  0,  0,  n1, 0,  0,
                                       n2, 0,  0,  0,  n3, 0,
                                       0,  n2, 0,  0,  0,  n3);

    m_linearKalman->processNoiseCov *= m_accelNoiseMag;

    setIdentity(m_linearKalman->measurementNoiseCov, cv::Scalar::all(0.1));

    setIdentity(m_linearKalman->errorCovPost, cv::Scalar::all(.1));

    m_initialized = true;
}

#ifdef USE_OCV_UKF
//---------------------------------------------------------------------------
class AcceleratedModel: public cv::tracking::UkfSystemModel
{
public:
    AcceleratedModel(track_t deltaTime, bool rectModel)
        :
          cv::tracking::UkfSystemModel(),
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
        {
            x_kplus1 += v_k + u_k;
        }
        else
        {
            x_kplus1 += v_k;
        }
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
    processNoiseCov.at<track_t>(0, 0) = 1e-14;
    processNoiseCov.at<track_t>(1, 1) = 1e-14;
    processNoiseCov.at<track_t>(2, 2) = 1e-6;
    processNoiseCov.at<track_t>(3, 3) = 1e-6;
    processNoiseCov.at<track_t>(4, 4) = 1e-6;
    processNoiseCov.at<track_t>(5, 5) = 1e-6;

    cv::Mat measurementNoiseCov = cv::Mat::zeros(MP, MP, Mat_t(1));
    measurementNoiseCov.at<track_t>(0, 0) = 1e-6;
    measurementNoiseCov.at<track_t>(1, 1) = 1e-6;

    cv::Mat initState(DP, 1, Mat_t(1));
    initState.at<track_t>(0, 0) = xy0.x;
    initState.at<track_t>(1, 0) = xy0.y;
    initState.at<track_t>(2, 0) = xyv0.x;
    initState.at<track_t>(3, 0) = xyv0.y;
    initState.at<track_t>(4, 0) = 0;
    initState.at<track_t>(5, 0) = 0;

    cv::Mat P = 1e-6 * cv::Mat::eye(DP, DP, Mat_t(1));

    cv::Ptr<AcceleratedModel> model(new AcceleratedModel(m_deltaTime, false));
    cv::tracking::UnscentedKalmanFilterParams params(DP, MP, CP, 0, 0, model);
    params.dataType = Mat_t(1);
    params.stateInit = initState.clone();
    params.errorCovInit = P.clone();
    params.measurementNoiseCov = measurementNoiseCov.clone();
    params.processNoiseCov = processNoiseCov.clone();

    params.alpha = 1.0;
    params.beta = 2.0;
    params.k = -2.0;

    m_uncsentedKalman = cv::tracking::createUnscentedKalmanFilter(params);

    m_initialized = true;
}

//---------------------------------------------------------------------------
void TKalmanFilter::CreateUnscented(cv::Rect_<track_t> rect0, Point_t rectv0)
{
    int MP = 4;
    int DP = 8;
    int CP = 0;

    cv::Mat processNoiseCov = cv::Mat::zeros(DP, DP, Mat_t(1));
    processNoiseCov.at<track_t>(0, 0) = 1e-3;
    processNoiseCov.at<track_t>(1, 1) = 1e-3;
    processNoiseCov.at<track_t>(2, 2) = 1e-3;
    processNoiseCov.at<track_t>(3, 3) = 1e-3;
    processNoiseCov.at<track_t>(4, 4) = 1e-3;
    processNoiseCov.at<track_t>(5, 5) = 1e-3;
    processNoiseCov.at<track_t>(6, 6) = 1e-3;
    processNoiseCov.at<track_t>(7, 7) = 1e-3;

    cv::Mat measurementNoiseCov = cv::Mat::zeros(MP, MP, Mat_t(1));
    measurementNoiseCov.at<track_t>(0, 0) = 1e-3;
    measurementNoiseCov.at<track_t>(1, 1) = 1e-3;
    measurementNoiseCov.at<track_t>(2, 2) = 1e-3;
    measurementNoiseCov.at<track_t>(3, 3) = 1e-3;

    cv::Mat initState(DP, 1, Mat_t(1));
    initState.at<track_t>(0, 0) = rect0.x;
    initState.at<track_t>(1, 0) = rect0.y;
    initState.at<track_t>(2, 0) = rectv0.x;
    initState.at<track_t>(3, 0) = rectv0.y;
    initState.at<track_t>(4, 0) = 0;
    initState.at<track_t>(5, 0) = 0;
    initState.at<track_t>(6, 0) = rect0.width;
    initState.at<track_t>(7, 0) = rect0.height;

    cv::Mat P = 1e-3 * cv::Mat::eye(DP, DP, Mat_t(1));

    cv::Ptr<AcceleratedModel> model(new AcceleratedModel(m_deltaTime, true));
    cv::tracking::UnscentedKalmanFilterParams params(DP, MP, CP, 0, 0, model);
    params.dataType = Mat_t(1);
    params.stateInit = initState.clone();
    params.errorCovInit = P.clone();
    params.measurementNoiseCov = measurementNoiseCov.clone();
    params.processNoiseCov = processNoiseCov.clone();

    params.alpha = 1;
    params.beta = 2.0;
    params.k = -2.0;

    m_uncsentedKalman = cv::tracking::createUnscentedKalmanFilter(params);

    m_initialized = true;
}

//---------------------------------------------------------------------------
void TKalmanFilter::CreateAugmentedUnscented(Point_t xy0, Point_t xyv0)
{
    int MP = 2;
    int DP = 6;
    int CP = 0;

    cv::Mat processNoiseCov = cv::Mat::zeros(DP, DP, Mat_t(1));
    processNoiseCov.at<track_t>(0, 0) = 1e-14;
    processNoiseCov.at<track_t>(1, 1) = 1e-14;
    processNoiseCov.at<track_t>(2, 2) = 1e-6;
    processNoiseCov.at<track_t>(3, 3) = 1e-6;
    processNoiseCov.at<track_t>(4, 4) = 1e-6;
    processNoiseCov.at<track_t>(5, 5) = 1e-6;

    cv::Mat measurementNoiseCov = cv::Mat::zeros(MP, MP, Mat_t(1));
    measurementNoiseCov.at<track_t>(0, 0) = 1e-6;
    measurementNoiseCov.at<track_t>(1, 1) = 1e-6;

    cv::Mat initState(DP, 1, Mat_t(1));
    initState.at<track_t>(0, 0) = xy0.x;
    initState.at<track_t>(1, 0) = xy0.y;
    initState.at<track_t>(2, 0) = xyv0.x;
    initState.at<track_t>(3, 0) = xyv0.y;
    initState.at<track_t>(4, 0) = 0;
    initState.at<track_t>(5, 0) = 0;

    cv::Mat P = 1e-6 * cv::Mat::eye(DP, DP, Mat_t(1));

    cv::Ptr<AcceleratedModel> model(new AcceleratedModel(m_deltaTime, false));
    cv::tracking::AugmentedUnscentedKalmanFilterParams params(DP, MP, CP, 0, 0, model);
    params.dataType = Mat_t(1);
    params.stateInit = initState.clone();
    params.errorCovInit = P.clone();
    params.measurementNoiseCov = measurementNoiseCov.clone();
    params.processNoiseCov = processNoiseCov.clone();

    params.alpha = 1;
    params.beta = 2.0;
    params.k = -2.0;

    m_uncsentedKalman = cv::tracking::createAugmentedUnscentedKalmanFilter(params);

    m_initialized = true;
}

//---------------------------------------------------------------------------
void TKalmanFilter::CreateAugmentedUnscented(cv::Rect_<track_t> rect0, Point_t rectv0)
{
    int MP = 4;
    int DP = 8;
    int CP = 0;

    cv::Mat processNoiseCov = cv::Mat::zeros(DP, DP, Mat_t(1));
    processNoiseCov.at<track_t>(0, 0) = 1e-3;
    processNoiseCov.at<track_t>(1, 1) = 1e-3;
    processNoiseCov.at<track_t>(2, 2) = 1e-3;
    processNoiseCov.at<track_t>(3, 3) = 1e-3;
    processNoiseCov.at<track_t>(4, 4) = 1e-3;
    processNoiseCov.at<track_t>(5, 5) = 1e-3;
    processNoiseCov.at<track_t>(6, 6) = 1e-3;
    processNoiseCov.at<track_t>(7, 7) = 1e-3;

    cv::Mat measurementNoiseCov = cv::Mat::zeros(MP, MP, Mat_t(1));
    measurementNoiseCov.at<track_t>(0, 0) = 1e-3;
    measurementNoiseCov.at<track_t>(1, 1) = 1e-3;
    measurementNoiseCov.at<track_t>(2, 2) = 1e-3;
    measurementNoiseCov.at<track_t>(3, 3) = 1e-3;

    cv::Mat initState(DP, 1, Mat_t(1));
    initState.at<track_t>(0, 0) = rect0.x;
    initState.at<track_t>(1, 0) = rect0.y;
    initState.at<track_t>(2, 0) = rectv0.x;
    initState.at<track_t>(3, 0) = rectv0.y;
    initState.at<track_t>(4, 0) = 0;
    initState.at<track_t>(5, 0) = 0;
    initState.at<track_t>(6, 0) = rect0.width;
    initState.at<track_t>(7, 0) = rect0.height;

    cv::Mat P = 1e-3 * cv::Mat::eye(DP, DP, Mat_t(1));

    cv::Ptr<AcceleratedModel> model(new AcceleratedModel(m_deltaTime, true));
    cv::tracking::AugmentedUnscentedKalmanFilterParams params(DP, MP, CP, 0, 0, model);
    params.dataType = Mat_t(1);
    params.stateInit = initState.clone();
    params.errorCovInit = P.clone();
    params.measurementNoiseCov = measurementNoiseCov.clone();
    params.processNoiseCov = processNoiseCov.clone();

    params.alpha = 1;
    params.beta = 2.0;
    params.k = -2.0;

    m_uncsentedKalman = cv::tracking::createAugmentedUnscentedKalmanFilter(params);

    m_initialized = true;
}
#endif

//---------------------------------------------------------------------------
Point_t TKalmanFilter::GetPointPrediction()
{
    if (m_initialized)
    {
        cv::Mat prediction;

        switch (m_type)
        {
        case tracking::KalmanLinear:
            prediction = m_linearKalman->predict();
            break;

        case tracking::KalmanUnscented:
        case tracking::KalmanAugmentedUnscented:
#ifdef USE_OCV_UKF
            prediction = m_uncsentedKalman->predict();
#else
            prediction = m_linearKalman->predict();
            std::cerr << "UnscentedKalmanFilter was disabled in CMAKE! Set KalmanLinear in constructor." << std::endl;
#endif
            break;
        }

        m_lastPointResult = Point_t(prediction.at<track_t>(0), prediction.at<track_t>(1));
    }
    else
    {

    }
    return m_lastPointResult;
}

//---------------------------------------------------------------------------
Point_t TKalmanFilter::Update(Point_t pt, bool dataCorrect)
{
    if (!m_initialized)
    {
        if (m_initialPoints.size() < MIN_INIT_VALS)
        {
            if (dataCorrect)
            {
                m_initialPoints.push_back(pt);
            }
        }
        if (m_initialPoints.size() == MIN_INIT_VALS)
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
                CreateLinear(xy0, xyv0);
                break;

            case tracking::KalmanUnscented:
#ifdef USE_OCV_UKF
                CreateUnscented(xy0, xyv0);
#else
                CreateLinear(xy0, xyv0);
                std::cerr << "UnscentedKalmanFilter was disabled in CMAKE! Set KalmanLinear in constructor." << std::endl;
#endif
                break;

            case tracking::KalmanAugmentedUnscented:
#ifdef USE_OCV_UKF
                CreateAugmentedUnscented(xy0, xyv0);
#else
                CreateLinear(xy0, xyv0);
                std::cerr << "AugmentedUnscentedKalmanFilter was disabled in CMAKE! Set KalmanLinear in constructor." << std::endl;
#endif
                break;
            }
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
            estimated = m_linearKalman->correct(measurement);
            break;

        case tracking::KalmanUnscented:
        case tracking::KalmanAugmentedUnscented:
#ifdef USE_OCV_UKF
            estimated = m_uncsentedKalman->correct(measurement);
#else
            estimated = m_linearKalman->correct(measurement);
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
        {
            m_lastPointResult = pt;
        }
    }
    return m_lastPointResult;
}

//---------------------------------------------------------------------------
cv::Rect TKalmanFilter::GetRectPrediction()
{
    if (m_initialized)
    {
        cv::Mat prediction;

        switch (m_type)
        {
        case tracking::KalmanLinear:
            prediction = m_linearKalman->predict();
            break;

        case tracking::KalmanUnscented:
        case tracking::KalmanAugmentedUnscented:
#ifdef USE_OCV_UKF
            prediction = m_uncsentedKalman->predict();
#else
            prediction = m_linearKalman->predict();
            std::cerr << "UnscentedKalmanFilter was disabled in CMAKE! Set KalmanLinear in constructor." << std::endl;
#endif
            break;
        }

        m_lastRectResult = cv::Rect_<track_t>(prediction.at<track_t>(0), prediction.at<track_t>(1), prediction.at<track_t>(2), prediction.at<track_t>(3));
    }
    else
    {

    }
    return cv::Rect(static_cast<int>(m_lastRectResult.x), static_cast<int>(m_lastRectResult.y), static_cast<int>(m_lastRectResult.width), static_cast<int>(m_lastRectResult.height));
}

//---------------------------------------------------------------------------
cv::Rect TKalmanFilter::Update(cv::Rect rect, bool dataCorrect)
{
    if (!m_initialized)
    {
        if (m_initialRects.size() < MIN_INIT_VALS)
        {
            if (dataCorrect)
            {
                m_initialRects.push_back(rect);
            }
        }
        if (m_initialRects.size() == MIN_INIT_VALS)
        {
            std::vector<Point_t> initialPoints;
            Point_t averageSize(0, 0);
            for (const auto& r : m_initialRects)
            {
                initialPoints.push_back(Point_t(r.x, r.y));
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
                CreateLinear(rect0, rectv0);
                break;

            case tracking::KalmanUnscented:
#ifdef USE_OCV_UKF
                CreateUnscented(rect0, rectv0);
#else
                CreateLinear(rect0, rectv0);
                std::cerr << "UnscentedKalmanFilter was disabled in CMAKE! Set KalmanLinear in constructor." << std::endl;
#endif
                break;

            case tracking::KalmanAugmentedUnscented:
#ifdef USE_OCV_UKF
                CreateAugmentedUnscented(rect0, rectv0);
#else
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
            estimated = m_linearKalman->correct(measurement);

            m_lastRectResult.x = estimated.at<track_t>(0);   //update using measurements
            m_lastRectResult.y = estimated.at<track_t>(1);
            m_lastRectResult.width = estimated.at<track_t>(2);
            m_lastRectResult.height = estimated.at<track_t>(3);
            break;

        case tracking::KalmanUnscented:
        case tracking::KalmanAugmentedUnscented:
#ifdef USE_OCV_UKF
            estimated = m_uncsentedKalman->correct(measurement);

            m_lastRectResult.x = estimated.at<track_t>(0);   //update using measurements
            m_lastRectResult.y = estimated.at<track_t>(1);
            m_lastRectResult.width = estimated.at<track_t>(6);
            m_lastRectResult.height = estimated.at<track_t>(7);
#else
            estimated = m_linearKalman->correct(measurement);

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

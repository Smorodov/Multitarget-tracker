#include "track.h"

#include "dat/dat_tracker.hpp"
#ifdef USE_STAPLE_TRACKER
#include "staple/staple_tracker.hpp"
#include "ldes/ldes_tracker.h"
#endif

///
/// \brief CTrack
/// \param pt
/// \param region
/// \param deltaTime
/// \param accelNoiseMag
/// \param trackID
/// \param filterObjectSize
/// \param externalTrackerForLost
///
CTrack::CTrack(
        const CRegion& region,
        tracking::KalmanType kalmanType,
        track_t deltaTime,
        track_t accelNoiseMag,
        size_t trackID,
        bool filterObjectSize,
        tracking::LostTrackType externalTrackerForLost
        )
    :
      m_trackID(trackID),
      m_skippedFrames(0),
      m_lastRegion(region),
      m_predictionPoint(region.m_rrect.center),
      m_predictionRect(region.m_rrect),
      m_kalman(kalmanType, deltaTime, accelNoiseMag),
      m_filterObjectSize(filterObjectSize),
      m_outOfTheFrame(false),
      m_externalTrackerForLost(externalTrackerForLost)
{
    if (filterObjectSize)
        m_kalman.Update(region.m_brect, true);
    else
        m_kalman.Update(m_predictionPoint, true);

    m_trace.push_back(m_predictionPoint, m_predictionPoint);
}

///
/// \brief CTrack::CalcDistCenter
/// \param reg
/// \return
///
track_t CTrack::CalcDistCenter(const CRegion& reg) const
{
    Point_t diff = m_predictionPoint - reg.m_rrect.center;
    return sqrtf(sqr(diff.x) + sqr(diff.y));
}

///
/// \brief CTrack::CalcDistRect
/// \param reg
/// \return
///
track_t CTrack::CalcDistRect(const CRegion& reg) const
{
    std::array<track_t, 5> diff;
    diff[0] = reg.m_rrect.center.x - m_lastRegion.m_rrect.center.x;
    diff[1] = reg.m_rrect.center.y - m_lastRegion.m_rrect.center.y;
    diff[2] = static_cast<track_t>(m_lastRegion.m_rrect.size.width - reg.m_rrect.size.width);
    diff[3] = static_cast<track_t>(m_lastRegion.m_rrect.size.height - reg.m_rrect.size.height);
    diff[4] = static_cast<track_t>(m_lastRegion.m_rrect.angle - reg.m_rrect.angle);

    track_t dist = 0;
    for (size_t i = 0; i < diff.size(); ++i)
    {
        dist += sqr(diff[i]);
    }
    return sqrtf(dist);
}

///
/// \brief CTrack::CalcDistJaccard
/// \param reg
/// \return
///
track_t CTrack::CalcDistJaccard(const CRegion& reg) const
{
    track_t intArea = static_cast<track_t>((reg.m_brect & m_lastRegion.m_brect).area());
    track_t unionArea = static_cast<track_t>(reg.m_brect.area() + m_lastRegion.m_brect.area() - intArea);

    return 1 - intArea / unionArea;
}

///
/// \brief CTrack::CalcDistHist
/// \param reg
/// \return
///
track_t CTrack::CalcDistHist(const CRegion& reg, cv::UMat currFrame) const
{
	track_t res = 1;

	if (reg.m_hist.empty())
	{
		int bins = 64;
		std::vector<int> histSize;
		std::vector<float> ranges;
		std::vector<int> channels;

		for (int i = 0, stop = currFrame.channels(); i < stop; ++i)
		{
			histSize.push_back(bins);
			ranges.push_back(0);
			ranges.push_back(255);
			channels.push_back(i);
		}

		std::vector<cv::UMat> regROI = { currFrame(reg.m_brect) };
		cv::calcHist(regROI, channels, cv::Mat(), reg.m_hist, histSize, ranges, false);
		cv::normalize(reg.m_hist, reg.m_hist, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
	}
	if (!reg.m_hist.empty() && !m_lastRegion.m_hist.empty())
	{
#if (((CV_VERSION_MAJOR == 4) && (CV_VERSION_MINOR < 2)) || (CV_VERSION_MAJOR == 3))
		res = static_cast<track_t>(cv::compareHist(reg.m_hist, m_lastRegion.m_hist, CV_COMP_BHATTACHARYYA));
		//res = 1.f - static_cast<track_t>(cv::compareHist(reg.m_hist, m_lastRegion.m_hist, CV_COMP_CORREL));
#else
		res = static_cast<track_t>(cv::compareHist(reg.m_hist, m_lastRegion.m_hist, cv::HISTCMP_BHATTACHARYYA));
#endif
	}

	return res;
}

///
/// \brief CTrack::Update
/// \*param region
/// \param dataCorrect
/// \param max_trace_length
/// \param prevFrame
/// \param currFrame
/// \param trajLen
///
void CTrack::Update(
        const CRegion& region,
        bool dataCorrect,
        size_t max_trace_length,
        cv::UMat prevFrame,
        cv::UMat currFrame,
        int trajLen
        )
{
    if (m_filterObjectSize) // Kalman filter for object coordinates and size
    {
        RectUpdate(region, dataCorrect, prevFrame, currFrame);
    }
    else // Kalman filter only for object center
    {
        PointUpdate(region.m_rrect.center, region.m_rrect.size, dataCorrect, currFrame.size());
    }

    if (dataCorrect)
    {
        //std::cout << m_lastRegion.m_brect << " - " << region.m_brect << std::endl;

        m_lastRegion = region;
        m_trace.push_back(m_predictionPoint, region.m_rrect.center);

        CheckStatic(trajLen, currFrame, region);
    }
    else
    {
        m_trace.push_back(m_predictionPoint);
    }

    if (m_trace.size() > max_trace_length)
    {
        m_trace.pop_front(m_trace.size() - max_trace_length);
    }
}

///
/// \brief CTrack::IsStatic
/// \return
///
bool CTrack::IsStatic() const
{
    return m_isStatic;
}

///
/// \brief CTrack::IsStaticTimeout
/// \param framesTime
/// \return
///
bool CTrack::IsStaticTimeout(int framesTime) const
{
    return (m_staticFrames > framesTime);
}

///
/// \brief CTrack::IsOutOfTheFrame
/// \return
///
bool CTrack::IsOutOfTheFrame() const
{
	return m_outOfTheFrame;
}

///
/// \brief CTrack::IsInsideArea
///        If result <= 1 then center of the object is inside ellipse with prediction and velocity
/// \param pt
/// \return
///
track_t CTrack::IsInsideArea(const Point_t& pt, track_t minVal) const
{
	auto velocity = m_kalman.GetVelocity();
    velocity[0] = std::max(minVal, 3 * velocity[0]);
    velocity[1] = std::max(minVal, 3 * velocity[1]);
	track_t res = sqr(pt.x - m_predictionPoint.x) / sqr(velocity[0]) + sqr(pt.y - m_predictionPoint.y) / sqr(velocity[1]);
    return res;
}

///
/// \brief CTrack::WidthDist
/// \param reg
/// \return
///
track_t CTrack::WidthDist(const CRegion& reg) const
{
    if (m_lastRegion.m_rrect.size.width < reg.m_rrect.size.width)
        return 1.f - m_lastRegion.m_rrect.size.width / reg.m_rrect.size.width;
    else
        return 1.f - reg.m_rrect.size.width / m_lastRegion.m_rrect.size.width;
}

///
/// \brief CTrack::HeightDist
/// \param reg
/// \return
///
track_t CTrack::HeightDist(const CRegion& reg) const
{
    if (m_lastRegion.m_rrect.size.height < reg.m_rrect.size.height)
        return 1.f - m_lastRegion.m_rrect.size.height / reg.m_rrect.size.height;
    else
        return 1.f - reg.m_rrect.size.height / m_lastRegion.m_rrect.size.height;
}

///
/// \brief CTrack::CheckStatic
/// \param trajLen
/// \return
///
bool CTrack::CheckStatic(int trajLen, cv::UMat currFrame, const CRegion& region)
{
    if (!trajLen || static_cast<int>(m_trace.size()) < trajLen)
    {
        m_isStatic = false;
        m_staticFrames = 0;
        m_staticFrame = cv::UMat();
    }
    else
    {
        track_t kx = 0;
        track_t bx = 0;
        track_t ky = 0;
        track_t by = 0;
        get_lin_regress_params(m_trace, m_trace.size() - trajLen, m_trace.size(), kx, bx, ky, by);
        track_t speed = sqrt(sqr(kx * trajLen) + sqr(ky * trajLen));
        const track_t speedThresh = 10;
        if (speed < speedThresh)
        {
            if (!m_isStatic)
            {
                m_staticFrame = currFrame.clone();
                m_staticRect = region.m_brect;
#if 0
#ifndef SILENT_WORK
                cv::namedWindow("m_staticFrame", cv::WINDOW_NORMAL);
                cv::Mat img = m_staticFrame.getMat(cv::ACCESS_READ).clone();
                cv::rectangle(img, m_staticRect, cv::Scalar(255, 0, 255), 1);
                for (size_t i = m_trace.size() - trajLen; i < m_trace.size() - 1; ++i)
                {
                    cv::line(img, m_trace[i], m_trace[i + 1], cv::Scalar(0, 0, 0), 1, cv::LINE_8);
                }
                std::string label = "(" + std::to_string(kx) + ", "  + std::to_string(ky) + ") = " + std::to_string(speed);
                cv::line(img,
                         cv::Point(bx, by),
                         cv::Point(kx * trajLen + bx, ky * trajLen + by),
                         cv::Scalar(0, 0, 0), 1, cv::LINE_8);
                cv::putText(img, label, m_staticRect.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
                cv::imshow("m_staticFrame", img);
                std::cout << "m_staticRect = " << m_staticRect << std::endl;
                cv::waitKey(1);
#endif
#endif
            }

            ++m_staticFrames;
            m_isStatic = true;
        }
        else
        {
            m_isStatic = false;
            m_staticFrames = 0;
            m_staticFrame = cv::UMat();
        }
    }

    return m_isStatic;
}

///
/// \brief GetLastRect
/// \return
///
cv::RotatedRect CTrack::GetLastRect() const
{
    if (m_filterObjectSize)
    {
        return m_predictionRect;
    }
    else
    {
        return cv::RotatedRect(cv::Point2f(m_predictionPoint.x, m_predictionPoint.y), m_predictionRect.size, m_predictionRect.angle);
    }
}

///
/// \brief CTrack::LastRegion
/// \return
///
const CRegion& CTrack::LastRegion() const
{
    return m_lastRegion;
}

///
/// \brief CTrack::ConstructObject
/// \return
///
TrackingObject CTrack::ConstructObject() const
{
    return TrackingObject(GetLastRect(), m_trackID, m_trace, IsStatic(), IsOutOfTheFrame(),
                          m_lastRegion.m_type, m_lastRegion.m_confidence, m_kalman.GetVelocity());
}

///
/// \brief CTrack::SkippedFrames
/// \return
///
size_t CTrack::SkippedFrames() const
{
    return m_skippedFrames;
}

///
/// \brief CTrack::SkippedFrames
/// \return
///
size_t& CTrack::SkippedFrames()
{
    return m_skippedFrames;
}

///
/// \brief RectUpdate
/// \param region
/// \param dataCorrect
/// \param prevFrame
/// \param currFrame
///
void CTrack::RectUpdate(
        const CRegion& region,
        bool dataCorrect,
        cv::UMat prevFrame,
        cv::UMat currFrame
        )
{
    m_kalman.GetRectPrediction();

    bool recalcPrediction = true;

    auto Clamp = [](int& v, int& size, int hi) -> int
    {
        int res = 0;

        if (size < 2)
            size = 2;

        if (v < 0)
        {
            res = v;
            v = 0;
            return res;
        }
        else if (v + size > hi - 1)
        {
			res = v;
            v = hi - 1 - size;
            if (v < 0)
            {
                size += v;
                v = 0;
            }
            res -= v;
            return res;
        }
        return res;
    };

    auto UpdateRRect = [&](cv::Rect prevRect, cv::Rect newRect)
    {
        m_predictionRect.center.x += newRect.x - prevRect.x;
        m_predictionRect.center.y += newRect.y - prevRect.y;
        m_predictionRect.size.width *= newRect.width / static_cast<float>(prevRect.width);
        m_predictionRect.size.height *= newRect.height / static_cast<float>(prevRect.height);
    };

    switch (m_externalTrackerForLost)
    {
    case tracking::TrackNone:
        break;

    case tracking::TrackKCF:
    case tracking::TrackMIL:
    case tracking::TrackMedianFlow:
    case tracking::TrackGOTURN:
    case tracking::TrackMOSSE:
	case tracking::TrackCSRT:
#ifdef USE_OCV_KCF
        if (!dataCorrect)
        {
            cv::Rect brect = m_predictionRect.boundingRect();

            cv::Size roiSize(std::max(2 * brect.width, currFrame.cols / 4), std::max(2 * brect.height, currFrame.rows / 4));
            if (roiSize.width > currFrame.cols)
            {
                roiSize.width = currFrame.cols;
            }
            if (roiSize.height > currFrame.rows)
            {
                roiSize.height = currFrame.rows;
            }
            cv::Point roiTL(brect.x + brect.width / 2 - roiSize.width / 2, brect.y + brect.height / 2 - roiSize.height / 2);
            cv::Rect roiRect(roiTL, roiSize);
            Clamp(roiRect.x, roiRect.width, currFrame.cols);
            Clamp(roiRect.y, roiRect.height, currFrame.rows);

            bool inited = false;
            if (!m_tracker || m_tracker.empty())
            {
                CreateExternalTracker(currFrame.channels());

                cv::Rect2d lastRect(brect.x - roiRect.x, brect.y - roiRect.y, brect.width, brect.height);
                if (m_staticFrame.empty())
                {
                    int dx = 1;//m_predictionRect.width / 8;
                    int dy = 1;//m_predictionRect.height / 8;
                    lastRect = cv::Rect2d(brect.x - roiRect.x - dx, brect.y - roiRect.y - dy, brect.width + 2 * dx, brect.height + 2 * dy);
                }
                else
                {
                    lastRect = cv::Rect2d(m_staticRect.x - roiRect.x, m_staticRect.y - roiRect.y, m_staticRect.width, m_staticRect.height);
                }

                if (lastRect.x >= 0 &&
                        lastRect.y >= 0 &&
                        lastRect.x + lastRect.width < roiRect.width &&
                        lastRect.y + lastRect.height < roiRect.height &&
                        lastRect.area() > 0)
                {
                    if (m_staticFrame.empty())
                    {
                        m_tracker->init(cv::UMat(prevFrame, roiRect), lastRect);
                    }
                    else
                    {
                        m_tracker->init(cv::UMat(m_staticFrame, roiRect), lastRect);
                    }
#if 0
#ifndef SILENT_WORK
                    cv::Mat tmp = cv::UMat(prevFrame, roiRect).getMat(cv::ACCESS_READ).clone();
                    cv::rectangle(tmp, lastRect, cv::Scalar(255, 255, 255), 2);
                    cv::imshow("init", tmp);
#endif
#endif

                    inited = true;
                    m_outOfTheFrame = false;
                }
                else
                {
                    m_tracker.release();
                    m_outOfTheFrame = true;
                }
            }
            cv::Rect2d newRect;
            if (!inited && !m_tracker.empty() && m_tracker->update(cv::UMat(currFrame, roiRect), newRect))
            {
#if 0
#ifndef SILENT_WORK
                cv::Mat tmp2 = cv::UMat(currFrame, roiRect).getMat(cv::ACCESS_READ).clone();
                cv::rectangle(tmp2, newRect, cv::Scalar(255, 255, 255), 2);
                cv::imshow("track", tmp2);
#endif
#endif

                cv::Rect prect(cvRound(newRect.x) + roiRect.x, cvRound(newRect.y) + roiRect.y, cvRound(newRect.width), cvRound(newRect.height));

                UpdateRRect(brect, m_kalman.Update(prect, true));

                recalcPrediction = false;
            }
        }
        else
        {
            if (m_tracker && !m_tracker.empty())
            {
                m_tracker.release();
            }
        }
#else
        std::cerr << "KCF tracker was disabled in CMAKE! Set lostTrackType = TrackNone in constructor." << std::endl;
#endif
        break;

    case tracking::TrackDAT:
    case tracking::TrackSTAPLE:
    case tracking::TrackLDES:
        if (!dataCorrect)
        {
            bool inited = false;
            cv::Rect brect = m_predictionRect.boundingRect();
            if (!m_VOTTracker)
            {
                CreateExternalTracker(currFrame.channels());

                cv::Rect2d lastRect(brect.x, brect.y, brect.width, brect.height);
                if (!m_staticFrame.empty())
                {
                    lastRect = cv::Rect2d(m_staticRect.x, m_staticRect.y, m_staticRect.width, m_staticRect.height);
                }

                if (lastRect.x >= 0 &&
                        lastRect.y >= 0 &&
                        lastRect.x + lastRect.width < prevFrame.cols &&
                        lastRect.y + lastRect.height < prevFrame.rows &&
                        lastRect.area() > 0)
                {
                    if (m_staticFrame.empty())
                    {
                        cv::Mat mat = prevFrame.getMat(cv::ACCESS_READ);
                        m_VOTTracker->Initialize(mat, lastRect);
                        m_VOTTracker->Train(mat, true);
                    }
                    else
                    {
                        cv::Mat mat = m_staticFrame.getMat(cv::ACCESS_READ);
                        m_VOTTracker->Initialize(mat, lastRect);
                        m_VOTTracker->Train(mat, true);
                    }

                    inited = true;
                    m_outOfTheFrame = false;
                }
                else
                {
					m_VOTTracker = nullptr;
                    m_outOfTheFrame = true;
                }
            }
            if (!inited && m_VOTTracker)
            {
                constexpr float confThresh = 0.3f;
                cv::Mat mat = currFrame.getMat(cv::ACCESS_READ);
                float confidence = 0;
                cv::RotatedRect newRect = m_VOTTracker->Update(mat, confidence);
                if (confidence > confThresh)
                {
                    m_VOTTracker->Train(mat, false);

					if (newRect.angle > 0.5f)
					{
						m_predictionRect = newRect;
                        m_kalman.Update(newRect.boundingRect(), true);
					}
					else
					{
                        UpdateRRect(brect, m_kalman.Update(newRect.boundingRect(), true));
					}

                    recalcPrediction = false;
                }
            }
        }
        else
        {
            if (m_VOTTracker)
            {
                m_VOTTracker = nullptr;
            }
        }
        break;
    }

    if (recalcPrediction)
    {
        UpdateRRect(m_predictionRect.boundingRect(), m_kalman.Update(region.m_brect, dataCorrect));
    }

    cv::Rect brect = m_predictionRect.boundingRect();
    int dx = Clamp(brect.x, brect.width, currFrame.cols);
    int dy = Clamp(brect.y, brect.height, currFrame.rows);
    m_predictionRect.center.x += dx;
    m_predictionRect.center.y += dy;

    m_outOfTheFrame = (dx != 0) || (dy != 0);

    m_predictionPoint = m_predictionRect.center;
}

///
/// \brief CreateExternalTracker
///
void CTrack::CreateExternalTracker(int channels)
{
    switch (m_externalTrackerForLost)
    {
    case tracking::TrackNone:
        if (m_VOTTracker)
        {
            m_VOTTracker = nullptr;
        }
#ifdef USE_OCV_KCF
        if (m_tracker && !m_tracker.empty())
        {
            m_tracker.release();
        }
#endif
        break;

    case tracking::TrackKCF:
#ifdef USE_OCV_KCF
        if (!m_tracker || m_tracker.empty())
        {
            cv::TrackerKCF::Params params;
			if (channels == 1)
			{
				params.compressed_size = 1;
				params.desc_pca = cv::TrackerKCF::GRAY;
				params.desc_npca = cv::TrackerKCF::GRAY;
			}
			else
			{
				params.compressed_size = 3;
				params.desc_pca = cv::TrackerKCF::CN;
				params.desc_npca = cv::TrackerKCF::CN;
			}
            params.resize = true;
            params.detect_thresh = 0.7f;
#if (((CV_VERSION_MAJOR == 3) && (CV_VERSION_MINOR >= 3)) || (CV_VERSION_MAJOR > 3))
            m_tracker = cv::TrackerKCF::create(params);
#else
            m_tracker = cv::TrackerKCF::createTracker(params);
#endif
        }
#endif
        if (m_VOTTracker)
        {
            m_VOTTracker = nullptr;
        }
        break;

    case tracking::TrackMIL:
#ifdef USE_OCV_KCF
        if (!m_tracker || m_tracker.empty())
        {
            cv::TrackerMIL::Params params;

#if (((CV_VERSION_MAJOR == 3) && (CV_VERSION_MINOR >= 3)) || (CV_VERSION_MAJOR > 3))
            m_tracker = cv::TrackerMIL::create(params);
#else
            m_tracker = cv::TrackerMIL::createTracker(params);
#endif
        }
#endif
        if (m_VOTTracker)
        {
            m_VOTTracker = nullptr;
        }
        break;

    case tracking::TrackMedianFlow:
#ifdef USE_OCV_KCF
        if (!m_tracker || m_tracker.empty())
        {
            cv::TrackerMedianFlow::Params params;

#if (((CV_VERSION_MAJOR == 3) && (CV_VERSION_MINOR >= 3)) || (CV_VERSION_MAJOR > 3))
            m_tracker = cv::TrackerMedianFlow::create(params);
#else
            m_tracker = cv::TrackerMedianFlow::createTracker(params);
#endif
        }
#endif
        if (m_VOTTracker)
        {
            m_VOTTracker = nullptr;
        }
        break;

    case tracking::TrackGOTURN:
#ifdef USE_OCV_KCF
        if (!m_tracker || m_tracker.empty())
        {
            cv::TrackerGOTURN::Params params;

#if (((CV_VERSION_MAJOR == 3) && (CV_VERSION_MINOR >= 3)) || (CV_VERSION_MAJOR > 3))
            m_tracker = cv::TrackerGOTURN::create(params);
#else
            m_tracker = cv::TrackerGOTURN::createTracker(params);
#endif
        }
#endif
        if (m_VOTTracker)
        {
            m_VOTTracker = nullptr;
        }
        break;

    case tracking::TrackMOSSE:
#ifdef USE_OCV_KCF
        if (!m_tracker || m_tracker.empty())
        {
#if (((CV_VERSION_MAJOR == 3) && (CV_VERSION_MINOR > 3)) || (CV_VERSION_MAJOR > 3))
            m_tracker = cv::TrackerMOSSE::create();
#else
            m_tracker = cv::TrackerMOSSE::createTracker();
#endif
        }
#endif
        if (m_VOTTracker)
        {
            m_VOTTracker = nullptr;
        }
        break;

	case tracking::TrackCSRT:
#ifdef USE_OCV_KCF
		if (!m_tracker || m_tracker.empty())
		{
#if (CV_VERSION_MAJOR >= 4)
			cv::TrackerCSRT::Params params;
			params.psr_threshold = 0.04f; // 0.035f;
			if (channels == 1)
			{
				params.use_gray = true;
				params.use_rgb = false;
			}
			else
			{
				params.use_gray = false;
				params.use_rgb = true;
			}
			m_tracker = cv::TrackerCSRT::create(params);
#endif
		}
#endif
        if (m_VOTTracker)
        {
            m_VOTTracker = nullptr;
        }
		break;

    case tracking::TrackDAT:
#ifdef USE_OCV_KCF
		if (m_tracker && !m_tracker.empty())
        {
            m_tracker.release();
        }
#endif
        if (!m_VOTTracker)
        {
            m_VOTTracker = std::unique_ptr<DAT_TRACKER>(new DAT_TRACKER());
        }
        break;

    case tracking::TrackSTAPLE:
#ifdef USE_OCV_KCF
        if (m_tracker && !m_tracker.empty())
        {
            m_tracker.release();
        }
#endif
#ifdef USE_STAPLE_TRACKER
        if (!m_VOTTracker)
        {
            m_VOTTracker = std::unique_ptr<STAPLE_TRACKER>(new STAPLE_TRACKER());
        }
#else
		std::cerr << "Project was compiled without STAPLE tracking!" << std::endl;
#endif
        break;
#if 1
	case tracking::TrackLDES:
#ifdef USE_OCV_KCF
		if (m_tracker && !m_tracker.empty())
		{
			m_tracker.release();
		}
#endif
#ifdef USE_STAPLE_TRACKER
		if (!m_VOTTracker)
		{
			m_VOTTracker = std::unique_ptr<LDESTracker>(new LDESTracker());
		}
#else
		std::cerr << "Project was compiled without STAPLE tracking!" << std::endl;
#endif
		break;
#endif
    }
}

///
/// \brief PointUpdate
/// \param pt
/// \param dataCorrect
///
void CTrack::PointUpdate(
        const Point_t& pt,
        const cv::Size& newObjSize,
        bool dataCorrect,
        const cv::Size& frameSize
        )
{
    m_kalman.GetPointPrediction();

    m_predictionPoint = m_kalman.Update(pt, dataCorrect);

    if (dataCorrect)
    {
        const int a1 = 1;
        const int a2 = 9;
        m_predictionRect.size.width = (a1 * newObjSize.width + a2 * m_predictionRect.size.width) / (a1 + a2);
        m_predictionRect.size.height = (a1 * newObjSize.height + a2 * m_predictionRect.size.height) / (a1 + a2);
    }

    auto Clamp = [](track_t& v, int hi) -> bool
    {
        if (v < 0)
        {
            v = 0;
            return true;
        }
        else if (hi && v > hi - 1)
        {
			v = static_cast<track_t>(hi - 1);
            return true;
        }
        return false;
    };
    m_outOfTheFrame = false;
    m_outOfTheFrame |= Clamp(m_predictionPoint.x, frameSize.width);
    m_outOfTheFrame |= Clamp(m_predictionPoint.y, frameSize.height);
}

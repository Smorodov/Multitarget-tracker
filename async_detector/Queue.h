#pragma once

#include <queue>
#include <deque>
#include <list>
#include <vector>
#include <algorithm>
#include <mutex>
#include <condition_variable>
#include <atomic>

#define SHOW_QUE_LOG 0
#if SHOW_QUE_LOG
///
/// \brief currTime
/// \return
///
inline std::string CurrTime_()
{
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);

    std::ostringstream oss;
    oss << std::put_time(&tm, "%d.%m.%Y %H:%M:%S:");
    return oss.str();
}

#define QUE_LOG std::cout << CurrTime_()
#define QUE_ERR_LOG (std::cerr << CurrTime_())
#endif

struct FrameInfo;
typedef std::shared_ptr<FrameInfo> frame_ptr;
///
/// A threadsafe-queue with Frames
///
class FramesQueue
{
private:
    typedef std::list<frame_ptr> queue_t;

public:
    ///
    /// \brief FramesQueue
    ///
    FramesQueue()
          : m_que(),
          m_mutex(),
          m_cond(),
          m_break(false)
    {}

    FramesQueue(const FramesQueue&) = delete;
    FramesQueue(FramesQueue&&) = delete;
    FramesQueue& operator=(const FramesQueue&) = delete;
    FramesQueue& operator=(FramesQueue&&) = delete;

    ///
    ~FramesQueue(void) = default;

    ///
    /// \brief AddNewFrame
    /// \param frameInfo
    ///
    void AddNewFrame(frame_ptr frameInfo, size_t maxQueueSize)
    {
#if SHOW_QUE_LOG
        QUE_LOG << "AddNewFrame start: " << frameInfo->m_dt << std::endl;
#endif
		std::lock_guard<std::mutex> lock(m_mutex);

		if (!maxQueueSize || (maxQueueSize > 0 && m_que.size() < maxQueueSize))
			m_que.push_back(frameInfo);

#if SHOW_QUE_LOG
		QUE_LOG << "AddNewFrame end: " << frameInfo->m_dt << ", frameInd " << frameInfo->m_frameInd << ", queue size " << m_que.size() << std::endl;
#endif

		m_cond.notify_all();
    }

#if SHOW_QUE_LOG
    ///
    /// \brief PrintQue
    ///
    void PrintQue()
    {
        QUE_LOG << "m_que (" << m_que.size() << "): ";
        size_t i = 0;
        for (auto it : m_que)
        {
            if (it->m_inDetector.load() != FrameInfo::StateNotProcessed && it->m_inTracker.load() != FrameInfo::StateNotProcessed)
                std::cout << i << " d" << it->m_inDetector.load() << " t" << it->m_inTracker.load() << "; ";
            else if (it->m_inDetector.load() != FrameInfo::StateNotProcessed)
                std::cout << i << " d" << it->m_inDetector.load() << "; ";
            else if (it->m_inTracker.load() != FrameInfo::StateNotProcessed)
                std::cout << i << " t" << it->m_inTracker.load() << "; ";
            else
                std::cout << i << "; ";

            ++i;
        }
        std::cout << std::endl;
    }
#endif
    ///
    /// \brief GetLastUndetectedFrame
    /// \return
    ///
    frame_ptr GetLastUndetectedFrame()
    {
#if SHOW_QUE_LOG
        QUE_LOG << "GetLastUndetectedFrame start" << std::endl;
#endif
        std::unique_lock<std::mutex> lock(m_mutex);
        while (m_que.empty() || m_que.back()->m_inDetector.load() != FrameInfo::StateNotProcessed)
        {
            if (m_break.load())
                break;

            m_cond.wait(lock);
            //PrintQue();
        }
        if (!m_break.load())
        {
            frame_ptr frameInfo = m_que.back();
            assert(frameInfo->m_inDetector.load() == FrameInfo::StateNotProcessed);
            assert(frameInfo->m_inTracker.load() == FrameInfo::StateNotProcessed);
            frameInfo->m_inDetector.store(FrameInfo::StateInProcess);

			queue_t::reverse_iterator it = m_que.rbegin();
			for (++it; it != m_que.rend(); ++it)
			{
				if ((*it)->m_inDetector.load() == FrameInfo::StateNotProcessed)
					(*it)->m_inDetector.store(FrameInfo::StateSkipped);
				else
					break;
			}
#if SHOW_QUE_LOG
			PrintQue();
            QUE_LOG << "GetLastUndetectedFrame end: " << frameInfo->m_dt << ", frameInd " << frameInfo->m_frameInd << std::endl;
#endif
            return frameInfo;
        }
        return nullptr;
    }

    ///
    /// \brief SearchUntracked
    /// \return
    ///
    queue_t::iterator SearchUntracked()
    {
        queue_t::iterator res_it = m_que.end();
        for (queue_t::iterator it = m_que.begin(); it != m_que.end(); ++it)
        {
			if ((*it)->m_inDetector.load() == FrameInfo::StateInProcess || (*it)->m_inDetector.load() == FrameInfo::StateNotProcessed)
			{
				break;
			}
            else if ((*it)->m_inTracker.load() == FrameInfo::StateNotProcessed)
            {
                res_it = it;
                break;
            }
        }
        return res_it;
    }

    ///
    /// \brief GetFirstDetectedFrame
    /// \return
    ///
    frame_ptr GetFirstDetectedFrame()
    {
#if SHOW_QUE_LOG
        QUE_LOG << "GetFirstDetectedFrame start" << std::endl;
#endif
        std::unique_lock<std::mutex> lock(m_mutex);
        queue_t::iterator it = SearchUntracked();
        while (it == m_que.end())
        {
            if (m_break.load())
                break;

            m_cond.wait(lock);
            it = SearchUntracked();
            //PrintQue();
        }
        if (!m_break.load())
        {
            frame_ptr frameInfo = *it;
            assert(frameInfo->m_inTracker.load() == FrameInfo::StateNotProcessed);
            assert(frameInfo->m_inDetector.load() != FrameInfo::StateInProcess && frameInfo->m_inDetector.load() != FrameInfo::StateNotProcessed);
            frameInfo->m_inTracker.store(FrameInfo::StateInProcess);
#if SHOW_QUE_LOG
            QUE_LOG << "GetFirstDetectedFrame end: " << frameInfo->m_dt << ", frameInd " << frameInfo->m_frameInd << std::endl;
#endif
            return frameInfo;
        }
        return nullptr;
    }

    ///
    /// \brief GetFirstProcessedFrame
    /// \return
    ///
    frame_ptr GetFirstProcessedFrame()
    {
#if SHOW_QUE_LOG
        QUE_LOG << "GetFirstProcessedFrame start" << std::endl;
#endif
        std::unique_lock<std::mutex> lock(m_mutex);
        while (m_que.empty() || m_que.front()->m_inTracker.load() != FrameInfo::StateCompleted)
        {
            if (m_break.load())
                break;

            m_cond.wait(lock);
            //PrintQue();
        }
        if (!m_break.load())
        {
            frame_ptr frameInfo = std::move(m_que.front());
            m_que.pop_front();
#if SHOW_QUE_LOG
            QUE_LOG << "GetFirstProcessedFrame end: " << frameInfo->m_dt << ", frameInd " << frameInfo->m_frameInd << std::endl;
#endif
            return frameInfo;
        }
        return nullptr;
    }

    ///
    /// \brief Signal
    ///
    void Signal(
#if SHOW_QUE_LOG
		int64 ts
#else
		int64 /*ts*/
#endif
	)
    {
#if SHOW_QUE_LOG
        QUE_LOG << "Signal start:" << ts << std::endl;
#endif
        m_cond.notify_all();
#if SHOW_QUE_LOG
        QUE_LOG << "Signal end: " << ts << std::endl;
#endif
    }

    void SetBreak(bool val)
    {
#if SHOW_QUE_LOG
        QUE_LOG << "SetBreak start:" << val << std::endl;
#endif
        m_break = val;
        Signal(0);
#if SHOW_QUE_LOG
        QUE_LOG << "SetBreak end:" << val << std::endl;
#endif
    }

private:
    queue_t m_que;
    mutable std::mutex m_mutex;
    std::condition_variable m_cond;
    std::atomic<bool> m_break;
};

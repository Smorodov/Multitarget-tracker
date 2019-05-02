#pragma once

#include <queue>
#include <deque>
#include <list>
#include <vector>
#include <algorithm>
#include <mutex>
#include <condition_variable>

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

///
/// A threadsafe-queue
///
template <class T>
class SafeQueue
{
public:
    ///
    SafeQueue(void)
        : m_que()
        , m_mutex()
        , m_cond()
    {
    }

    ///
    virtual ~SafeQueue(void)
    {
    }

protected:
    typedef std::list<T> queue_t;
    queue_t m_que;
    mutable std::mutex m_mutex;
    std::condition_variable m_cond;

    ///
    /// Add an element to the queue
    ///
    void enqueue(T t)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_que.push_back(t);
        m_cond.notify_all();
    }

    ///
    /// Add an element to the queue
    ///
    void enqueue(T t, size_t maxQueueSize)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_que.push(t);

        if (m_que.size() > maxQueueSize)
        {
            m_que.pop();
        }
        m_cond.notify_all();
    }

    ///
    /// Add an element to the queue
    ///
    template<typename RET_V, typename RET_F>
    RET_V enqueue(T t, RET_F && F)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_que.push(t);
        RET_V ret = F(m_que.front());
        m_cond.notify_all();

        return ret;
    }

    ///
    /// Get the "front"-element
    /// If the queue is empty, wait till a element is avaiable
    ///
    T dequeue(void)
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        while (m_que.empty())
        {
            // release lock as long as the wait and reaquire it afterwards
            m_cond.wait(lock);
        }
        T val = m_que.front();
        m_que.pop();

        return val;
    }

    ///
    size_t size()
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        size_t res = m_que.size();
        return res;
    }
};


struct FrameInfo;
///
/// A threadsafe-queue with Frames
///
class FramesQueue : public SafeQueue<FrameInfo>
{
public:
    ///
    /// \brief AddNewFrame
    /// \param frameInfo
    ///
    void AddNewFrame(FrameInfo& frameInfo)
    {
        //QUE_LOG << "AddNewFrame start: " << frameInfo.m_dt << std::endl;
        enqueue(frameInfo);
        //QUE_LOG << "AddNewFrame end: " << frameInfo.m_dt << std::endl;
    }

    ///
    /// \brief PrintQue
    ///
    void PrintQue()
    {
        QUE_LOG << "m_que (" << m_que.size() << "): ";
        size_t i = 0;
        for (queue_t::iterator it = m_que.begin(); it != m_que.end(); ++it)
        {
            if (it->m_inDetector && it->m_inTracker)
            {
                std::cout << i << " d" << it->m_inDetector << " t" << it->m_inTracker << "; ";
            }
            else if (it->m_inDetector)
            {
                std::cout << i << " d" << it->m_inDetector << "; ";
            }
            else if (it->m_inTracker)
            {
                std::cout << i << " t" << it->m_inTracker << "; ";
            }
            else
            {
                std::cout << i << "; ";
            }
            ++i;
        }
        std::cout << std::endl;
    }

    ///
    /// \brief GetLastUndetectedFrame
    /// \return
    ///
    FrameInfo& GetLastUndetectedFrame()
    {
        //QUE_LOG << "GetLastUndetectedFrame start" << std::endl;

        std::unique_lock<std::mutex> lock(m_mutex);
        while (m_que.empty() || m_que.back().m_inDetector)
        {
            m_cond.wait(lock);
            //PrintQue();
        }
        FrameInfo& frameInfo = m_que.back();
        frameInfo.m_inDetector = 1;

        //QUE_LOG << "GetLastUndetectedFrame end: " << frameInfo.m_dt << std::endl;

        return frameInfo;
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
            if (it->m_inDetector != 1 && it->m_inTracker == 0)
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
    FrameInfo& GetFirstDetectedFrame()
    {
        //QUE_LOG << "GetFirstDetectedFrame start" << std::endl;

        std::unique_lock<std::mutex> lock(m_mutex);
        queue_t::iterator it = SearchUntracked();
        while (it == m_que.end())
        {
            m_cond.wait(lock);
            it = SearchUntracked();
            //PrintQue();
        }
        FrameInfo& frameInfo = *it;
		frameInfo.m_inTracker = 1;

        //QUE_LOG << "GetFirstDetectedFrame end: " << frameInfo.m_dt << std::endl;
        return frameInfo;
    }

    ///
    /// \brief GetFirstProcessedFrame
    /// \return
    ///
    FrameInfo GetFirstProcessedFrame()
    {
        //QUE_LOG << "GetFirstProcessedFrame start" << std::endl;

        std::unique_lock<std::mutex> lock(m_mutex);
        while (m_que.empty() || m_que.front().m_inTracker != 2)
        {
            m_cond.wait(lock);
            //PrintQue();
        }
        FrameInfo frameInfo = m_que.front();
        m_que.pop_front();

        //QUE_LOG << "GetFirstProcessedFrame end: " << frameInfo.m_dt << std::endl;

        return frameInfo;
    }

    ///
    /// \brief Signal
    ///
    void Signal(int64 ts)
    {
        //QUE_LOG << "Signal start:" << ts << std::endl;
        m_cond.notify_all();
        //QUE_LOG << "Signal end: " << ts << std::endl;
    }

private:
};

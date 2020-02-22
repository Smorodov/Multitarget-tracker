#pragma once
#include "defines.h"
#include "HungarianAlg/HungarianAlg.h"

///
/// \brief The SPSettings struct
///
struct SPSettings
{
    track_t m_distThres = 0.8f;
    size_t m_maxHistory = 10;
};

///
/// \brief The ShortPathCalculator class
///
class ShortPathCalculator
{
public:
    ShortPathCalculator(const SPSettings& settings)
        : m_settings(settings)
    {
    }
    virtual ~ShortPathCalculator()
    {
    }

    virtual void Solve(const distMatrix_t& costMatrix, size_t N, size_t M, assignments_t& assignment, track_t maxCost) = 0;
protected:
    SPSettings m_settings;
};

///
/// \brief The SPHungrian class
///
class SPHungrian : public ShortPathCalculator
{
public:
    SPHungrian(const SPSettings& settings)
        : ShortPathCalculator(settings)
    {
    }

    void Solve(const distMatrix_t& costMatrix, size_t N, size_t M, assignments_t& assignment, track_t /*maxCost*/)
    {
        m_solver.Solve(costMatrix, N, M, assignment, AssignmentProblemSolver::optimal);
    }

private:
    AssignmentProblemSolver m_solver;
};

///
/// \brief The SPBipart class
///
class SPBipart : public ShortPathCalculator
{
public:
    SPBipart(const SPSettings& settings)
        : ShortPathCalculator(settings)
    {
    }

    void Solve(const distMatrix_t& costMatrix, size_t N, size_t M, assignments_t& assignment, track_t maxCost);
};

///
/// \brief The SPmuSSP class
///
class SPmuSSP : public ShortPathCalculator
{
public:
    SPmuSSP(const SPSettings& settings)
        : ShortPathCalculator(settings)
    {
    }

    void Solve(const distMatrix_t& costMatrix, size_t N, size_t M, assignments_t& assignment, track_t maxCost);

private:

    ///
    /// \brief The Node struct
    ///
    struct Node
    {
        std::vector<std::pair<size_t, track_t>> m_arcs;
        void Add(size_t ind, track_t weight)
        {
            m_arcs.emplace_back(ind, weight);
        }
        void Resize(size_t count)
        {
            m_arcs.resize(count);
        }
    };
    ///
    /// \brief The Layer struct
    ///
    struct Layer
    {
        std::vector<Node> m_nodes;
        size_t m_arcsCount = 0;

        void Resize(size_t count)
        {
            m_nodes.resize(count);
        }
        const Node& Back() const
        {
            return m_nodes.back();
        }
        Node& Back()
        {
            return m_nodes.back();
        }
        size_t Size() const
        {
            return m_nodes.size();
        }
        const Node& operator[](size_t ind) const
        {
            return m_nodes[ind];
        }
        Node& operator[](size_t ind)
        {
            return m_nodes[ind];
        }
    };

    std::deque<Layer> m_detects;
};

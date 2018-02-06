#include "Ctracker.h"
#include "HungarianAlg.h"

#include <GTL/GTL.h>
#include "mygraph.h"
#include "mwbmatching.h"
#include "tokenise.h"

#include "hough3d/hough.h"
#include <Eigen/Dense>

// ---------------------------------------------------------------------------
// Tracker. Manage tracks. Create, remove, update.
// ---------------------------------------------------------------------------
CTracker::CTracker(
        bool useLocalTracking,
        tracking::DistType distType,
        tracking::KalmanType kalmanType,
        tracking::FilterGoal filterGoal,
        tracking::LostTrackType lostTrackType,
        tracking::MatchType matchType,
        track_t dt_,
        track_t accelNoiseMag_,
        track_t dist_thres_,
        size_t maximum_allowed_skipped_frames_,
        size_t max_trace_length_
        )
    :
      m_useLocalTracking(useLocalTracking),
      m_distType(distType),
      m_kalmanType(kalmanType),
      m_filterGoal(filterGoal),
      m_lostTrackType(lostTrackType),
      m_matchType(matchType),
      dt(dt_),
      accelNoiseMag(accelNoiseMag_),
      dist_thres(dist_thres_),
      maximum_allowed_skipped_frames(maximum_allowed_skipped_frames_),
      max_trace_length(max_trace_length_),
      NextTrackID(0),
      m_useHough3D(true)
{
}

// ---------------------------------------------------------------------------
//
// ---------------------------------------------------------------------------
CTracker::~CTracker(void)
{
}

//
// orthogonal least squares fit with libeigen
// rc = largest eigenvalue
//
int orthogonal_LSQ(const PointCloud &pc, Vector3d* a, Vector3d* b)
{
    // anchor point is mean value
    *a = pc.meanValue();

    // copy points to libeigen matrix
    Eigen::MatrixXf points = Eigen::MatrixXf::Constant(pc.points.size(), 3, 0);
    for (int i = 0; i < points.rows(); i++)
    {
        points(i, 0) = pc.points.at(i).x;
        points(i, 1) = pc.points.at(i).y;
        points(i, 2) = pc.points.at(i).z;
    }

    // compute scatter matrix ...
    Eigen::MatrixXf centered = points.rowwise() - points.colwise().mean();
    Eigen::MatrixXf scatter = (centered.adjoint() * centered);

    // ... and its eigenvalues and eigenvectors
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> eig(scatter);
    Eigen::MatrixXf eigvecs = eig.eigenvectors();

    // we need eigenvector to largest eigenvalue
    // libeigen yields it as LAST column
    b->x = eigvecs(0, 2);
    b->y = eigvecs(1, 2);
    b->z = eigvecs(2, 2);

    int rc = eig.eigenvalues()(2);
    return rc;
}

// ---------------------------------------------------------------------------
//
// ---------------------------------------------------------------------------
void CTracker::Update(
        const regions_t& regions,
        cv::UMat grayFrame,
        float fps
        )
{
    if (m_prevFrame.size() == grayFrame.size())
    {
        if (m_useLocalTracking)
        {
            m_localTracker.Update(tracks, m_prevFrame, grayFrame);
        }
    }

    if (m_useHough3D)
    {
        UpdateHough3D(regions, grayFrame, fps);
    }
    else
    {
        UpdateHungrian(regions, grayFrame, fps);
    }

    grayFrame.copyTo(m_prevFrame);
}

// ---------------------------------------------------------------------------
//
// ---------------------------------------------------------------------------
void CTracker::UpdateHungrian(
        const regions_t& regions,
        cv::UMat grayFrame,
        float /*fps*/
        )
{
    size_t N = tracks.size();		// треки
    size_t M = regions.size();	// детекты

    assignments_t assignment(N, -1); // назначения

    if (!tracks.empty())
    {
        // Матрица расстояний от N-ного трека до M-ного детекта.
        distMatrix_t Cost(N * M);

        // -----------------------------------
        // Треки уже есть, составим матрицу расстояний
        // -----------------------------------
        const track_t maxPossibleCost = grayFrame.cols * grayFrame.rows;
        track_t maxCost = 0;
        switch (m_distType)
        {
        case tracking::DistCenters:
            for (size_t i = 0; i < tracks.size(); i++)
            {
                for (size_t j = 0; j < regions.size(); j++)
                {
                    auto dist = tracks[i]->CheckType(regions[j].m_type) ? tracks[i]->CalcDist((regions[j].m_rect.tl() + regions[j].m_rect.br()) / 2) : maxPossibleCost;
                    Cost[i + j * N] = dist;
                    if (dist > maxCost)
                    {
                        maxCost = dist;
                    }
                }
            }
            break;

        case tracking::DistRects:
            for (size_t i = 0; i < tracks.size(); i++)
            {
                for (size_t j = 0; j < regions.size(); j++)
                {
                    auto dist = tracks[i]->CheckType(regions[j].m_type) ? tracks[i]->CalcDist(regions[j].m_rect) : maxPossibleCost;
                    Cost[i + j * N] = dist;
                    if (dist > maxCost)
                    {
                        maxCost = dist;
                    }
                }
            }
            break;

        case tracking::DistJaccard:
            for (size_t i = 0; i < tracks.size(); i++)
            {
                for (size_t j = 0; j < regions.size(); j++)
                {
                    auto dist = tracks[i]->CheckType(regions[j].m_type) ? tracks[i]->CalcDistJaccard(regions[j].m_rect) : 1;
                    Cost[i + j * N] = dist;
                    if (dist > maxCost)
                    {
                        maxCost = dist;
                    }
                }
            }
            break;
        }
        // -----------------------------------
        // Solving assignment problem (tracks and predictions of Kalman filter)
        // -----------------------------------
        if (m_matchType == tracking::MatchHungrian)
        {
            AssignmentProblemSolver APS;
            APS.Solve(Cost, N, M, assignment, AssignmentProblemSolver::optimal);
        }
        else
        {
            MyGraph G;
            G.make_directed();

            std::vector<node> nodes(N + M);

            for (size_t i = 0; i < nodes.size(); ++i)
            {
                nodes[i] = G.new_node();
            }

            edge_map<int> weights(G, 100);
            for (size_t i = 0; i < tracks.size(); i++)
            {
                bool hasZeroEdge = false;

                for (size_t j = 0; j < regions.size(); j++)
                {
                    track_t currCost = Cost[i + j * N];

                    edge e = G.new_edge(nodes[i], nodes[N + j]);

                    if (currCost < dist_thres)
                    {
                        int weight = maxCost - currCost + 1;
                        G.set_edge_weight(e, weight);
                        weights[e] = weight;
                    }
                    else
                    {
                        if (!hasZeroEdge)
                        {
                            G.set_edge_weight(e, 0);
                            weights[e] = 0;
                        }
                        hasZeroEdge = true;
                    }
                }
            }

            edges_t L = MAX_WEIGHT_BIPARTITE_MATCHING(G, weights);
            for (edges_t::iterator it = L.begin(); it != L.end(); ++it)
            {
                node a = it->source();
                node b = it->target();
                assignment[b.id()] = a.id() - N;
            }
        }

        // -----------------------------------
        // clean assignment from pairs with large distance
        // -----------------------------------
        for (size_t i = 0; i < assignment.size(); i++)
        {
            if (assignment[i] != -1)
            {
                if (Cost[i + assignment[i] * N] > dist_thres)
                {
                    assignment[i] = -1;
                    tracks[i]->m_skippedFrames++;
                }
            }
            else
            {
                // If track have no assigned detect, then increment skipped frames counter.
                tracks[i]->m_skippedFrames++;
            }
        }

        // -----------------------------------
        // If track didn't get detects long time, remove it.
        // -----------------------------------
        for (int i = 0; i < static_cast<int>(tracks.size()); i++)
        {
            if (tracks[i]->m_skippedFrames > maximum_allowed_skipped_frames)
            {
                tracks.erase(tracks.begin() + i);
                assignment.erase(assignment.begin() + i);
                i--;
            }
        }
    }

    // -----------------------------------
    // Search for unassigned detects and start new tracks for them.
    // -----------------------------------
    for (size_t i = 0; i < regions.size(); ++i)
    {
        if (find(assignment.begin(), assignment.end(), i) == assignment.end())
        {
            tracks.push_back(std::make_unique<CTrack>(regions[i],
                                                      m_kalmanType,
                                                      dt,
                                                      accelNoiseMag,
                                                      NextTrackID++,
                                                      m_filterGoal == tracking::FilterRect,
                                                      m_lostTrackType));
        }
    }

    // Update Kalman Filters state

    for (size_t i = 0; i < assignment.size(); i++)
    {
        // If track updated less than one time, than filter state is not correct.

        if (assignment[i] != -1) // If we have assigned detect, then update using its coordinates,
        {
            tracks[i]->m_skippedFrames = 0;
            tracks[i]->Update(regions[assignment[i]], true, max_trace_length, m_prevFrame, grayFrame);
        }
        else				     // if not continue using predictions
        {
            tracks[i]->Update(CRegion(), false, max_trace_length, m_prevFrame, grayFrame);
        }
    }
}

// ---------------------------------------------------------------------------
//
// ---------------------------------------------------------------------------
void CTracker::UpdateHough3D(
        const regions_t& regions,
        cv::UMat grayFrame,
        float fps
        )
{
    std::vector<Point_t> points3d;

    for (const auto& region : regions)
    {
        points3d.push_back(Point_t((region.m_rect.tl() + region.m_rect.br()) / 2));
    }

    m_points3D.push_back(points3d);
    if (m_points3D.size() > Hough3DTimeline)
    {
        m_points3D.pop_front();
    }

    if (m_points3D.size() != Hough3DTimeline)
    {
        return;
    }

    track_t dTime = std::max(1.f, 1000.f / fps);

    size_t allPoints = 0;
    for (const auto& points : m_points3D)
    {
        allPoints += points.size();
    }

    if (allPoints > 2)
    {
        PointCloud X;
        X.points.reserve(allPoints);

        track_t ptTime = 0;
        for (const auto& points : m_points3D)
        {
            for (const auto& pt : points)
            {
                X.points.push_back(Vector3d(ptTime, pt.x, pt.y));
            }

            ptTime += dTime;
        }

        // center cloud and compute new bounding box
        Vector3d minP;
        Vector3d maxP;
        X.getMinMax3D(&minP, &maxP);
        track_t d = (maxP - minP).norm();
        X.shiftToOrigin();
        Vector3d minPshifted;
        Vector3d maxPshifted;
        X.getMinMax3D(&minPshifted, &maxPshifted);

        // estimate size of Hough space
        // number of icosahedron subdivisions for direction discretization
        int granularity = 4;
        int num_directions[7] = {12, 21, 81, 321, 1281, 5121, 20481};

        track_t opt_dx = d / 64.0;

        track_t num_x = floor(d / opt_dx + 0.5f);
        track_t num_cells = num_x * num_x * num_directions[granularity];

        std::unique_ptr<Hough> hough = std::make_unique<Hough>(minPshifted, maxPshifted, opt_dx, granularity);
        hough->add(X);

        // iterative Hough transform (Algorithm 1 in IPOL paper)
        PointCloud Y;	// points close to line
        size_t opt_minvotes = Hough3DTimeline / 2;
        std::deque<std::pair<Vector3d, Vector3d>> lines;
        do
        {
            Vector3d a; // anchor point of line
            Vector3d b; // direction of line

            hough->subtract(Y); // do it here to save one call

            hough->getLine(&a, &b);

            X.pointsCloseToLine(a, b, opt_dx, &Y);

            if (!orthogonal_LSQ(Y, &a, &b))
                break;

            X.pointsCloseToLine(a, b, opt_dx, &Y);

            if (Y.points.size() < opt_minvotes)
                break;

            if (!orthogonal_LSQ(Y, &a, &b))
                break;

            a = a + X.shift;

            lines.push_back(std::make_pair(a, b));

            X.removePoints(Y);

        } while (X.points.size() > 1);

        if (1)
        {
            std::cout << "Hough3D: points = " << allPoints << ", lines = " << lines.size() << std::endl;

            cv::Mat dbgLines;
            cv::cvtColor(grayFrame.getMat(cv::ACCESS_READ), dbgLines, CV_GRAY2BGR);

            for (const auto& points : m_points3D)
            {
                for (const auto& pt : points)
                {
                    cv::circle(dbgLines, cv::Point(pt.x, pt.y), 4, cv::Scalar(0, 0, 255));

                }
            }

            for (auto line : lines)
            {
                track_t minT = line.first.x;
                track_t maxT = line.first.x + dTime * (Hough3DTimeline - 1) * line.second.x;
                track_t dt = fabs(maxT - minT);

                //std::cout << "a = " << line.first << ", b = " << line.second << ", dt = " << dt << std::endl;

                cv::line(dbgLines,
                         cv::Point(line.first.y, line.first.z),
                         cv::Point(line.first.y + dt * line.second.y, line.first.z + dt * line.second.z),
                         cv::Scalar(255, 0, 0),
                         2);
            }
            cv::imshow("hough3d", dbgLines);
        }
    }
}
